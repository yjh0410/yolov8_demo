import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ----------------- CNN basic components -----------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim,        # in channels
                 out_dim,       # out channels 
                 kernel_size=1, # kernel size 
                 stride=1,      # padding
                 groups=1,      # groups
                 use_act: bool = True,
                ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size//2, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :List  = [3, 3],
                 shortcut    :bool  = False,
                 ) -> None:
        super(Bottleneck, self).__init__()
        # ----------------- Network setting -----------------
        self.cv1 = ConvModule(in_dim,  out_dim, kernel_size=kernel_size[0], stride=1)
        self.cv2 = ConvModule(out_dim, out_dim, kernel_size=kernel_size[1], stride=1)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, spp_pooling_size: int = 5, neck_expand_ratio:float = 0.5):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = round(in_dim * neck_expand_ratio)
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1, stride=1)
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=spp_pooling_size, stride=1, padding=spp_pooling_size // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Upsample(nn.Module):
    def __init__(self, scale_factor=2.0, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def _forward_impl(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, dim=self.dim)

class YoloStage(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks :int  = 1,
                 shortcut   :bool = False,
                 ) -> None:
        super(YoloStage, self).__init__()
        inter_dim = out_dim // 2
        self.cv1 = ConvModule(in_dim, inter_dim * 2, kernel_size=1, stride=1)
        self.cv2 = ConvModule((2 + num_blocks) * inter_dim, out_dim, kernel_size=1, stride=1)
        self.m = nn.ModuleList([Bottleneck(inter_dim, inter_dim, [3, 3], shortcut) for _ in range(num_blocks)])

    def _forward_impl(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.cv1(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.cv2(torch.cat(out, dim=1))

        return out

    def forward(self, x):
        return self._forward_impl(x)


# ----------------- YOLO detection head -----------------
class DFL(nn.Module):
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        # reshape: [bs, c, m] -> [bs, 4, reg_max, a] -> [bs, reg_max, 4, a]
        x = x.view(b, 4, self.c1, a).transpose(2, 1)
        x = F.softmax(x, dim=1)
        return self.conv(x).view(b, 4, a) # [bs, 4, a]

class Detect(nn.Module):
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, num_classes=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = [8, 16, 32]  # strides computed during build
        
        reg_dim = max((16, ch[0] // 4, self.reg_max * 4))
        cls_dim = max(ch[0], min(self.num_classes, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(ConvModule(x, reg_dim, kernel_size=3, stride=1),
                          ConvModule(reg_dim, reg_dim, kernel_size=3, stride=1),
                          nn.Conv2d(reg_dim, 4 * self.reg_max, 1))
                          for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(ConvModule(x, cls_dim, kernel_size=3, stride=1),
                          ConvModule(cls_dim, cls_dim, kernel_size=3, stride=1),
                          nn.Conv2d(cls_dim, self.num_classes, 1))
                          for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # cls & reg head outputs
        for i in range(self.nl):
            # shape: [bs, nc + nb, h, w]
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # create anchors and stride tensors
        self.anchors, self.strides = (x.transpose(0, 1) for x in self.make_anchors(x, self.stride, 0.5))
        self.shape = shape

        # split cls output and reg outputs
        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)

        # decode box
        dist = self.dfl(box)
        lt, rb = torch.chunk(dist, chunks=2, dim=1)
        x1y1 = self.anchors.unsqueeze(0) - lt
        x2y2 = self.anchors.unsqueeze(0) + rb
        dbox = torch.cat([x1y1, x2y2], dim=1) # xyxy bbox

        dbox = dbox * self.strides.unsqueeze(1)

        out = torch.cat([dbox, cls.sigmoid()], dim=1)

        return out


# ----------------- YOLO segmentation head -----------------
class Proto(nn.Module):
    def __init__(self, in_dim: int, inter_dim: int = 256, out_dim: int = 32):
        super().__init__()
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=3)
        self.upsample = nn.ConvTranspose2d(inter_dim, inter_dim, kernel_size=2, stride=2, padding=0, bias=True)
        self.cv2 = ConvModule(inter_dim, inter_dim, kernel_size=3)
        self.cv3 = ConvModule(inter_dim, out_dim)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Segment(Detect):
    def __init__(self, num_classes: int = 80, num_masks: int = 32, npr: int = 256, ch=()):
        super().__init__(num_classes, ch)
        self.num_masks = num_masks    # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.num_masks)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.num_masks)
        self.cv4 = nn.ModuleList(
            nn.Sequential(ConvModule(x, c4, kernel_size=3),
                          ConvModule(c4, c4, kernel_size=3),
                          nn.Conv2d(c4, self.num_masks, kernel_size=1))
                          for x in ch)

    def forward(self, x):
        protos = self.proto(x[0])  # mask protos: [bs, nm, h, w]

        # ---------- Mask head output ----------
        seg_out = []
        for i in range(self.nl):
            # [bs, nm, h, w] -> [bs, nm, hw]
            mc_i = self.cv4[i](x[i]).flatten(2)
            seg_out.append(mc_i)
        # [bs, nm, m]
        seg_out = torch.cat(seg_out, dim=2)

        # ---------- BBox head output ----------
        # [bs, 4+nc, m]
        det_out = self.detect(self, x)

        # [bs, 4 + nc + nm, m]
        output = torch.cat([det_out, seg_out], dim=1)

        return output, protos
