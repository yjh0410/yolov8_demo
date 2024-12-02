import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule, YoloStage, SPPF, Upsample, Concat, Segment
except:
    from  modules import ConvModule, YoloStage, SPPF, Upsample, Concat, Segment


# Yolov8 model for the general instance segmentation
class Yolov8Segmentor(nn.Module):
    def __init__(self,
                 model_scale: str = "l",
                 num_classes: int = 80,
                 num_masks:   int = 32,
                 ):
        super().__init__()
        self.model_scale = model_scale
        self.num_classes = num_classes
        self.num_masks = num_masks

        # --------- Model scaling factors ---------
        if model_scale == "n":
            width, depth, ratio = 0.25, 0.34, 2.0
        if model_scale == "s":
            width, depth, ratio = 0.50, 0.34, 2.0
        if model_scale == "m":
            width, depth, ratio = 0.75, 0.67, 1.5
        if model_scale == "l":
            width, depth, ratio = 1.00, 1.00, 1.0
        if model_scale == "x":
            width, depth, ratio = 1.25, 1.34, 1.0

        # --------- Backbone ---------
        self.model = nn.ModuleList([
            ConvModule(3, int(64 * width), kernel_size=3, stride=2), # layer-0
            ConvModule(int(64 * width), int(128 * width), kernel_size=3, stride=2), # layer-1
            YoloStage(in_dim     = int(128 * width),
                      out_dim    = int(128 * width),
                      num_blocks = int(3 * depth),
                      shortcut   = True,
                      ), # layer-2
            ConvModule(int(128 * width), int(256 * width), kernel_size=3, stride=2), # layer-3
            YoloStage(in_dim     = int(256 * width),
                      out_dim    = int(256 * width),
                      num_blocks = int(6 * depth),
                      shortcut   = True,
                      ), # layer-4
            ConvModule(int(256 * width), int(512 * width), kernel_size=3, stride=2), # layer-5
            YoloStage(in_dim     = int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(6 * depth),
                      shortcut   = True,
                      ), # layer-6
            ConvModule(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2), # layer-7
            YoloStage(in_dim     = int(512 * width * ratio),
                      out_dim    = int(512 * width * ratio),
                      num_blocks = int(3 * depth),
                      shortcut   = True,
                      ), # layer-8
            SPPF(int(512 * width * ratio), int(512 * width * ratio), spp_pooling_size=5, neck_expand_ratio=0.5), # layer-9
        ])

        # --------- PaFPN ---------
        self.model.extend([
            Upsample(scale_factor=2.0, mode='nearest'), # layer-10
            Concat(dim=1), # layer-11
            YoloStage(in_dim     = int(512 * width * ratio) + int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(3 * depth),
                      shortcut   = False,
                      ),  # layer-12
            Upsample(scale_factor=2.0, mode='nearest'), # layer-13
            Concat(dim=1), # layer-14
            YoloStage(in_dim     = int(512 * width) + int(256 * width),
                      out_dim    = int(256 * width),
                      num_blocks = int(3 * depth),
                      shortcut   = False,
                      ),  # layer-15
            ConvModule(int(256 * width), int(256 * width), kernel_size=3, stride=2), # layer-16
            Concat(dim=1), # layer-14
            YoloStage(in_dim     = int(256 * width) + int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(3 * depth),
                      shortcut   = False,
                      ),  # layer-18
            ConvModule(int(512 * width), int(512 * width), kernel_size=3, stride=2), # layer-19
            Concat(dim=1), # layer-20
            YoloStage(in_dim     = int(512 * width) + int(512 * width * ratio),
                      out_dim    = int(512 * width * ratio),
                      num_blocks = int(3 * depth),
                      shortcut   = False,
                      ),  # layer-21
        ])

        # --------- Task Head ---------
        self.model.append(
            Segment(num_classes=self.num_classes,
                    num_masks=32,
                    npr=int(256 * width),
                    ch=[int(256 * width), int(512 * width), int(512 * width * ratio)],
                    ),  # layer-22
        )

    def _forward_impl(self, x):
        # ------------- Backbone -------------
        pyramid_feats = []
        for i, layer in enumerate(self.model[:10]):
            x = layer(x)
            if i == 4:  # P3 feature
                pyramid_feats.append(x)
            if i == 6:  # P4 feature
                pyramid_feats.append(x)
            if i == 9:  # P5 feature (after SPPF)
                pyramid_feats.append(x)

        # ------------------ Top down FPN ------------------
        c3, c4, c5 = pyramid_feats
        ## P5 -> P4
        p5_up = self.model[10](c5)
        p4 = self.model[12](self.model[11]([p5_up, c4]))

        ## P4 -> P3
        p4_up = self.model[13](p4)
        p3 = self.model[15](self.model[14]([p4_up, c3]))

        # ------------------ Bottom up FPN ------------------
        ## p3 -> P4
        p3_ds = self.model[16](p3)
        p4 = self.model[18](self.model[17]([p3_ds, p4]))

        ## P4 -> 5
        p4_ds = self.model[19](p4)
        p5 = self.model[21](self.model[20]([p4_ds, c5]))

        pyramid_feats = [p3, p4, p5] # [P3, P4, P5]

        # ------------------ Task head ------------------
        # output: [bs, 4 + nc + nm, hw]; protos: [bs, nm, h, w]
        output, protos = self.model[22](pyramid_feats)

        return output, protos

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)
    model = Yolov8Segmentor(model_scale="s")
    print(model)

    output, protos = model(x)
    print(output.shape, protos.shape)
