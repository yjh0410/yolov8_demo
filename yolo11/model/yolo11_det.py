import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule, C2PSA, YoloStage, SPPF, Upsample, Concat, Detect
except:
    from  modules import ConvModule, C2PSA, YoloStage, SPPF, Upsample, Concat, Detect


# Yolo11 model for the general object detector
class Yolo11Detector(nn.Module):
    def __init__(self, model_scale: str = "l", num_classes: int = 80):
        super().__init__()
        self.model_scale = model_scale
        self.num_classes = num_classes

        # --------- Model scaling factors ---------
        if model_scale == "n":
            width, depth, ratio = 0.25, 0.50, 2.0
        if model_scale == "s":
            width, depth, ratio = 0.50, 0.50, 2.0
        if model_scale == "m":
            width, depth, ratio = 1.00, 0.50, 1.0
        if model_scale == "l":
            width, depth, ratio = 1.00, 1.00, 1.0
        if model_scale == "x":
            width, depth, ratio = 1.50, 1.00, 1.0

        # --------- Backbone ---------
        self.model = nn.ModuleList([
            ConvModule(3, int(64 * width), kernel_size=3, stride=2), # layer-0
            ConvModule(int(64 * width), int(128 * width), kernel_size=3, stride=2), # layer-1
            YoloStage(in_dim     = int(128 * width),
                      out_dim    = int(256 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.25,
                      use_c3k    = False if model_scale in "ns" else True,
                      ), # layer-2
            ConvModule(int(256 * width), int(256 * width), kernel_size=3, stride=2), # layer-3
            YoloStage(in_dim     = int(256 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.25,
                      use_c3k    = False if model_scale in "ns" else True,
                      ), # layer-4
            ConvModule(int(512 * width), int(512 * width), kernel_size=3, stride=2), # layer-5
            YoloStage(in_dim     = int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = True,
                      ), # layer-6
            ConvModule(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2), # layer-7
            YoloStage(in_dim     = int(512 * width * ratio),
                      out_dim    = int(512 * width * ratio),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = True,
                      ), # layer-8
            SPPF(in_dim  = int(512 * width * ratio),
                 out_dim = int(512 * width * ratio),
                 spp_pooling_size = 5,
                 neck_expand_ratio = 0.5,
                 ), # layer-9
            C2PSA(in_dim = int(512 * width * ratio),
                  out_dim = int(512 * width * ratio),
                  num_blocks = int(2 * depth),
                  expansion = 0.5,
                  ), # layer-10
        ])

        # --------- PaFPN ---------
        self.model.extend([
            Upsample(scale_factor=2.0, mode='nearest'), # layer-11
            Concat(dim=1), # layer-12
            YoloStage(in_dim     = int(512 * width * ratio) + int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = False if model_scale in "ns" else True,
                      ),  # layer-13
            Upsample(scale_factor=2.0, mode='nearest'), # layer-14
            Concat(dim=1), # layer-15
            YoloStage(in_dim     = int(512 * width) + int(512 * width),
                      out_dim    = int(256 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = False if model_scale in "ns" else True,
                      ),  # layer-16
            ConvModule(int(256 * width), int(256 * width), kernel_size=3, stride=2), # layer-17
            Concat(dim=1), # layer-18
            YoloStage(in_dim     = int(256 * width) + int(512 * width),
                      out_dim    = int(512 * width),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = False if model_scale in "ns" else True,
                      ),  # layer-19
            ConvModule(int(512 * width), int(512 * width), kernel_size=3, stride=2), # layer-20
            Concat(dim=1), # layer-21
            YoloStage(in_dim     = int(512 * width) + int(512 * width * ratio),
                      out_dim    = int(512 * width * ratio),
                      num_blocks = int(2 * depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = True,
                      ),  # layer-22
        ])

        # --------- Task Head ---------
        self.model.append(
            Detect(num_classes=self.num_classes, ch=[int(256 * width), int(512 * width), int(512 * width * ratio)]),  # layer-23
        )

    def _forward_impl(self, x):
        # ------------- Backbone -------------
        pyramid_feats = []
        for i, layer in enumerate(self.model[:11]):
            x = layer(x)
            if i == 4:  # P3 feature
                pyramid_feats.append(x)
            if i == 6:  # P4 feature
                pyramid_feats.append(x)
            if i == 10:  # P5 feature (after SPPF & C2PSA)
                pyramid_feats.append(x)

        # ------------------ Top down FPN ------------------
        c3, c4, c5 = pyramid_feats
        ## P5 -> P4
        p5_up = self.model[11](c5)
        p4 = self.model[13](self.model[12]([p5_up, c4]))

        ## P4 -> P3
        p4_up = self.model[14](p4)
        p3 = self.model[16](self.model[15]([p4_up, c3]))

        # ------------------ Bottom up FPN ------------------
        ## p3 -> P4
        p3_ds = self.model[17](p3)
        p4 = self.model[19](self.model[18]([p3_ds, p4]))

        ## P4 -> 5
        p4_ds = self.model[20](p4)
        p5 = self.model[22](self.model[21]([p4_ds, c5]))

        pyramid_feats = [p3, p4, p5] # [P3, P4, P5]

        # ------------------ Task head ------------------
        output = self.model[23](pyramid_feats)

        return output

    def forward(self, x):
        return self._forward_impl(x)
    

if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)
    model = Yolo11Detector(model_scale="s")
    print(model.state_dict().keys())

    output = model(x)
    print(output.shape)
