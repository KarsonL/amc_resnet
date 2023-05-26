# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_class,  profile='normal'):
        super(MobileNet, self).__init__()

        # original
        if profile == 'normal':
            in_planes = 32
            cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        # 0.5 AMC
        elif profile == '0.5flops':
            # in_planes = 24 # 官方
            # cfg = [48, (96, 2), 80, (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752]
            # in_planes = 32 # 0.5flops_cifar10_1
            # cfg = [56, (104,2), 104, (200,2), 200, (368,2), 360, 328, 304, 272, 392, (400,2), 464]
            # in_planes = 24  # 0.5flops_cifar10_2_1
            # cfg = [56, (96,2), 104, (192,2), 200, (424,2), 416, 392, 368, 312, 224, (216,2), 208]
            # in_planes = 32  # 0.5flops_cifar10_1_td3
            # cfg = [56, (104,2), 104, (200,2), 200, (360,2), 344, 328, 312, 296, 280, (496,2), 568]
            # in_planes = 32  # 0.5flops_cifar10_1_ddpg
            # cfg = [56, (104,2), 96, (192,2), 192, (368,2), 344, 328, 312, 296, 416, (688,2), 208]
            # in_planes = 32  # 0.5flops_cifar10_1_ddpg_2
            # cfg = [64, (128,2), 64, (184,2), 216, (416,2), 344, 400, 376, 312, 200, (208,2), 208]
            # in_planes = 32  # 0.5flops_cifar10_1_td3_2
            # cfg = [56, (96,2), 96, (192,2), 184, (360,2), 368, 360, 352, 336, 336, (440,2), 208]
            # in_planes = 32 #DDPG_6
            # cfg = [56, (112,2), 104, (200,2), 200, (360,2), 384, 344, 320, 304, 256, (432,2), 208]
            # in_planes = 24 #TD3_6
            # cfg = [48, (96,2), 88, (184,2), 168, (344,2), 376, 376, 360, 400, 384, (752,2), 208]
            # in_planes = 32 #DDPG_6_1 #0.5 + 0.5flops
            # cfg = [48, (80,2), 72, (128,2), 128, (248,2), 264, 224, 184, 168, 136, (192,2), 168]
            # in_planes = 24 #DDPG_6_1 #0.25
            # cfg = [40, (72,2), 64, (136,2), 128, (256,2), 256, 264, 272, 248, 104, (208,2), 208]
            # in_planes = 8 #DDPG_9_flop_reward
            # cfg = [24, (48,2), 40, (88,2), 112, (208,2), 224, 224, 240, 248, 144, (104,2), 104]
            # in_planes = 32 #DDPG_9_reward
            # cfg = [56, (104,2), 96, (184,2), 184, (360,2), 360, 352, 344, 328, 312, (584,2), 312]
            # in_planes = 32#DDPG_10_env2
            # cfg = [56, (96,2), 96, (184,2), 184, (360,2), 360, 352, 360, 352, 360, (480,2), 208]
            # in_planes = 32#DDPG_11_env2
            # cfg = [56, (96,2), 96, (184,2), 184, (360,2), 360, 352, 360, 352, 360, (480,2), 208]
            in_planes = 24
            cfg = [32, (72,2), 64, (120,2), 112, (224,2), 224, 192, 200, 208, 184, (208,2),208]
            
        else:
            raise NotImplementedError

        self.conv1 = conv_bn(3, in_planes, stride=2)

        self.features = self._make_layers(in_planes, cfg, conv_dw)

        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
