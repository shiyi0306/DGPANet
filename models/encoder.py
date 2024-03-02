"""
Backbones supported by torchvison.
"""
import torch
import torch.nn as nn
import torchvision


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet101'):
        super().__init__()
        # using pretrained model's weights
        if pretrained_weights == 'deeplabv3':
            self.pretrained_weights = torch.load(
                "./pretrained_model/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth", map_location='cpu')
        elif pretrained_weights == 'resnet101':
            self.pretrained_weights = torch.load("./pretrained_model/hub/checkpoints/resnet101-63fe2227.pth",
                                                 map_location='cpu')
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        # self.reduce1 = nn.Sequential(
        #              nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
        #              nn.ReLU(inplace=True),
        #              nn.Dropout2d(p=0.5)
        # )
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        # self.reduce2 = nn.Sequential(
        #              nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=False),
        #              nn.ReLU(inplace=True),
        #              nn.Dropout2d(p=0.5)
        # )
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def forward(self, x):
        features = dict()
        x = self.backbone["conv1"](x)
        # print('conv1',x.shape)
        x = self.backbone["bn1"](x)
        # print('bn1', x.shape)
        x = self.backbone["relu"](x)
        # print('relu', x.shape)
        # features['down1'] = x
        x = self.backbone["maxpool"](x)
        # print('maxpool', x.shape)
        x = self.backbone["layer1"](x)
        # print('layer1', x.shape)
        x = self.backbone["layer2"](x)
        # print('layer1', x.shape)
        features['layer2'] = x
        x = self.backbone["layer3"](x)
        # print('layer3', x.shape)
        features['down2'] = self.reduce1(x)
        # print('featuresdown2', features['down2'].shape)
        x = self.backbone["layer4"](x)
        # print('layer4', x.shape)
        # features['down3'] = self.reduce2(x)
        # print('featuresdown3', features['down3'].shape)
        
        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        # print('avgpool', t.shape)
        t = torch.flatten(t, 1)
        # print('flatten', t.shape)
        t = self.backbone["fc"](t)
        # print('fc', t.shape)
        t = self.reduce1d(t)
        # print('reduce1d(t)', t.shape)

        return (features, t)

        # return features

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(keys)):
                if keys[i] in new_keys:
                    new_dic[keys[i]] = self.pretrained_weights[keys[i]]

            self.load_state_dict(new_dic)
