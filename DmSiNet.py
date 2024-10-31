import torch
import torch.nn as nn

import torch.nn.functional as F
from mmcv.cnn.utils.flops_counter import flops_to_string, add_flops_counting_methods, get_model_parameters_number
from torchsummary import torchsummary

import FeatureFusionModule
from DenseBlock import Denseasppblock


class DmSiNet(nn.Module):
    def __init__(self, classes=3, pretrained=False):
        super(DmSiNet, self).__init__()
        # assert layers in [50,101,152]
        assert classes > 1
        # resnet = resnet18(pretrained=pretrained)

        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.classifier = nn.Sequential(nn.Conv2d(64, classes, 1, 1))
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
        )




        self.layer1 = Denseasppblock(3,64)
        self.layer2 = Denseasppblock(64, 128)
        self.layer3 = Denseasppblock(128, 256)
        self.layer4 = Denseasppblock(256, 512)
        self.layer5 = Denseasppblock(512, 1024)


        self.ffm1 = FeatureFusionModule.FFModule(512, 1024, 512)
        self.ffm2 = FeatureFusionModule.FFModule(256, 512, 256)
        self.ffm3 = FeatureFusionModule.FFModule(128, 256, 128)
        self.ffm4 = FeatureFusionModule.FFModule(64, 128, 64)

    # 输入x为 3,512,512
    def forward(self, x):

        layer1 = self.layer1(x) #64.256.256
        layer2 = self.layer2(layer1)#128.128.128
        layer3 = self.layer3(layer2)#256.64.64

        layer4 = self.layer4(layer3)#512.32.32
        layer5 = self.layer5(layer4)#1024.16.16

        up1 = self.up(layer5) #1024.32.32
        out_1 = self.ffm1(layer4, up1) # 1024,32,32 up1 layer4 512,32，32
        # out_1的大小是512,23,32
        up2 = self.up(out_1) #
        out_2 = self.ffm2(layer3, up2)  #128.128.128 512.32.32 output 128.128.128

        up3 = self.up(out_2) # 128.256.256
        out_3 = self.ffm3(layer2, up3) #output 64.256.256

        up4 = self.up(out_3)
        out_4 = self.ffm4(layer1, up4)

        up5 = self.up(out_4)
        classifier = self.classifier(up5)
        return classifier


if __name__ == '__main__':
    # from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model = DmSiNet(classes=3).cuda()

    torchsummary.summary(model, input_size=(3,512, 512), device="cuda")
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = torch.FloatTensor(1,3, 512, 512)
    batch = batch.cuda()

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)
    print(1)

    # # print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    # print('Params: ' + str(get_model_parameters_number(model)))
    # print('Output shape: {}'.format(list(out.shape)))
    # total_paramters = sum(p.numel() for p in model.parameters())
    # print('Total paramters: {}'.format(total_paramters))
