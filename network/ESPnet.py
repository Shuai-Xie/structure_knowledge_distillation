import torch
import torch.nn as nn


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)  # PReLU

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    """
    /2 downsample + pyramid dilated conv + HFF
    split - transform - merge
    """

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, kSize=3, stride=2)  # 与 DilatedParllelResidualBlockB 的区别，多了下采样

        # middle dilated conv
        self.d1 = CDilated(n, n1, 3, 1, d=1)  # stride, dilate rate
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        # nOut = 4*n + n1

        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # HFF
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # concat
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 5)  # 64/5
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    '''将 input img 下采样到 和 feature map 同样大小; 采用 avgpool 比 resize 结果更平滑, 因为是均值 pool
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return [classifier, output2_cat]


class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, classes=20, p=2, q=3, encoderFile=None):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes 背景一起训练???
        :param p: depth multiplier, alpha_2
        :param q: depth multiplier, alpha_3
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        self.level1 = CBR(3, 16, kSize=3, stride=2)  # 1/2
        self.sample1 = InputProjectionA(samplingTimes=1)  # avgpool samplingTimes, 3x3, stride=2, 得到想要的输出 size
        self.sample2 = InputProjectionA(samplingTimes=2)

        self.b1 = BR(16 + 3)  # img + level1
        self.level2_0 = DownSamplerB(16 + 3, 64)  # ESP, 1/4

        self.level2 = nn.ModuleList()
        for i in range(0, p):  # 2
            self.level2.append(DilatedParllelResidualBlockB(64, 64))  # 级联 ESP，不改变 feature map size
        self.b2 = BR(128 + 3)  # img + level2

        self.level3_0 = DownSamplerB(128 + 3, 128)  # ESP, 1/8
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))

        self.b3 = BR(256)
        self.classifier_1 = C(256, classes, 1, 1)  # 1x1 conv, classifier

        # self.encoder = ESPNet_Encoder(classes, p, q)
        # if encoderFile != None:
        #     self.encoder.load_state_dict(torch.load(encoderFile))
        #     print('Encoder loaded!')
        # # load the encoder modules
        # self.modules_ = []
        # for i, m in enumerate(self.encoder.children()):
        #     self.modules_.append(m)

        # 加上轻量的 decoder, (d)
        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)  # decoder 第1层桥接，处理 output1_cat, 图上第2个concat
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * classes),
                                           DilatedParllelResidualBlockB(2 * classes, classes, add=False))

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.level1(input)  # 1/2

        # 不同尺寸的 img
        inp1 = self.sample1(input)  # 1/2
        inp2 = self.sample2(input)  # 1/4

        output0_cat = self.b1(torch.cat([output0, inp1], 1))  # 16+3=19, 1/2
        output1_0 = self.level2_0(output0_cat)  # down-sampled, 19->64, 1/4

        for i, layer in enumerate(self.level2):  # ModuleList, 级联的 ESP
            if i == 0:  # 接口 module
                output1 = layer(output1_0)
            else:  # 内部 module
                output1 = layer(output1)  # 64, 1/4

        # concat 第1个 ESP 的输出 output1_0
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))  # 64+64+3

        output2_0 = self.level3_0(output1_cat)  # down-sampled, 128, 1/8
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))  # 128+128

        out = self.classifier_1(output2_cat)  # 256,C
        #
        # output2_cat = self.modules_[9](torch.cat([output2_0, output2], 1))  # concatenate for feature map width expansion

        # out = self.modules_[10](output2_cat)

        output2_c = self.up_l3(self.br(out))  # x2, C

        output1_C = self.level3_C(output1_cat)  # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1)))  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)

        return [classifier, output2_cat, out]  # multi stage loss


def get_ESPnet_encoder(num_classes=19):
    model = ESPNet_Encoder(num_classes, p=2, q=8)
    return model


def get_ESPnet_decoder(num_classes=19):
    model = ESPNet(num_classes, p=2, q=8)

    return model


if __name__ == '__main__':
    model = get_ESPnet_decoder(19)
    # model = get_psp_resnet18_pre(num_classes=19)
    # print(model)

    model_dict = model.state_dict()
    paras = sum(p.numel() for p in model.parameters() if p.requires_grad)  # numel 参数数量
    sum = 0
    print(paras)
# , encoderFile="/teamscratch/msravcshare/v-yifan/ESPNet/train/0.4results_enc_01_enc_2_8/model_298.pth"
