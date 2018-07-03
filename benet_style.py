import torch
import cv2
import numpy as np
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import scipy.misc

def loadim(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]])
    im = torch.from_numpy(im)
    im = im.type('torch.FloatTensor')
    im = im/128 - 2

    return im

vgg11 = models.vgg11(pretrained=True)

class VGG16_conv7(nn.Module):
    def __init__(self):
        super(VGG16_conv7, self).__init__()
        self.features = nn.Sequential(
            # stop at conv7
            *list(vgg11.features.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv6(nn.Module):
    def __init__(self):
        super(VGG16_conv6, self).__init__()
        self.features = nn.Sequential(
            # stop at conv6
            *list(vgg11.features.children())[:-5]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv5(nn.Module):
    def __init__(self):
        super(VGG16_conv5, self).__init__()
        self.features = nn.Sequential(
            # stop at conv5
            *list(vgg11.features.children())[:-8]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv4(nn.Module):
    def __init__(self):
        super(VGG16_conv4, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(vgg11.features.children())[:-10]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv3(nn.Module):
    def __init__(self):
        super(VGG16_conv3, self).__init__()
        self.features = nn.Sequential(
            # stop at conv3
            *list(vgg11.features.children())[:-13]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv2(nn.Module):
    def __init__(self):
        super(VGG16_conv2, self).__init__()
        self.features = nn.Sequential(
            # stop at conv2
            *list(vgg11.features.children())[:-15]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class VGG16_conv1(nn.Module):
    def __init__(self):
        super(VGG16_conv1, self).__init__()
        self.features = nn.Sequential(
            # stop at conv1
            *list(vgg11.features.children())[:-18]
        )

    def forward(self, x):
        x = self.features(x)
        return x



if __name__ == '__main__':

    pass

    vgg1 = VGG16_conv1()
    vgg2 = VGG16_conv2()
    vgg3 = VGG16_conv3()
    vgg4 = VGG16_conv4()
    vgg5 = VGG16_conv5()
    vgg6 = VGG16_conv6()
    vgg7 = VGG16_conv7()

    cont_im = loadim('landscape-small.png')

    cont_im = cont_im.unsqueeze(0)



    cont_im.requires_grad = True

    style_im = loadim('van-gogh-small.png')
    style_im = style_im.unsqueeze(0)

    opt = optim.SGD([cont_im], lr=0.0001)

    y1_style = vgg1(style_im)
    y2_style = vgg2(style_im)
    y3_style = vgg3(style_im)
    y4_cont = vgg4(style_im)
    y5_cont = vgg5(cont_im)
    y6_cont = vgg6(cont_im)
    y7_cont = vgg7(cont_im)

    for i in range(20):

        print('Iteration', i)

        opt.zero_grad()

        y1_ = vgg1(cont_im)
        y2_ = vgg2(cont_im)
        y3_ = vgg3(cont_im)
        y4_ = vgg4(cont_im)
        y5_ = vgg5(cont_im)
        y6_ = vgg6(cont_im)
        y7_ = vgg7(cont_im)

        y1_d = y1_style - y1_
        y2_d = y2_style - y2_
        y3_d = y3_style - y3_
        y4_d = y4_cont - y4_
        y5_d = y5_cont - y5_
        y6_d = y6_cont - y6_
        y7_d = y7_cont - y7_

        y1_d = y1_d * y1_d
        y2_d = y2_d * y2_d
        y3_d = y3_d * y3_d
        y4_d = y4_d * y4_d
        y5_d = y5_d * y5_d
        y6_d = y6_d * y6_d
        y7_d = y7_d * y7_d

        loss = torch.tensor(0, dtype=torch.float)
        loss.requires_grad = True
        for dif in [y1_d, y2_d, y3_d, y4_d, y5_d, y6_d, y7_d]:
            l = torch.sum(dif)
            loss = loss + l

        loss.backward(retain_graph =True)
        opt.step()

        b = cont_im[0].detach().numpy()

        b = np.rollaxis(b, 0, 3)
        print(type(b))

        scipy.misc.imsave('content' + str(i) + '.jpg', b)


