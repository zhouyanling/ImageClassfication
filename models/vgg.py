#-*-coding:utf-8-*-
import torch
import torch.nn as nn

cfg = {
    'vgg11': [64,'M', 128, 'M', 256,256, 'M', 512, 512, 'M'],
    'vgg13': [64,64,'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64,64,'M', 128, 128, 'M',  256,256, 256, 'M', 512,512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64,64,'M', 128, 128, 'M',  256,256,256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self,vgg_name, num_class):
        super(VGG,self).__init__()
        self.size = 0.0
        self.num_class = num_class
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512,10)

        self.classifier = self._fc_layers(num_class)

    def forward(self, x):
        inputSize = x.size()
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg :
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else :
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,
                                     padding=1, bias= False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _fc_layers(self, num_class):
        #layers = []
        tmptensor = torch.randn(64, 512, 7, 7)
        tmptensor = tmptensor.view(tmptensor.size(0), -1)
        tmp1 = tmptensor.size(1)
        layers = nn.Linear(tmp1, num_class)
        # layers += [nn.Linear(tmp1,4096),
        #            nn.ReLU(True),
        #            nn.Dropout()]
        #
        # layers += [nn.Linear(4096,4096),
        #            nn.ReLU(True),
        #            nn.Dropout(),
        #            nn.Linear(4096, num_class)]
        #return nn.Sequential(*layers)
        return nn.Sequential(layers)