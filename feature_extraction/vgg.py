import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable as V


vgg_feat_list = ['conv1_1', 'ReLU1_1', 'conv1_2', 'ReLU1_2','maxpool1',
                 'conv2_1','ReLU2_1','conv2_2','ReLU2_2','maxpool2',
                 'conv3_1','ReLU3_1','conv3_2','ReLU3_2','conv3_3',
                 'ReLU3_3','conv3_4','ReLU3_4','maxpool3',
                 'conv4_1','ReLU4_1','conv4_2','ReLU4_2','conv4_3',
                 'ReLU4_3','conv4_4','ReLU4_4','maxpool4',
                 'conv5_1','ReLU5_1','conv5_2','ReLU5_2','conv5_3',
                 'ReLU5_3','conv5_4','ReLU5_4','maxpool5']

vgg_classifier_list = ['fc6','ReLU6','Dropout6','fc7','ReLU7','Dropout7','fc8']


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select_feats = ['maxpool1', 'maxpool2', 'maxpool3',
                             'maxpool4', 'maxpool5']
        self.select_classifier = ['fc6' , 'fc7', 'fc8']

        self.feat_list = self.select_feats + self.select_classifier

        self.vgg_feats = models.vgg19(pretrained=True).features
        self.vgg_classifier = models.vgg19(pretrained=True).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        """Extract multiple feature maps."""
        features = []
        for name, layer in self.vgg_feats._modules.items():
            x = layer(x)
            if vgg_feat_list[int(name)] in self.select_feats:
                features.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        for name, layer in self.vgg_classifier._modules.items():
            x = layer(x)
            if vgg_classifier_list[int(name)] in self.select_classifier:
                features.append(x)
        return features


def load_vgg():
    model = VGGNet()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
    return
