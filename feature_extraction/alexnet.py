import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


alex_feat_list = ['conv1','ReLU1','maxpool1',\
'conv2','ReLU2','maxpool2',\
'conv3','ReLU3',\
'conv4','ReLU4',\
'conv5','ReLU5','maxpool5',\
]

alex_classifier_list = ['Dropout6','fc6','ReLU6','Dropout7','fc7','ReLU7','fc8']


class AlexNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(AlexNet, self).__init__()
        self.select_feats = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.select_classifier = ['fc6' , 'fc7', 'fc8']

        self.feat_list = self.select_feats + self.select_classifier

        self.alex_feats = models.alexnet(pretrained=True).features
        self.alex_classifier = models.alexnet(pretrained=True).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        """Extract multiple feature maps."""
        features = []
        for name, layer in self.alex_feats._modules.items():
            x = layer(x)
            if alex_feat_list[int(name)] in self.feat_list:
                features.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        for name, layer in self.alex_classifier._modules.items():
            x = layer(x)
            if alex_classifier_list[int(name)] in self.feat_list:
                features.append(x)
        return features


def load_alexnet():
    model = AlexNet()
    #state_dict = model_zoo.load_url(model_urls['alexnet'])
    #model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
