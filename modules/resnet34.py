import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet34NoHead(ResNet):
    def __init__(self):
        super(ResNet34NoHead, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=0)
        # Remove the final fully connected layer (classification head)
        self.fc = nn.Identity()
        self.rep_dim = 512

    def forward(self, x):

        # Perform forward pass without the classification head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)



        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def get_resnet(state_dict_path = None):
    resnet34 = ResNet34NoHead()
    if state_dict_path is not None:
        pretrained_weights = torch.load(state_dict_path)
        backbone_weights = {k: v for k, v in pretrained_weights.items() if 'fc' not in k}
        resnet34.load_state_dict(backbone_weights, strict=True) #?

    return resnet34