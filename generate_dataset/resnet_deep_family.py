import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MyBasicBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(MyBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu3 = nn.ReLU(inplace=True)
 
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu3(out)
        return out
 
class MyBottleneck(nn.Module):
    def __init__(self, in_planes, mid_planes_1, mid_planes_2, out_planes, stride=1, linear_shortcut=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes_1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes_1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_planes_1, mid_planes_2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes_2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_planes_2, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)
 
        self.downsample = nn.Sequential()
        if linear_shortcut:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
 
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu3(out)
        return out

 
class ResNetDeep(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.maxpool1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out

class MyResNetDeepBasic(nn.Module):
    def __init__(self, node_num, num_blocks):
        super().__init__()
        self.node_num = node_num
        self.idx = 1
        self.conv1 = nn.Conv2d(3, node_num[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(node_num[1])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(node_num[-2], node_num[-1])

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(MyBasicBlock(self.node_num[self.idx], self.node_num[self.idx+1], self.node_num[self.idx+2], stride))
            self.idx += 2
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.maxpool1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out
        
class MyResNetDeep(nn.Module):
    def __init__(self, node_num, num_blocks):
        super().__init__()
        self.node_num = node_num
        self.idx = 1
        self.conv1 = nn.Conv2d(3, node_num[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(node_num[1])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(node_num[-2], node_num[-1])

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            linear_shortcut = (i == 0)
            layers.append(MyBottleneck(self.node_num[self.idx], self.node_num[self.idx+1], self.node_num[self.idx+2], self.node_num[self.idx+3], stride, linear_shortcut))
            self.idx += 3
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.maxpool1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out

 
def myresnet18(num_classes=10):
    return ResNetDeep(BasicBlock, [2,2,2,2], num_classes)
 
def myresnet34(num_classes=10):
    return ResNetDeep(BasicBlock, [3,4,6,3], num_classes)
 
def myresnet50(num_classes=10):
    return ResNetDeep(Bottleneck, [3,4,6,3], num_classes)
 
def myresnet101(num_classes=10):
    return ResNetDeep(Bottleneck, [3,4,23,3], num_classes)
 
def myresnet152(num_classes=10):
    return ResNetDeep(Bottleneck, [3,8,36,3], num_classes)

def myresnet26(num_classes=10):
    return ResNetDeep(Bottleneck, [2,2,2,2], num_classes)
    

if __name__ == "__main__":
    pass