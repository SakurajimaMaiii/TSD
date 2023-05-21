# coding=utf-8
import torch.nn as nn
from torchvision import models
import torchvision
import torch
import timm  #load ViT or MLP-mixer
from network.common_network import Identity

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, 
            "resnet34": models.resnet34, 
            "resnet50": models.resnet50,
            "resnet101": models.resnet101, 
            "resnet152": models.resnet152, 
            "resnext50": models.resnext50_32x4d, 
            "resnext101": models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class ViTBase(nn.Module):
    def __init__(self,model_name):
        self.KNOWN_MODELS = {
        'ViT-B16': 'vit_base_patch16_224_in21k', 
        'ViT-B32': 'vit_base_patch32_224_in21k',
        'ViT-L16': 'vit_large_patch16_224_in21k',
        'ViT-L32': 'vit_large_patch32_224_in21k',
        'ViT-H14': 'vit_huge_patch14_224_in21k'
    }
    
        self.FEAT_DIM = {
        'ViT-B16': 768, 
        'ViT-B32': 768,
        'ViT-L16': 1024,
        'ViT-L32': 1024,
        'ViT-H14': 1280
    }    
        super().__init__()
        self.vit_backbone = timm.create_model(self.KNOWN_MODELS[model_name],pretrained=True,num_classes=0)
        self.in_features = self.FEAT_DIM[model_name]
    
    def forward(self,x):
        return self.vit_backbone(x)
        


effnet_dict = {"efficientnet_b0": models.efficientnet_b0, 
         "efficientnet_b1": models.efficientnet_b1,
         "efficientnet_b2": models.efficientnet_b2,
         "efficientnet_b3": models.efficientnet_b3,
         "efficientnet_b4": models.efficientnet_b4,
         "efficientnet_b5": models.efficientnet_b5,
         "efficientnet_b6": models.efficientnet_b6,
         "efficientnet_b7": models.efficientnet_b7}


class EfficientBase(nn.Module):
    def __init__(self,backbone="efficientnet_b4"):
        super().__init__()
        self.network = effnet_dict[backbone](pretrained=True)
        self.in_features = self.network.classifier[1].in_features
        self.network.classifier = Identity()
        
        
    def forward(self,x):
        return self.network(x)



mlp_mixer_path = {'Mixer-B16':mlp_mixer_b16_path,
                  'Mixer-L16':mlp_mixer_l16_path}

class MLPMixer(nn.Module):
    KNOWN_MODELS = {
        'Mixer-B16': timm.models.mlp_mixer.mixer_b16_224_in21k,
        'Mixer-L16': timm.models.mlp_mixer.mixer_l16_224_in21k,
    }
    def __init__(self,backbone="Mixer-L16"):
        super().__init__()
        func = self.KNOWN_MODELS[backbone]
        self.network = func(pretrained=True)
        self.in_features = self.network.norm.normalized_shape[0]
        self.network.head = Identity()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    