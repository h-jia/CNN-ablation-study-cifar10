import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)    
    
    
class VGG_wo_bn(nn.Module):
    def __init__(self):
        super(VGG_wo_bn, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),          
                           nn.ReLU(inplace=True)]
                in_channels = x        
        return nn.Sequential(*layers)    
    
    
class VGG_wo_maxpool(nn.Module):
    def __init__(self):
        super(VGG_wo_maxpool, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(524288, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                pass
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),          
                           nn.ReLU(inplace=True)]
                in_channels = x        
        return nn.Sequential(*layers)        
    
class VGG_wo_relu(nn.Module):
    def __init__(self):
        super(VGG_wo_relu, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x)]            
                in_channels = x        
        return nn.Sequential(*layers)    
    
class VGG_wo_bn_relu(nn.Module):
    def __init__(self):
        super(VGG_wo_bn_relu, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]         
                in_channels = x        
        return nn.Sequential(*layers)    
                           
class VGG_linear_1(nn.Module):
    def __init__(self):
        super(VGG_linear_1, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(524288, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                pass
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]                   
                in_channels = x        
        return nn.Sequential(*layers)    

                           
                           
class VGG_linear_2(nn.Module):
    def __init__(self):
        super(VGG_linear_2, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(524288, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                pass
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]                   
                in_channels = x        
        return nn.Sequential(*layers)    
                           
class VGG_linear_3(nn.Module):
    def __init__(self):
        super(VGG_linear_3, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]    
                in_channels = x  
        return nn.Sequential(*layers)    

class VGG_linear_4(nn.Module):
    def __init__(self):
        super(VGG_linear_4, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),                   
                           nn.ReLU(inplace=True)]      
                in_channels = x  
        return nn.Sequential(*layers) 

                           
class VGG_linear_5(nn.Module):
    def __init__(self):
        super(VGG_linear_5, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),                   
                           nn.Sigmoid()]      
                in_channels = x  
        return nn.Sequential(*layers)             
                           
class VGG_linear_6(nn.Module):
    def __init__(self):
        super(VGG_linear_6, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]      
                in_channels = x  
        return nn.Sequential(*layers)                                 

class VGG_linear_7(nn.Module):
    def __init__(self):
        super(VGG_linear_7, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 64, 'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x     
        return nn.Sequential(*layers)                               
                           
class VGG_shallow_1(nn.Module):
    def __init__(self):
        super(VGG_shallow_1, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256,'M', 512, 'M', 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)    


class VGG_shallow_2(nn.Module):
    def __init__(self):
        super(VGG_shallow_2, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M'])
        self.classifier = nn.Linear(256 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)   

                           
                           
class VGG_bottleneck_1(nn.Module):
    def __init__(self):
        super(VGG_bottleneck_1, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 64, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)    
                               
                           
class VGG_bottleneck_2(nn.Module):
    def __init__(self):
        super(VGG_bottleneck_2, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 16, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)    

                           
class VGG_sigmoid(nn.Module):
    def __init__(self):
        super(VGG_sigmoid, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.Sigmoid()]
                in_channels = x
        return nn.Sequential(*layers)                            