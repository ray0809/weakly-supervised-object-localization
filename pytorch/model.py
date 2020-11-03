import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet50Net(nn.Module):
    def __init__(self, rz_size, class_num):
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        self.upsample = nn.Upsample((rz_size, rz_size))
        resnet.fc = nn.Linear(resnet.fc.in_features, class_num, bias=True)
        self.resnet = resnet

        self._extract_feat_layers = ['layer4']
        self.rz_size = rz_size


    def forward(self, x):
        x = self.upsample(x) 
        x = self.resnet(x)
        return x

    def inference(self, x):
        outputs = []
        x = self.upsample(x)
        for name, layer in self.resnet.named_children():
            x = layer(x)
            if name in self._extract_feat_layers:
                out = F.interpolate(x, (self.rz_size, self.rz_size), mode='bilinear')
                outputs.append(out)
            if name == 'avgpool':
                x = torch.flatten(x, 1)
        x = F.softmax(x, dim=1)
        outputs.append(x)
        return outputs


class Inceptionv3Net(nn.Module):
    def __init__(self, rz_size, class_num):
        super().__init__()

        inception = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.upsample = nn.Upsample((rz_size, rz_size))

        self.fc = nn.Linear(inception.fc.in_features, class_num, bias=False)
        
        named_children = list(inception.named_children())
        

        self._extract_feat_layers = ['Mixed_7c']
        self.rz_size = rz_size

    def forward(self, x):
        x = self.upsample(x)
        x = self.inception(x)

        return x

    def inference(self, x):
        extra_tag = False
        outputs = []
        x = self.upsample(x)
        for name, layer in self.inception.named_children():
            x = layer(x)
            if name in self._extract_feat_layers:
                outputs.append(x)
                extra_tag = True

            

if __name__ == "__main__":
    m = ResNet50Net(224, 10)
    m = m.eval()

    inp = torch.randn(1,3,32,32)
    out = m(inp)
    print(out.shape)


    out1 = m.inference(inp)
    print(out1[0].shape, out1[1].shape)


    w = m.resnet.fc.weight.data.cpu().numpy().T
    print(w.shape)



    mm = Inceptionv3Net(299, 10)
    mm = mm.eval()
    out=  mm(inp)
    print(mm)
    print(out.shape)
    
