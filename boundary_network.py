"""Sensor-only and sensor-conditioned frozen/trainable visual boundary experts."""
import torch
from torch import nn
from torchvision.models import resnet50


class Prefix(nn.Module):
    def __init__(self, weights):
        super().__init__()
        net = resnet50(weights=None)
        net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        self.blocks = nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1,net.layer2,net.layer3)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.requires_grad_(False)

    def forward(self, rgb):
        x = rgb.permute(0,3,1,2).float()/255.
        # Quantization is intentional and identical for cache and fresh inference.
        return self.blocks((x-self.mean)/self.std).half()


class Expert(nn.Module):
    def __init__(self, sensor_dim, kind, weights=None):
        super().__init__()
        self.kind = kind
        self.sensor = nn.Sequential(nn.Linear(sensor_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),
                                    nn.Linear(128,128),nn.LayerNorm(128),nn.GELU())
        self.sensor_pair = nn.Linear(128,2)
        self.sensor_aux = nn.Linear(128,6)
        if kind != 'sensor_only':
            self.film = nn.Linear(128,4096)
            nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
            self.visual = nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,128),nn.GELU(),nn.Dropout(.15))
            self.correction = nn.Linear(256,2)
            nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
            self.image_aux = nn.Linear(2048,6)
        if kind == 'finetune_film':
            net = resnet50(weights=None)
            if weights: net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
            self.layer4 = net.layer4

    def train(self, mode=True):
        super().train(mode)
        # No batch, time or held-fold adaptation of normalization statistics.
        for block in self.modules():
            if isinstance(block,nn.BatchNorm2d): block.eval()
        return self

    def forward(self, sensor, visual=None):
        s = self.sensor(sensor)
        pair = self.sensor_pair(s)
        aux = self.sensor_aux(s)
        image_aux = None
        if self.kind != 'sensor_only':
            if self.kind == 'finetune_film':
                visual = self.layer4(visual).mean(dim=(-2,-1))
            gamma,beta = self.film(s).chunk(2,dim=1)
            modulated = visual*(1+.1*torch.tanh(gamma))+.1*beta
            h = self.visual(modulated)
            pair = pair+self.correction(torch.cat([s,h],dim=1))
            image_aux = self.image_aux(visual)
        return pair,aux,image_aux
