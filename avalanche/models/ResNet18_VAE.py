from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
from avalanche.models import MLP

class ResizeConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,scale_factor,mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=1)

    def forward(self,x):
        x = F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)
        x = self.conv(x)
        return x

class ResNet18Encoder(nn.Module):
    def __init__(self,z_dim=32):
        super(ResNet18Encoder, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = models.resnet18(pretrained=False)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, 2*self.z_dim)
    def forward(self, X):
        x = self.ResNet18(X)
        mu = x[:,:self.z_dim]
        logvar = x[:,self.z_dim:]
        # return mu, logvar
        return x

class BasicBlockDecoder(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Decoder(nn.Module):
    def __init__(self,num_Blocks = [2,2,2,2],z_dim=32,nc=3):
        super(ResNet18Decoder, self).__init__()
        self.in_planes = 512
        self.linear = nn.Linear(z_dim,512)
        self.layer4 = self._make_layer(BasicBlockDecoder, 256, num_Blocks[3],stride=2)
        self.layer3 = self._make_layer(BasicBlockDecoder, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDecoder, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDecoder, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64,nc,kernel_size=3,scale_factor=2)

    def _make_layer(self,BasicBlockDecoder,planes,num_Blocks,stride):
        strides = [stride] + [1] * (num_Blocks -1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDecoder(self.in_planes,stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,z):
        x = self.linear(z)
        x = x.view(z.size(0),512,1,1)
        x = F.interpolate(x,scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x,size=(112,112),mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        x =x.view(x.size(0),3,224,224)
        return x

class RES_VAE(nn.Module):
    def __init__(self, z_dim):
        super(RES_VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ResNet18Encoder(z_dim=self.z_dim)
        self.cal_mean = MLP([self.z_dim * 2,self.z_dim],last_activation=False)
        self.cal_var = MLP([self.z_dim * 2,self.z_dim],last_activation=False)
        self.decoder = ResNet18Decoder(z_dim=self.z_dim)

    def forward(self, x):
        #mean, logvar = self.encoder(x)
        represnetations = self.encoder(x)
        mean = self.cal_mean(represnetations)
        logvar = self.cal_var(represnetations)

        z = self.sampling(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    @staticmethod
    def sampling(mean, logvar):
        """
            VAE 'reparametrization trick'
        """
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size".
        """
        z = (
            torch.randn((batch_size, self.z_dim)).cuda()
            if batch_size
            else torch.randn((1, self.z_dim)).cuda()
        )
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

MSE_loss = nn.MSELoss(reduction="sum")

def loss_func(x, forward_output):
    recon_x, mean, logvar = forward_output
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = MSE_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD



