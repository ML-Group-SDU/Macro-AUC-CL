# from torchvision.models import vit_b_16, swin_b, ViT_B_16_Weights,  Swin_B_Weights
from torch import nn
import torch


class VisionTransformerB16(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerB16, self).__init__()
        # self.vit = vit_b_16(weights=ViT_B_16_Weights)

        # self.extractor = nn.Sequential(*list(self.vit.children())[:-1])

        self.num_feature = self.vit.heads.head.in_features
        self.new_heads = nn.Sequential(nn.Linear(self.num_feature, num_classes))


    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.new_heads(x)

        return x


class SwinTransformerSelf(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerSelf, self).__init__()

        # self.swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.num_feature = self.swin.head.in_features
        self.new_heads = nn.Sequential(nn.Linear(self.num_feature, num_classes))

    def forward(self, x):
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.swin.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.new_heads(x)
        return x
