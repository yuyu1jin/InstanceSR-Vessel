from torch import nn as nn
from torch.nn import functional as F

from torch.nn.utils import spectral_norm

# discriminator network, outputs a scalar (note: not yet the final real/fake probability), corresponds to c(x) in the paper
class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(Discriminator, self).__init__()
        self.num_feat = num_feat
        self.skip_connection = skip_connection
        
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 2, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 2, 3, 2, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer5 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 4, 3, 2, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer6 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )
        self.layer7 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 8, 3, 2, 1, bias=False)), 
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.fc1 = nn.Linear(num_feat * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        # shared layers
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        num_feat = self.num_feat

        x0 = self.conv0(x)
        x0 = self.lrelu(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6) # cannel = num_feat * 8

        # GAP
        x7 = F.adaptive_avg_pool2d(x7, (1,1)).view(x7.size(0), -1)
        # FC
        out = self.fc1(x7)
        out = self.lrelu(out)
        out = self.fc2(out)

        return out