from model.resnet50 import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch

class Stage2Head(nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(Stage2Head, self).__init__()
        
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone_stage2 = nn.Sequential(*list(self.backbone.children())[:6])  # conv1~layer2

        self.stemconv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        hidden_dim = 256
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 2)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, x):

        x = (x - self.mean) / self.std

        x = self.backbone_stage2(x)      # (B,512,H/8,W/8)
        x = self.stemconv(x)             # (B,512,H/8,W/8)
        x = self.gap(x).flatten(1)       # (B,512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # cls branch
        cls_logits = self.cls_head(x)
        # reg branch
        box_refine = self.reg_head(x)
        box_refine = torch.sigmoid(box_refine)

        return cls_logits, box_refine
