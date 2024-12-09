from torch import nn 
from torchvision import models
import torch 

class MTL_Swin_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.swin_b(weights = models.Swin_B_Weights.IMAGENET1K_V1)
        self.n_features = self.net.head.in_features
        self.net.head = nn.Identity()
        self.net.head1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 18)
        )
        self.net.head2 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 7),
            ## 1,2 / 3,4 / 5,6 / 7,8 / 9,10,11 / 12 - 18
        )

    def forward(self, x):
        output = self.net(x)
        
        output1 = self.net.head1(output)
        output2 = self.net.head2(output)
        
        return output1, output2
