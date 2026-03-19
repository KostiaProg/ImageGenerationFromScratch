from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms

class Discriminator(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(self.hidden3(self.hidden2(self.hidden1(x))))
    
class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU()
        )
        self.out = nn.Sequential(
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(self.hidden3(self.hidden2(self.hidden1(x))))