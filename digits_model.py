from typing import *
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor


class DigitsDIRLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            )  
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            ) 

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            ) 

        # bottleneck
        self.emb = nn.Linear(1152, 256)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.emb(x.flatten(1))
        return x


class DigitsDIRL(nn.Module):
    def __init__(self, sizing: Tuple[int], l_values: Tuple[float, float], num_classes: int=10):
        super().__init__()
        self.sizing = sizing
        self.l_values = l_values
        self.num_classes = num_classes

        # CNN encoder
        self.encoder = DigitsDIRLEncoder()

        # classifier
        self.cls = nn.Sequential(nn.Linear(256, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, num_classes))

        # domain discriminator
        self.dd = nn.Sequential(nn.Linear(256, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, 2))

        # conditional domain discriminator
        self.cdd = nn.Sequential(nn.Linear(256, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, 2))


    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        embs = self.encode(x)

        # label predictions
        class_preds = self.cls(embs)

        # domain predictions
        ads_embs = self.flip_gradient(embs, self.l_values[0])
        domain_preds = self.dd(ads_embs)

        # conditional domain predictions
        ads_embs_cond = self.flip_gradient(embs, self.l_values[1])
        cond_domain_preds = []
        for label in range(self.num_classes):
            cond_domain_preds.append(self.cdd(ads_embs_cond))

        return embs, domain_preds, class_preds, cond_domain_preds

