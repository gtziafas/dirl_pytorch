from typing import *
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor
from grad_reverse import GradReverse


class DigitsDIRLEncoder(nn.Module):
    def __init__(self, num_features: int, emb_dim: int):
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
        self.emb = nn.Linear(num_features, emb_dim)

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
    def __init__(self, 
                num_classes: int=10, 
                emb_dim: int=128 
                ):
        super().__init__()
        self.num_classes = num_classes

        # freeze gradients to remove unwanted contributions during adversarial step
        self.freeze_gradient = GradReverse(0)

        # CNN encoder
        self.encoder = DigitsDIRLEncoder(1152, 256)

        # classifier
        self.cls = nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, num_classes))

        # domain discriminator
        self.dd = nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, 2))

        # conditional domain discriminator
        # self.cdd = nn.ModuleList([nn.Sequential(nn.Linear(128, 100),
        #                     nn.LeakyReLU(),
        #                     nn.Linear(100, 50),
        #                     nn.LeakyReLU(),
        #                     nn.Linear(50, 2)) for _ in range(num_classes)])

        # parallel version
        self.cdd_1 = nn.Linear(emb_dim, 100 * num_classes)
        self.cdd_2 = nn.Conv1d(100 * num_classes, 50 * num_classes, kernel_size=1, groups=num_classes)
        self.cdd_3 = nn.Conv1d(50 * num_classes, 2 * num_classes, kernel_size=1, groups=num_classes)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor, freeze_gradient: bool = False) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        embs = self.encode(x)

        # class label predictions
        class_preds = self.cls(embs)

        # freeze_gradient:
        #   - In classification step (step A), we freeze the gradients of the
        #     embeddings when discriminating, so that the encoder is only trained
        #     by class labels, while discriminators are fine-tuned on top of fixed 
        #     features.
        #
        #   - In adversarial step (step B), only the encoder is trained by using
        #     flipped domain labels, updating it towards a point that fools the 
        #     discriminators.
        embs_f = self.freeze_gradient(embs) if freeze_gradient else embs

        # domain predictions 
        domain_preds = self.dd(embs_f)

        # class-conditioned domain predictions - flip aswell
        # cond_domain_preds = list(map(self.cdd, self.num_classes * [embs_frozen]))

        # parallel version 
        cond_domain_preds = self.cdd_1(embs_f).unsqueeze(-1) # B x 100K x 1
        cond_domain_preds = F.leaky_relu(cond_domain_preds)
        cond_domain_preds = self.cdd_2(cond_domain_preds) # B x 50K x 1
        cond_domain_preds = F.leaky_relu(cond_domain_preds)
        cond_domain_preds = self.cdd_3(cond_domain_preds) # B x 2K x 1
        cond_domain_preds = list(cond_domain_preds.squeeze().view(self.num_classes, -1, 2)) # [K x] B x 2    

        return embs, domain_preds, class_preds, cond_domain_preds

