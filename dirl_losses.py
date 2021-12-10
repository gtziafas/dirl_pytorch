from typing import *
import torch
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F 


class ClassificationLoss(nn.Module):
    # equivalent to tensoflow's softmax_cross_enropy_with_logits loss
    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
        return loss.mean()


class DomainLoss(nn.Module):
    # discriminate input images as source / target
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: Tensor, domain: Tensor, target_start_id: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        # if not given, assumes 50% source - 50% target in batch
        target_start_id = logits.shape[0] // 2 if target_start_id is None else target_start_id

        # discriminate domain of input 
        domain_loss = F.binary_cross_entropy_with_logits(logits, domain)

        # adversarial loss for the target data
        logits_target = logits[target_start_id:]
        adv_domain_target = torch.logical_not(domain)[target_start_id:].float()
        adv_domain_loss = F.binary_cross_entropy_with_logits(logits_target, adv_domain_target)

        return domain_loss, adv_domain_loss


class SoftTripletLoss(nn.Module):
    def __init__(self, margin: float, sigmas: List[float], l2_normalization: bool):
        super().__init__()
        self.margin = margin 
        self.sigmas = sigmas 
        self.l2_normalization = l2_normalization
        self.pdist_norm = lambda x: torch.square(x).sum(1)

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        num_anchors = embs.shape[0]
        labels = labels.argmax(-1)
        if self.l2_normalization:
            embs = F.normalize(embs, p=2, dim=1)

        pdist_matrix = self.pdist_norm(embs.unsqueeze(2) - embs.T).T
        beta = 1. / (2. * torch.tensor(self.sigmas, device=embs.device).unsqueeze(1))
        pdist_matrix = torch.matmul(-beta, pdist_matrix.flatten().unsqueeze(0))
        pdist_matrix = pdist_matrix.exp().sum(0).view(num_anchors, num_anchors)
        pdist_matrix /= pdist_matrix.sum(1).unsqueeze(1)

        mask_positives = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_negatives = torch.logical_not(mask_positives).float()
        
        anchors_rep = torch.tile(pdist_matrix, [1, num_anchors])
        anchors_rep = anchors_rep.reshape(num_anchors * num_anchors, num_anchors)
        anchors_rep_t = torch.tile(pdist_matrix, [num_anchors, 1])

        kl_loss = (anchors_rep * (anchors_rep.log() - anchors_rep_t.log())).sum(1)
        kl_loss = kl_loss.view(num_anchors, num_anchors)
        kl_div_pw_pos = torch.multiply(mask_positives, kl_loss)
        kl_div_pw_neg = torch.multiply(mask_negatives, kl_loss)
        kl_loss = kl_div_pw_pos.mean(1, keepdim=True) - kl_div_pw_neg.mean(1, keepdim=True) + torch.tensor(self.margin)
        kl_loss = torch.maximum(kl_loss, torch.zeros_like(kl_loss))

        return kl_loss.mean()


class ConditionalDomainLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_list: List[Tensor], labels: Tensor, domain: Tensor, target_start_id: int) -> Tuple[Tensor, Tensor]:
        batch_size = labels.shape[0]
        sizing = [target_start_id, batch_size - target_start_id]
        labels_source, labels_target = torch.split(labels, sizing)
        domain_source, domain_target = torch.split(domain, sizing)
        logits_list_source, logits_list_target = zip(*[torch.split(l, sizing) for l in logits_list])

        lossA, lossB  = 0., 0.
        for class_id in range(self.num_classes):
            is_class_source = (labels_source.argmax(1) == class_id).unsqueeze(1) 
            masked_domain_source = torch.masked_select(domain_source, is_class_source).view(-1, 2)
            masked_class_dann_source = torch.masked_select(logits_list_source[class_id], is_class_source).view(-1, 2)

            is_class_target = (labels_target.argmax(1) == class_id).unsqueeze(1) 
            masked_domain_target = torch.masked_select(domain_target, is_class_target).view(-1, 2)
            masked_class_dann_target = torch.masked_select(logits_list_target[class_id], is_class_target).view(-1, 2)
            masked_adv_domain_target = torch.masked_select(torch.logical_not(domain_target).float(), is_class_target).view(-1, 2)
            
            masked_domain = torch.cat((masked_domain_source, masked_domain_target), dim=0)
            masked_class_dann = torch.cat((masked_class_dann_source, masked_class_dann_target), dim=0)
            
            lossA += F.binary_cross_entropy_with_logits(masked_class_dann, masked_domain)
            lossB += F.binary_cross_entropy_with_logits(masked_class_dann_target, masked_adv_domain_target)        
        
        lossA /= self.num_classes
        lossB /= self.num_classes
        
        return lossA, lossB