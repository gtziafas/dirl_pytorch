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
        adv_domain_labels_target = (1 - domain)[target_start_id:]
        adv_domain_loss = F.binary_cross_entropy_with_logits(logits_target, adv_domain_labels_target)

        return domain_loss, adv_domain_loss


class SoftTripletLoss(nn.Module):
    # soft triplet loss adapted from https://github.com/ajaytanwani/DIRL/blob/master/src/dirl_core/triplet_loss_distribution.py
    def __init__(self, margin: float, sigmas: List[float], l2_normalization: bool):
        super().__init__()
        self.margin = margin 
        self.sigmas = sigmas 
        self.l2_normalization = l2_normalization

    def compute_pairwise_distances(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the squared pairwise Euclidean distances between x and y.
        Args:
            x: a tensor of shape [num_x_samples, num_features]
            y: a tensor of shape [num_y_samples, num_features]
        Returns:
            a distance matrix of dimensions [num_x_samples, num_y_samples].
        Raises:
            ValueError: if the inputs do no matched the specified dimensions.
        """
        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[-1] != y.shape[-1]:
            raise ValueError('The number of features should be the same.')

        norm = lambda x: torch.square(x).sum(1)
        return norm(x.unsqueeze(2) - y.T).T

    def kl_dist(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.distributions.Categorical(probs=x)
        y = torch.distributions.Categorical(probs=y)
        return torch.distributions.kl_divergence(x, y)

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        labels = labels.argmax(-1)
        if self.l2_normalization:
            embs = F.normalize(embs, p=2, dim=1)

        pdist_matrix = self.compute_pairwise_distances(embs, embs)
        beta = 1. / (2. * torch.tensor(self.sigmas, device=embs.device).unsqueeze(1))
        pdist_matrix = torch.matmul(-beta, pdist_matrix.reshape(1, -1))
        pdist_matrix = pdist_matrix.exp().sum(0).reshape(embs.shape[0], embs.shape[0])

        mask_anchors = labels.unsqueeze(0) == labels.unsqueeze(1).float()
        mask_negatives = torch.logical_not(mask_anchors).float()

        pdist_matrix = pdist_matrix / torch.norm(pdist_matrix, p=1, dim=1, keepdim=True)
        rep_anchors_rows = torch.tile(pdist_matrix, [1, pdist_matrix.shape[0]])
        rep_anchors_rows = rep_anchors_rows.reshape(pdist_matrix.shape[0] * pdist_matrix.shape[1], pdist_matrix.shape[0])
        pdist_matrix = torch.tile(pdist_matrix, [pdist_matrix.shape[0], 1])

        kl_loss = self.kl_dist(rep_anchors_rows + 1e-6, pdist_matrix + 1e-6)
        kl_loss = kl_loss.reshape(embs.shape[0], embs.shape[0])
        kl_div_pw_pos = torch.multiply(mask_anchors, kl_loss)
        kl_div_pw_neg = torch.multiply(mask_negatives, kl_loss)
        kl_loss = kl_div_pw_pos.mean(1, keepdim=True) - kl_div_pw_neg.mean(1, keepdim=True) + torch.tensor(self.margin)
        kl_loss = torch.maximum(kl_loss, torch.zeros_like(kl_loss))

        return kl_loss.mean()


class ConditionalDomainLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_list: List[Tensor], labels: Tensor, domain: Tensor, target_start_id: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        lossA, lossB  = 0., 0.
        for label in range(self.num_classes):
            is_class_source = labels[:target_start_id].argmax(1) == label 
            masked_domain_source = domain[:target_start_id][is_class_source]
            masked_class_dann_source = logits_list[label][:target_start_id][is_class_source]
            # masked_adv_domain_source = (1 - domain)[:target_start_id][is_class_source]

            is_class_target_1 = labels[target_start_id :].argmax(1) == label 
            masked_domain_target_1 = domain[target_start_id :][is_class_target_1]
            masked_class_dann_target_1 = logits_list[label][target_start_id :][is_class_target_1]
            masked_adv_domain_target_1 = (1 - domain)[target_start_id :][is_class_target_1]

            m_shape_ratio = torch.tensor(max(masked_domain_source.shape[0] // (masked_domain_target_1.shape[0]+1e-06), 1))
            masked_domain = torch.cat((masked_domain_source, masked_domain_target_1), dim=0)
            masked_class_dann = torch.cat((masked_class_dann_source, masked_class_dann_target_1), dim=0)
            # masked_adv_domain = torch.cat((masked_adv_domain_source, masked_adv_domain_target_1), dim=0)

            lossA += F.binary_cross_entropy_with_logits(masked_class_dann, masked_domain)
            lossB += F.binary_cross_entropy_with_logits(masked_class_dann_target_1, masked_adv_domain_target_1)        
        lossA /= self.num_classes
        lossB /= self.num_classes
        return lossA, lossB