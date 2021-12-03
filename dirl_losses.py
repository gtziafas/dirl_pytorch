from typing import *
import torch
import torch.nn as nn 
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
    def __init__(self, batch_size: int):
        super().__init__()
        assert batch_size % 2 == 0, 'batch size must be divided by 2'
        self.batch_size = batch_size

        # 0 = source, 1 = target - always remain like this because of batch sampling
        self.domain_labels = torch.vstack([torch.tile(torch.tensor([1, 0]), [batch_size // 2, 1]), 
                    torch.tile(torch.tensor([0, 1]), [batch_size // 2, 1])])
        self.adv_domain_labels = torch.tile(torch.tensor([1, 0]), [batch_size, 1])
    
    def forward(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        # discriminate domain of input 
        domain_loss = F.binary_cross_entropy_with_logits(logits, self.domain_labels)

        # adversarial loss for the target data
        logits_target = logits[self.batch_size // 2 :]
        adv_domain_labels_target = self.adv_domain_labels[self.batch_size // 2 :]
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
        X = torch.distributions.Categorical(probs=x)
        Y = torch.distributions.Categorical(probs=y)
        return torch.distributions.kl_divergence(X, Y)

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        labels = labels.argmax(-1)
        if self.l2_normalization:
            embs = F.normalize(embs, p=2, dim=1)

        pdist_matrix = self.compute_pairwise_distances(embs, embs)
        beta = 1. / (2. * torch.tensor(self.sigmas).unsqueeze(1))
        pdist_matrix_pos = (-beta @ pdist_matrix.reshape(1, -1)).exp().sum(0).reshape(pdist_matrix.shape)

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_negatives = torch.logical_not(labels_equal).float()

        indices_equal = torch.eye(labels.shape[0]).bool()
        indices_not_equal = torch.logical_not(indices_equal)
        mask_positives = torch.logical_and(indices_not_equal, labels_equal).float()

        mask_anchors = labels_equal.float()
        anchors_dist = pdist_matrix_pos / torch.norm(pdist_matrix_pos, p=1, dim=1, keepdim=True)
        rep_anchors_rows = torch.tile(anchors_dist, [1, anchors_dist.shape[0]])
        rep_anchors_rows = rep_anchors_rows.reshape(anchors_dist.shape[0] * anchors_dist.shape[1], anchors_dist.shape[0])
        rep_anchors_matrices = torch.tile(anchors_dist, [anchors_dist.shape[0], 1])

        kl_loss_pw = self.kl_dist(rep_anchors_rows + 1e-6, rep_anchors_matrices + 1e-6)
        kl_loss_pw = kl_loss_pw.reshape(anchors_dist.shape[0], anchors_dist.shape[0])
        kl_div_pw_pos = torch.multiply(mask_anchors, kl_loss_pw)
        kl_div_pw_neg = torch.multiply(mask_negatives, kl_loss_pw)
        kl_loss = kl_div_pw_pos.mean(1, keepdim=True) - kl_div_pw_neg.mean(1, keepdim=True) + torch.tensor(self.margin)
        kl_loss = torch.maximum(kl_loss, torch.zeros_like(kl_loss))

        return kl_loss.mean()


class ConditionalDomainLoss(nn.Module):
    def __init__(self, num_classes: int, sizing: List[int]):
        super().__init__()
        self.num_classes = num_classes
        self.sizing = sizing 
        self.batch_size = sum(sizing)

        # 0 = source, 1 = target - always remain like this because of batch sampling
        self.domain_labels = torch.vstack([torch.tile(torch.tensor([1, 0]), [self.batch_size // 2, 1]), 
                    torch.tile(torch.tensor([0, 1]), [self.batch_size // 2, 1])])
        self.adv_domain_labels = torch.tile(torch.tensor([1, 0]), [self.batch_size, 1])

    def forward(self, logits_list: List[Tensor], labels: Tensor) -> Tuple[Tensor, Tensor]:
        labels_source = labels[:self.sizing[0]]
        domain_source = self.domain_labels[:self.sizing[0]]
        adv_domain_source = self.adv_domain_labels[:self.sizing[0]]
        labels_target_sup = labels[self.sizing[0] : sum(self.sizing[0:2])]
        domain_target_sup = self.domain_labels[self.sizing[0] : sum(self.sizing[0:2])]
        adv_domain_target_sup = self.adv_domain_labels[self.sizing[0] : sum(self.sizing[0:2])]

        lossA, lossB  = 0., 0.
        for label in range(self.num_classes):
            is_class_source = labels_source.argmax(1) == label 
            masked_domain_source = domain_source[is_class_source]
            masked_class_dann_source = logits_list[label][:self.sizing[0]][is_class_source]
            masked_adv_domain_source = adv_domain_source[is_class_source]

            is_class_target_1 = labels_target_sup.argmax(1) == label 
            masked_domain_target_1 = domain_target_sup[is_class_target_1]
            masked_class_dann_target_1 = logits_list[label][self.sizing[0] : sum(self.sizing[0:2])][is_class_target_1]
            masked_adv_domain_target_1 = adv_domain_target_sup[is_class_target_1]

            m_shape_ratio = torch.tensor(max(masked_domain_source.shape[0] // masked_domain_target_1.shape[0], 1))
            masked_domain = torch.cat((masked_domain_source, masked_domain_target_1), dim=0)
            masked_class_dann = torch.cat((masked_class_dann_source, masked_class_dann_target_1), dim=0)
            masked_adv_domain = torch.cat((masked_adv_domain_source, masked_adv_domain_target_1), dim=0)

            lossA += F.binary_cross_entropy_with_logits(masked_class_dann.float(), masked_domain.float())
            lossB += F.binary_cross_entropy_with_logits(masked_class_dann_target_1.float(), masked_adv_domain_target_1.float())        
        lossA /= self.num_classes
        lossB /= self.num_classes

        return lossA, lossB