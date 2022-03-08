from _types import *
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor
from torch.autograd import Function


class GradReverseFunction(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None 
        _, alpha_ = ctx.saved_tensors 
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradReverse(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        Potentially scales gradients with scalar alpha
        """
        super().__init__(*args, **kwargs)
        self.alpha = tensor(alpha, requires_grad=False)
        self.apply = GradReverseFunction.apply

    def forward(self, input_):
        return self.apply(input_, self.alpha)


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
    
    def forward(self, logits: Tensor, domain: Tensor, target_start_id: Maybe[int] = None) -> Tuple[Tensor, Tensor]:
        # if not given, assumes 50% source - 50% target in batch
        target_start_id = logits.shape[0] // 2 if target_start_id is None else target_start_id

        # discriminate domain of input 
        domain_loss = F.binary_cross_entropy_with_logits(logits, domain)

        # adversarial loss for the target data
        logits_target = logits[target_start_id:]
        adv_domain_target = torch.logical_not(domain)[target_start_id:].float()
        adv_domain_loss = F.binary_cross_entropy_with_logits(logits_target, adv_domain_target)

        return domain_loss, adv_domain_loss


class SoftTripletKLLoss(nn.Module):
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


class CNNEncoder(nn.Module):
    def __init__(self, emb_dim: int):

        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            )  

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            )  
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            ) 

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU(), 
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
                            nn.LeakyReLU()
                            ) 

        # bottleneck
        self.emb = nn.Linear(2048, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 3, stride=3)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.emb(x.flatten(1))
        return F.leaky_relu(x)


class CNNClassifier(nn.Module):
    def __init__(self, 
                num_classes: int, 
                emb_dim: int,
                dropout: float = 0.,
                input_normalize: bool = False
                ):
        super().__init__()
        self.encoder = CNNEncoder(emb_dim) 
        self.input_normalize = input_normalize
        self.classifier = nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, num_classes)).apply(self.init_weights)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.input_normalize:
            mean = x.mean(dim=[2,3], keepdim=True)
            std = torch.square(x - mean).mean(dim=[2,3], keepdim=True)
            x = (x - mean) / (std + 1e-5)

        x = self.encode(x)
        x = self.dropout(x)
        return self.classifier(x)


class DIRL(nn.Module):
    def __init__(self, 
                num_classes: int, 
                emb_dim: int,
                input_normalize: bool = False 
                ):
        super().__init__()
        self.input_normalize = input_normalize
        self.num_classes = num_classes

        # freeze gradients to remove unwanted contributions during adversarial step
        self.freeze_gradient = GradReverse(0)
        #self.reverse_gradient = GradReverse(-1)

        # CNN encoder
        self.encoder = CNNEncoder(emb_dim)

        # classifier
        self.cls = nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, num_classes)).apply(self.init_weights)

        # domain discriminator
        self.dd = nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, 2)).apply(self.init_weights)

        # conditional domain discriminator
        self.cdd = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 50),
                            nn.LeakyReLU(),
                            nn.Linear(50, 2)) for _ in range(num_classes)]).apply(self.init_weights)

        # parallel version
        # self.cdd_1 = nn.Linear(emb_dim, 100 * num_classes).apply(self.init_weights)
        # self.cdd_2 = nn.Conv1d(100 * num_classes, 50 * num_classes, kernel_size=1, groups=num_classes).apply(self.init_weights)
        # self.cdd_3 = nn.Conv1d(50 * num_classes, 2 * num_classes, kernel_size=1, groups=num_classes).apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor, freeze_gradient: bool = False) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        if self.input_normalize:
            mean = x.mean(dim=[2,3], keepdim=True)
            std = torch.square(x - mean).mean(dim=[2,3], keepdim=True)
            x = (x - mean) / (std + 1e-5)

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
        embs_d = self.freeze_gradient(embs) if freeze_gradient else embs

        # domain predictions 
        domain_preds = self.dd(embs_d)

        # class-conditioned domain predictions - flip aswell
        embs_f = self.freeze_gradient(embs) if freeze_gradient else embs
        cond_domain_preds = [f(embs_f) for f in self.cdd]

        # parallel version 
        # embs_f = self.freeze_gradient(embs) if freeze_gradient else embs
        # cond_domain_preds = self.cdd_1(embs_f).unsqueeze(-1) # B x 100K x 1
        # cond_domain_preds = F.leaky_relu(cond_domain_preds)
        # cond_domain_preds = self.cdd_2(cond_domain_preds) # B x 50K x 1
        # cond_domain_preds = F.leaky_relu(cond_domain_preds)
        # cond_domain_preds = self.cdd_3(cond_domain_preds) # B x 2K x 1
        # cond_domain_preds = list(cond_domain_preds.squeeze().view(self.num_classes, -1, 2)) # [K x] B x 2    

        return embs, domain_preds, class_preds, cond_domain_preds

