import os
from typing import * 
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 

from digits_dataset_loader import load_dataset, few_labels
from digits_model import DigitsDIRL
from dirl_losses import *


def plot_embedding(X, y, d, title=None, save_fig_path='tmp.png'):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    print(x_min, x_max)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(7,7))
#     ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'size': 14})

    plt.xticks([]), plt.yticks([])
    plt.xlim(min(X[:,0]), max(X[:,0])+0.05)
    plt.ylim(min(X[:,1]), max(X[:,1]) + 0.05)
    if title is not None:
        plt.title(title)

    plt.savefig(save_fig_path, format='png', bbox_inches='tight', pad_inches=2)


def main(mode: str, 
         source: str, 
         target: str, 
         examples_per_label: int,
         num_iterations: int,
         config_path: str,
         save_results: bool,
         num_classes: int,
         device: str
        ):
    # create directory to save results
    logdir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # load and partition data
    source_data, source_data_test, source_labels, source_labels_test = load_dataset(source)
    target_data, target_data_test, target_labels, target_labels_test = load_dataset(target)

    target_sup_size = examples_per_class * num_classes
    sizing = [target_sup_size * 4, target_sup_size, target_sup_size * 3]
    batch_size = sum(sizing)

    target_data_sup, target_labels_sup = few_labels(target_data, target_labels, examples_per_class, num_classes)

    source_dl = DataLoader(list(zip(source_data, source_labels)), shuffle=True, batch_size=sizing[0])
    target_dl = DataLoader(list(zip(target_data, target_labels)), shuffle=True, batch_size=sizing[2])
    source_dl_test = DataLoader(list(zip(source_data_test, source_labels_test)), shuffle=False, batch_size=sizing[0])
    target_dl_test = DataLoader(list(zip(target_data_test, target_labels_test)), shuffle=False, batch_size=sizing[2])

    # load model
    model = DigitsDIRL().to(device)

    # define criteria and optimizers
    domain_crit = DomainLoss(batch_size)
    classify_crit = ClassificationLoss()
    cond_domain_crit = ConditionalDomainLoss(num_classes, sizing)
    soft_triplet_crit = SoftTripletLoss(margin=0.7, sigmas=[0.01, 0.1, 0.5, 1.1], l2_normalization=True)

    opt_A = Adam(model.parameters(), lr=1e-03)
    opt_B = Adam(model.encoder.parameters(), lr=1e-03)

    # training loop
    num_test = min(5000, source_data_test.shape[0], target_data_test.shape[0])
    safety = 0
    _cfg = {
        'domain_weight' : 1.0,
        'classify_weight' : 1.0,
        'triplet_weight'  : 1.0,
        'trplet_margin' : 0.7,
        'triplet_KL_weight' : 1.0,
        'triplet_KL_margin' : 0.6,
        'class_dann_weight' : 1.0
    }

    for iteration in range(num_iterations):
        # concat data in batch
        x_batch_source, y_batch_source = next(source_dl.__iter__())
        x_batch_target, y_batch_target = next(target_dl.__iter__())
        x_batch = torch.cat([x_batch_source, target_data_sup, x_batch_target], dim=0)
        y_batch = torch.cat([y_batch_source, target_labels_test, y_batch_target], dim=0)

        # forward pass
        model.train()
        embs, domain_preds, class_preds, cond_domain_preds = model(x_batch)
        
        # losses computation
        classify_loss = classify_crit(class_preds, y_batch)
        triplet_loss = soft_triplet_crit(embs, y_batch)
        domain_loss, adv_domain_loss = domain_crit(domain_preds)
        cond_domain_loss, cond_domain_loss_adv = cond_domain_crit(cond_domain_preds, y_batch)
            
        # adversarial step
        if iteration > 300:
            adv_loss = _cfg['class_dann_weight'] * cond_domain_loss_adv + _cfg['domain_weight'] * adv_domain_loss + triplet_loss

            # backprop adversarial step
            adv_loss.backward()
            opt_B.step()
            opt_B.zero_grad()

        # backprop discrimination step
        loss = _cfg['classify_weight'] * classify_loss + _cfg['class_dann_weight'] * cond_domain_loss + _cfg['domain_weight'] * domain_loss
        loss.backward()
        opt_A.step()
        opt_A.zero_grad()

        # compute accuracy metrics
        







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="dirl", choices=["source_only", "triplet", "dann", "dirl"])
    parser.add_argument('-source', type=str, default='mnist', choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-target', type=str, default='svhn', choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-examples_per_label', type=int, default=10)
    parser.add_argument('-num_classes', type=int, default=10)
    parser.add_argument('-num_iterations', type=int, default=None)
    parser.add_argument('-config_path', type=str, default='./configs/dirl_digits_config.yml')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save_results', type=bool, default=False)

    kwargs = vars(parser.parse_args())
    main(**kwargs)