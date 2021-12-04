from typing import *
import os
import random
import numpy as np  
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 

from digits_dataset_loader import load_dataset, few_labels
from digits_model import DigitsDIRL
from dirl_losses import *


def batch_generator(data: List[Tensor], batch_size: int, shuffle=True, iter_shuffle=False):
    data = [torch.stack(d) for d in data]
    num = data[0].shape[0]
    
    def shuffle_aligned_list(data):
        p = np.random.permutation(num)
        return [d[p] for d in data]

    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= num:
          batch_count = 0
          if iter_shuffle:
            print("batch shuffling")
            data = shuffle_aligned_list(data)

    start = batch_count * batch_size
    end = start + batch_size
    batch_count += 1
    yield [d[start:end] for d in data]


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
    sup_id = sum(sizing[0:2])
    batch_size = sum(sizing)

    x_target_sup, y_target_sup = few_labels(target_data, target_labels, examples_per_class, num_classes)

    # source_dl = DataLoader(list(zip(source_data, source_labels)), shuffle=True, batch_size=sizing[0])
    # target_dl = DataLoader(list(zip(target_data, target_labels)), shuffle=True, batch_size=sizing[2])
    # source_dl_test = DataLoader(list(zip(source_data_test, source_labels_test)), shuffle=False, batch_size=sizing[0])
    # target_dl_test = DataLoader(list(zip(target_data_test, target_labels_test)), shuffle=False, batch_size=sizing[2])

    # batch generation and testing split
    gen_source = batch_generator([source_data, source_labels], sizing[0], iter_shuffle=False)
    gen_target = batch_generator([target_data, target_labels], sizing[2], iter_shuffle=True)

    num_test = min(5000, source_data_test.shape[0], target_data_test.shape[0])
    source_random_indices = list(range(source_data_test.shape[0]))
    target_random_indices = list(range(target_data_test.shape[0]))
    random.shuffle(source_random_indices)
    random.shuffle(target_random_indices)
    source_test_indices = source_random_indices[:num_test]
    target_test_indices = target_random_indices[:num_test]

    combined_test_imgs = torch.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]])
    combined_test_labels = torch.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]])
    combined_test_domain = torch.vstack([torch.tile(torch.tensor([1, 0.]), [num_test, 1]),
                                      torch.tile(torch.tensor([0, 1.]), [num_test, 1])])

    # load model
    model = DigitsDIRL().to(device)

    # define criteria and optimizers
    _cfg = {
        'domain_weight' : 1.0,
        'classify_weight' : 1.0,
        'triplet_weight'  : 1.0,
        'trplet_margin' : 0.7,
        'triplet_KL_weight' : 1.0,
        'triplet_KL_margin' : 0.6,
        'class_dann_weight' : 1.0
    }
    domain_crit = DomainLoss()
    classify_crit = ClassificationLoss()
    cdann_crit = ConditionalDomainLoss(num_classes)
    triplet_crit = SoftTripletLoss(margin=0.7, sigmas=[0.01, 0.1, 0.5, 1.1], l2_normalization=True)

    # construct domain annotations so that 50% source and 50% target
    domain_batch = torch.eye(2)[torch.tensor([0] * (batch_size // 2) + [1] * (batch_size // 2))].to(device)

    # define optimizer steps
    opt_A = Adam(model.parameters(), lr=1e-03)
    opt_B = Adam(model.encoder.parameters(), lr=1e-03)

    # training loop
    safety = 0
    print_strings = []
    classify_losses = []
    domain_losses = []
    cdann_losses = []
    triplet_losses = []
    cls_target_accus = []
    triplet_loss_list = []
    triplet_loss_KL = []
    the_l = []
    plot_losses = []
    max_test_run = 0

    for iteration in range(num_iterations):
        # concat data in batch
        x_batch_source, y_batch_source = next(gen_source)
        x_batch_target, y_batch_target = next(gen_target)
        x_batch = torch.stack([x_batch_source, x_target_sup, x_batch_target], dim=0).to(device)
        y_batch = torch.stack([y_batch_source, y_target_sup, y_batch_target], dim=0).to(device)

        model.train()
        opt_A.zero_grad()
        opt_B.zero_grad()

        triplet_loss = torch.tensor(-1., device=device)
        if iteration > 300:
            embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch)

            triplet_loss = triplet_crit(embs[:sup_id], y_batch[:sup_id])
            _, adv_domain_loss = domain_crit(domain_preds, domain_batch)
            _, cdann_loss_adv = cdann_crit(cdann_preds[:sup_id], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])
            
            loss = _cfg['class_dann_weight'] * cdann_loss_adv + _cfg['domain_weight'] * adv_domain_loss + triplet_loss

            # backprop adversarial step
            adv_loss.backward()
            opt_B.step()

            triplet_losses.append(triplet_loss.item())

        embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch, freeze_gradient=True)

        classify_loss = classify_crit(class_preds[:sup_id], y_batch[:sup_id])
        domain_loss, _ = domain_crit(domain_preds, domain_batch)
        cdann_loss, _ = cdann_crit(cdann_preds[:sup_id], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])

        loss = _cfg['classify_weight'] * classify_loss + _cfg['class_dann_weight'] * cdann_loss + _cfg['domain_weight'] * domain_loss

        # backprop classification step        
        loss.backward()
        opt_A.step()

        # compute and update metrics 
        domain_accu = (domain.argmax(1) == domain_preds.argmax(1)).mean(0).item() 
        interim = (y_batch.argmax(1) == class_preds.argmax(1)).float()
        cls_source_accu = interim[:sizing[0]].mean(0).item()
        cls_target_sup_accu = interim[sizing[0] : sup_id].mean(0).item()
        cls_target_accus.append(interim[sizing[0]:].mean(0).item())

        # test
        with torch.no_grad():
            model.eval()
            _size = combined_test_imgs.shape[0]
            out = model.forward(combined_test_imgs, combined_test_labels, combined_test_domain)
            interim = (combined_test_labels.argmax(1) == out[2].argmax(1)).float()
            source_accuracy_fin = interim[:_size // 2].mean(0)
            target_accuracy_fin = interim[_size // 2:].mean(0)

        domain_losses.append(domain_loss.item())
        classify_losses.apend(classify_loss.item())
        cdann_losses.append(cdann_loss.item())
        plot_losses.append(domain_loss.item(), classify_loss.item(), cdann_loss.item(), triplet_loss.item())
        
        if iteration % 100 == 0:
            epoch_string = "Iteration={}, batch losses: {:.4f} {:.4f} {:.4f} {:.4f} \t batch accuracies: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                iteration, domain_losses[-1], classify_losses[-1], cdann_losses[-1], triplet_losses[-1], domain_accu, cls_source_accu, 
                cls_target_sup_accu, cls_target_accus[-1])
            print_strings.append(epoch_string)
            print(epoch_string)

            if iteration > 300:
                max_test_run = max(max_test_run, target_accuracy_fin)
            print('Source Acc: {:.3f}, \t Target Acc: {:.3f}, \t Max Target Acc: {:.3f}'.format(source_accuracy_fin, target_accuracy_fin, max_test_run))

    if save_results:
        pass


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