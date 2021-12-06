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

# reproducability
SEED = torch.manual_seed(1312)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(1312)
random.seed(SEED)


class BatchGenerator(object):
    Sample = List[Tuple[Tensor, Tensor]]

    def __init__(self, 
                 dataset: List[Tuple[Tensor, Tensor]],
                 batch_size: int, 
                 shuffle: bool=True, 
                 iter_shuffle: bool=False
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.iter_shuffle = iter_shuffle 
        self.batch_count = 0 
        self.total_num = len(dataset)

    def _shuffle(self, ds: List[Any]) -> List[Any]:
        return ds if not self.shuffle else random.sample(ds, len(ds))

    def __next__(self) -> Sample:
        if (self.batch_count + 1) * self.batch_size > self.total_num:
            self.batch_count = 0
            if self.iter_shuffle:
                print("batch shuffling...")
                self.dataset = random.sample(self.dataset, self.total_num)

        start = self.batch_size * self.batch_count
        end = start + self.batch_size
        self.batch_count += 1
        return self._shuffle(self.dataset[start:end])


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
         examples_per_class: int,
         num_iterations: int,
         #config_path: str,
         save_results: bool,
         num_classes: int,
         device: str
        ):
    # create directory to save results
    logdir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # load and partition data
    source_data, source_labels, source_data_test, source_labels_test = load_dataset(source)
    target_data, target_labels, target_data_test, target_labels_test = load_dataset(target)

    target_sup_size = examples_per_class * num_classes
    sizing = [int(target_sup_size * 3.25), target_sup_size, int(target_sup_size * 2.25)]
    sup_id = sum(sizing[0:2])
    batch_size = sum(sizing)

    x_target_sup, y_target_sup = few_labels(target_data, target_labels, examples_per_class, num_classes)

    # source_dl = DataLoader(list(zip(source_data, source_labels)), shuffle=True, batch_size=sizing[0])
    # target_dl = DataLoader(list(zip(target_data, target_labels)), shuffle=True, batch_size=sizing[2])
    # source_dl_test = DataLoader(list(zip(source_data_test, source_labels_test)), shuffle=False, batch_size=sizing[0])
    # target_dl_test = DataLoader(list(zip(target_data_test, target_labels_test)), shuffle=False, batch_size=sizing[2])

    # batch generation and testing split
    gen_batch_source = BatchGenerator(list(zip(source_data, source_labels)), sizing[0], iter_shuffle=False)
    gen_batch_target = BatchGenerator(list(zip(target_data, target_labels)), sizing[2], iter_shuffle=True)

    num_test = min(5000, len(source_data_test), len(target_data_test))
    source_random_indices = list(range(len(source_data_test)))
    target_random_indices = list(range(len(target_data_test)))
    random.shuffle(source_random_indices)
    random.shuffle(target_random_indices)
    source_test_indices = torch.tensor(source_random_indices[:num_test])
    target_test_indices = torch.tensor(target_random_indices[:num_test])

    source_data_test = torch.stack(source_data_test)
    target_data_test = torch.stack(target_data_test)
    source_labels_test = torch.stack(source_labels_test)
    target_labels_test = torch.stack(target_labels_test)
    combined_test_imgs = torch.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]]).to(device)
    combined_test_labels = torch.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]]).to(device)
    combined_test_domain = torch.vstack([torch.tile(torch.tensor([1, 0.]), [num_test, 1]),
                                      torch.tile(torch.tensor([0, 1.]), [num_test, 1])]).to(device)

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
    triplet_crit = SoftTripletLoss(margin=_cfg['triplet_KL_margin'], sigmas=[0.01, 0.1, 0.5, 1.1], l2_normalization=True)

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
    print('Training...')
    for iteration in range(num_iterations):
        # concat data in batch
        x_batch_source, y_batch_source = zip(*next(gen_batch_source))
        x_batch_target, y_batch_target = zip(*next(gen_batch_target))
        x_batch = torch.stack(x_batch_source + x_target_sup + x_batch_target, dim=0).to(device)
        y_batch = torch.stack(y_batch_source + y_target_sup + y_batch_target, dim=0).to(device)

        model.train()
        #opt_A.zero_grad()
        #opt_B.zero_grad()

        triplet_loss = None
        if iteration > 300:
            #opt_B.zero_grad()
            embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch)

            triplet_loss = triplet_crit(embs[:sup_id], y_batch[:sup_id])
            _, adv_domain_loss = domain_crit(domain_preds, domain_batch)
            _, cdann_loss_adv = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])
            
            loss_B = _cfg['class_dann_weight'] * cdann_loss_adv + _cfg['domain_weight'] * adv_domain_loss + triplet_loss

            # backprop adversarial step
            loss_B.backward()
            opt_B.step()
            opt_B.zero_grad()

            triplet_loss = round(triplet_loss.item(), 4)

        #opt_A.zero_grad()
        embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch, freeze_gradient=True)

        classify_loss = classify_crit(class_preds[:sup_id], y_batch[:sup_id])
        domain_loss, _ = domain_crit(domain_preds, domain_batch)
        cdann_loss, _ = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])

        loss_A = _cfg['classify_weight'] * classify_loss + _cfg['class_dann_weight'] * cdann_loss + _cfg['domain_weight'] * domain_loss

        # backprop classification step        
        loss_A.backward()
        opt_A.step()
        opt_A.zero_grad()

        # test
        with torch.no_grad():
            # compute and update metrics 
            domain_accu = (domain_batch.argmax(1) == domain_preds.argmax(1)).float().mean(0).item() 
            interim = (y_batch.argmax(1) == class_preds.argmax(1)).float()
            cls_source_accu = interim[:sizing[0]].mean(0).item()
            cls_target_sup_accu = interim[sizing[0] : sup_id].mean(0).item()
            cls_target_accus.append(interim[sizing[0]:].mean(0).item())
            
            model.eval()
            _size = combined_test_imgs.shape[0]
            out = model.forward(combined_test_imgs)
            interim = (combined_test_labels.argmax(1) == out[2].argmax(1)).float()
            source_accuracy_fin = interim[:_size // 2].mean(0)
            target_accuracy_fin = interim[_size // 2:].mean(0)

        domain_losses.append(domain_loss.item())
        classify_losses.append(classify_loss.item())
        cdann_losses.append(cdann_loss.item())
        triplet_losses.append(triplet_loss)
        plot_losses.append([domain_loss.item(), classify_loss.item(), cdann_loss.item(), triplet_loss])
        
        if iteration % 100 == 0:
            epoch_string = "Iteration={}, batch losses: {:.4f} {:.4f} {:.4f} {} \t batch accuracies: {:.3f} {:.3f} {:.3f} {:.3f}".format(
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
    parser.add_argument('-target', type=str, default='usps', choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-examples_per_class', type=int, default=10)
    parser.add_argument('-num_classes', type=int, default=10)
    parser.add_argument('-num_iterations', type=int, default=10000)
    #parser.add_argument('-config_path', type=str, default='./configs/dirl_digits_config.yml')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save_results', type=bool, default=False)

    kwargs = vars(parser.parse_args())
    main(**kwargs)