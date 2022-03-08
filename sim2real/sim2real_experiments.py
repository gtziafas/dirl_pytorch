from _types import *
import os
import yaml
import random
import numpy as np  
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW, Optimizer
from sklearn.neighbors import KDTree
from sklearn.metrics import average_precision_score, silhouette_score
import matplotlib
from math import ceil
matplotlib.use('TkAgg')  # Need Tk for interactive plots.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import imgaug.augmenters as iaa

from sim2real_dataset_loader import load_datasets
from dirl_sim2real import *

# reproducability
SEED = torch.manual_seed(27)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(27)
np.random.seed(27)
random.seed(27)


def few_labels(data: Tensor, labels: Tensor, num_pts: int, num_classes: int):
    num_samples = data.shape[0]
    data_subset, labels_subset = [], []
    for class_id in range(num_classes):
        filterr = torch.where(labels == class_id)
        select_pts = random.sample(list(range(filterr[0].shape[0])), min(filterr[0].shape[0], num_pts))
        #data_subset.append(data[filterr][0:num_pts])
        #labels_subset.append(labels[filterr][0:num_pts])
        data_subset.append(data[filterr][select_pts])
        labels_subset.append(labels[filterr][select_pts])
    return torch.cat(data_subset, dim=0), torch.cat(labels_subset, dim=0) 


def train_epoch_classifier(model: nn.Module, 
                           dl: DataLoader,
                           opt: Optimizer, 
                           crit: nn.Module,
                           device: str
                           ) -> Tuple[float, float]:
    model.train()
    epoch_loss, total_correct = 0., 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        preds = model.forward(x)
        loss = crit(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(dl)
    accuracy = total_correct / len(dl.dataset)

    return epoch_loss, accuracy


@torch.no_grad()
def eval_epoch_classifier(model: nn.Module, 
                           dl: DataLoader,
                           crit: nn.Module,
                           device: str
                           ) -> Tuple[float, float]:
    model.eval()
    epoch_loss, total_correct = 0., 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        preds = model.forward(x)
        loss = crit(preds, y)
        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(dl)
    accuracy = total_correct / len(dl.dataset)

    return epoch_loss, accuracy


def real_baseline( 
         num_epochs: int,
         batch_size: int,
         dropout: float,
         lr: float,
         wd: float,
         emb_dim: int,
         device: str,
         checkpoint: Maybe[str] = None
        ):
    # load data
    print('Loading data...')
    if checkpoint is None:
        _, (xs, _, ys) = load_datasets()
    else:
        xs, _, ys = torch.load(checkpoint)['real']
    dataset = list(zip(xs, ys))

    test_size = ceil(.2 * len(dataset))
    train_ds, test_ds = random_split(dataset, [len(dataset) - test_size, test_size])
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    # init model and stuff
    model = CNNClassifier(num_classes= ys.max().item() + 1, emb_dim=emb_dim, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(reduction='mean').to(device)

    print('Training....')
    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch_classifier(model, train_dl, opt, crit, device)
        test_loss, test_accu = eval_epoch_classifier(model, test_dl, crit, device)

        print(f'Epoch {epoch+1}/{num_epochs}: train loss={train_loss:.4f}, accuracy={train_accu:.3f}, \
                \t test loss={test_loss:.4f}, accuracy={test_accu:.3f}')

    all_preds = []
    for x,y in test_dl:
        x = x.to(device)
        y = y.to(device)
        preds = model(x).argmax(-1)
        all_preds.append(preds)
    all_labels = torch.stack([t[1] for t in test_ds]).unsqueeze(1).cpu().numpy()
    all_preds = torch.cat(all_preds, dim=0).unsqueeze(1).cpu().numpy()
    print(f'REAL only SS = {silhouette_score(all_labels, all_preds)}')
    print(f'REAL only mAP = {average_precision_score(all_labels, all_preds)}')


def finetune_baselines(
         pretrain: bool,
         num_sim_epochs: int,
         num_ft_epochs: int,
         batch_size: int,
         dropout: float,
         lr: float,
         wd: float,
         emb_dim: int,
         device: str,
         checkpoint: Maybe[str] = None
        ):
    # load data
    print('Loading data...')
    if checkpoint is None:
       (xs_sim, _, ys_sim), (xs_real, _, ys_real) = load_datasets()
    else:
        chp = torch.load(checkpoint)
        xs_sim, _, ys_sim = chp['sim']
        xs_real, _, ys_real = chp['real']
    assert ys_sim.max() == ys_real.max()
    num_classes = ys_sim.max().item() + 1 

    sim_test_size, real_test_size = ceil(.2 * xs_sim.shape[0]), ceil(.2 * xs_real.shape[0])
    sim_train, sim_test = random_split(list(zip(xs_sim, ys_sim)), [xs_sim.shape[0] - sim_test_size, sim_test_size])
    real_train, real_test = random_split(list(zip(xs_real, ys_real)), [xs_real.shape[0] - real_test_size, real_test_size])
    
    model = CNNClassifier(num_classes=num_classes, emb_dim=emb_dim, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(reduction='mean').to(device)

    # first train in sim if desired
    if pretrain:
        sim_train_dl = DataLoader(sim_train, shuffle=True, batch_size=batch_size)
        sim_test_dl = DataLoader(sim_test, shuffle=False, batch_size=batch_size)

        print('Training model in synthetic data....')
        for epoch in range(num_sim_epochs):
            train_loss, train_accu = train_epoch_classifier(model, sim_train_dl, opt, crit, device)
            test_loss, test_accu = eval_epoch_classifier(model, sim_test_dl, crit, device)

            print(f'Epoch {epoch+1}/{num_sim_epochs}: train loss={train_loss:.4f}, accuracy={train_accu:.3f}, \
                    \t test loss={test_loss:.4f}, accuracy={test_accu:.3f}')

    real_test_dl = DataLoader(real_test, shuffle=False, batch_size=batch_size)
    loss, accu = eval_epoch_classifier(model, real_test_dl, crit, device)
    print(f'0-shot performance in real test split: Loss = {loss:.4f}, Accuracy={accu:.3f}')

    all_preds = []
    for x,y in real_test_dl:
        x = x.to(device)
        y = y.to(device)
        preds = model(x).argmax(-1)
        all_preds.extend(preds.detach().cpu().numpy())
    all_labels = [t[1] for t in real_test].cpu().numpy()
    print(f'SIM only SS = {silhouette_score(all_labels, all_preds)}')
    print(f'SIM only mAP = {average_precision_score(all_labels, all_preds)}')

    # fine-tuning experiments
    for k in [1, 5, 10, 20, 50, 100]:
        opt_ft = AdamW(model.parameters(), lr=1e-04, weight_decay=0.001)
        x_real_ft, y_real_ft = few_labels(torch.stack([t[0] for t in real_train]),
                                          torch.stack([t[1] for t in real_train]),
                                          num_pts = k,
                                          num_classes = num_classes)

        dl = DataLoader(list(zip(x_real_ft, y_real_ft)), shuffle=True, batch_size=6)
        
        print(f'Finetuning in real data using {k} labeled examples per class...')
        best = (0, 0,0)
        for epoch in range(num_ft_epochs):
            ft_loss, ft_accu = train_epoch_classifier(model, dl, opt_ft, crit, device)
            test_loss, test_accu = eval_epoch_classifier(model, real_test_dl, crit, device)
            print(f'Finetuning {epoch+1}/{num_ft_epochs}: loss = {ft_loss:.4f}, accuracy={ft_accu:.3f}, \
                \t Test loss={test_loss:.4f}, accuracy={test_accu:.3f}')
            if test_accu > best[-1]:
                best = (epoch+1, test_loss, test_accu)

        print(f'{k}-shot performance in real test split: Epoch={best[0]}, Loss={best[1]:.4f}, Accuracy={best[2]:.3f}')
        print()


def naive_baseline(
         num_pts: int,
         num_epochs: int,
         batch_size: int,
         dropout: float,
         lr: float,
         wd: float,
         emb_dim: int,
         device: str,
         checkpoint: Maybe[str] = None
        ):
    # load data
    print('Loading data...')
    if checkpoint is None:
       (xs_sim, _, ys_sim), (xs_real, _, ys_real) = load_datasets()
    else:
        chp = torch.load(checkpoint)
        xs_sim, _, ys_sim = chp['sim']
        xs_real, _, ys_real = chp['real']
    assert ys_sim.max() == ys_real.max()
    num_classes = ys_sim.max().item() + 1 

    sim_test_size, real_test_size = ceil(.2 * xs_sim.shape[0]), ceil(.2 * xs_real.shape[0])
    sim_train, sim_test = random_split(list(zip(xs_sim, ys_sim)), [xs_sim.shape[0] - sim_test_size, sim_test_size])
    real_train, real_test = random_split(list(zip(xs_real, ys_real)), [xs_real.shape[0] - real_test_size, real_test_size])
    
    model = CNNClassifier(num_classes=num_classes, emb_dim=emb_dim, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(reduction='mean').to(device)


    x_real_sup, y_real_sup =  few_labels([t[0] for t in real_train], 
                                      [t[1] for t in real_train],
                                      num_pts, num_classes)
    ds_train = sim_train + list(zip(list(x_real_sup), list(y_real_sup)))

    train_dl = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
    real_test_dl = DataLoader(real_test, shuffle=False, batch_size=batch_size)
    sim_test_dl = DataLoader(sim_test, shuffle=False, batch_size=batch_size)

    print('Training model in synthetic + real (few-shot) data....')
    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch_classifier(model, train_dl, opt, crit, device)
        sim_test_loss, sim_test_accu = eval_epoch_classifier(model, sim_test_dl, crit, device)
        real_test_loss, real_test_accu = eval_epoch_classifier(model, real_test_dl, crit, device)

        print(f'Epoch {epoch+1}/{num_sim_epochs}: train loss={train_loss:.4f}, accuracy={train_accu:.3f}, \
                \t SIM loss={sim_test_loss:.4f}, SIM accuracy={sim_test_accu:.3f}, \
                \t REAL loss={real_test_loss:.4f}, REAL accuracy={real_test_accu:.3f}')

    all_preds = []
    for x,y in real_test_dl:
        x = x.to(device)
        y = y.to(device)
        preds = model(x).argmax(-1)
        all_preds.extend(preds.detach().cpu().numpy())
    all_labels = [t[1] for t in real_test].cpu().numpy()
    print(f'SIM only SS = {silhouette_score(all_labels, all_preds)}')
    print(f'SIM only mAP = {average_precision_score(all_labels, all_preds)}')


#real_baseline(15, 64, 0.0, 1e-3, 0., 128, "cuda", "checkpoints/object75_datasets.p")
#finetune_baselines(False, 8, 20, 128, 0.0, 1e-3, 0., 128, "cuda", "checkpoints/object75_datasets.p")
#naive_baseline(10, 15, 128, 0., 1e-3, 0., 128, "cuda", "checkpoints/object75_datasets.p")


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


def augment_target(images: List[array]) -> List[array]:
    seq = iaa.Sequential([
        #iaa.Affine(rotate=(-25, 25)),
        #iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.Crop(percent=(0, 0.2)),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
    ])
    return seq(images=images)


def main(examples_per_class: int,
         config_path: str,
         save_results: bool,
         augment_real: int,
         device: str
        ):
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    # create directory to save results
    logdir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # load and partition data
    print('Loading data...')
    if cfg['dataset']['datasets_checkpoint'] is None:
       (xs_sim, _, ys_sim), (xs_real, _, ys_real) = load_datasets()
    else:
        chp = torch.load(cfg['dataset']['datasets_checkpoint'])
        xs_sim, _, ys_sim = chp['sim']
        xs_real, _, ys_real = chp['real']
    assert ys_sim.max() + 1 == ys_real.max() + 1 == cfg['dataset']['num_classes']
    num_classes = cfg['dataset']['num_classes']
    # convert to one-hot
    ys_sim = torch.eye(num_classes)[ys_sim]
    ys_real = torch.eye(num_classes)[ys_real]

    sim_test_size, real_test_size = ceil(.2 * xs_sim.shape[0]), ceil(.2 * xs_real.shape[0])
    sim_train, sim_test = random_split(list(zip(xs_sim, ys_sim)), [xs_sim.shape[0] - sim_test_size, sim_test_size])
    real_train, real_test = random_split(list(zip(xs_real, ys_real)), [xs_real.shape[0] - real_test_size, real_test_size])

    target_sup_size = examples_per_class * num_classes
    target_sup_size = 60
    sizing = [int(target_sup_size * s) for s in cfg['dataset']['sizing']]
    sup_id = sum(sizing[0:2])
    batch_size = sum(sizing)

    x_target_sup, y_target_sup = few_labels(xs_real, ys_real.argmax(-1), examples_per_class, num_classes)
    if augment_real:
        x_target_sup = [(x*0xff).transpose(0,2).numpy().astype('uint8') for x in x_target_sup]
        x_target_sup = augment_target(sum([[x] * augment_real for x in list(x_target_sup)], []))
        y_target_sup = sum([[y] * augment_real for y in list(y_target_sup)], [])
        x_target_sup = torch.stack([torch.tensor(x, dtype=floatt).div(0xff).transpose(2,0) for x in x_target_sup])
        y_target_sup = torch.stack(y_target_sup)   
    y_target_sup = torch.eye(num_classes)[y_target_sup]
    dl_target_sup = DataLoader(list(zip(x_target_sup, y_target_sup)), batch_size=sizing[1], shuffle=True)

    # batch generation and testing split
    dl_source = DataLoader(list(zip(xs_sim, ys_sim)), batch_size=sizing[0], shuffle=True, drop_last=True)
    dl_target = DataLoader(list(zip(xs_real, ys_real)), batch_size=sizing[2], shuffle=True, drop_last=True)

    source_data_test, source_labels_test = [torch.stack(x) for x in zip(*sim_test)]
    target_data_test, target_labels_test = [torch.stack(x) for x in zip(*real_test)]
    num_test = min(cfg['model']['test_size'], len(source_data_test), len(target_data_test))
    source_random_indices = list(range(len(source_data_test)))
    target_random_indices = list(range(len(target_data_test)))
    random.shuffle(source_random_indices)
    random.shuffle(target_random_indices)
    source_test_indices = torch.tensor(source_random_indices[:num_test])
    target_test_indices = torch.tensor(target_random_indices[:num_test])

    combined_test_imgs = torch.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]]).to(device)
    combined_test_labels = torch.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]]).to(device)
    combined_test_domain = torch.vstack([torch.tile(torch.tensor([1, 0.]), [num_test, 1]),
                                      torch.tile(torch.tensor([0, 1.]), [num_test, 1])]).to(device)

    # load model
    model = DIRL(num_classes=num_classes,
                 emb_dim=cfg['model']['embedding_size'],
                 input_normalize=cfg['model']['input_normalize']).to(device)

    # define criteria
    domain_crit = DomainLoss()
    classify_crit = ClassificationLoss()
    cdann_crit = ConditionalDomainLoss(num_classes)
    triplet_crit = SoftTripletKLLoss(margin=cfg['loss']['triplet_KL_margin'], 
                                     sigmas=cfg['loss']['triplet_KL_sigmas'], 
                                     l2_normalization=cfg['loss']['triplet_KL_l2_normalize'])

    # construct domain annotations so that 50% source and 50% target
    domain_batch = torch.eye(2)[torch.tensor([0] * (batch_size // 2) + [1] * (batch_size // 2))].to(device)

    # define optimizer steps
    # opt_A =  Adam([ 
    #              {'params': model.encoder.parameters(), 'weight_decay': cfg['model']['weight_decay']}, 
    #              {'params': model.cls.parameters()}, 
    #              {'params': model.dd.parameters()}, 
    #              {'params': model.cdd.parameters()},
    #          ], lr=cfg['model']['learning_rate'])  
    opt_A = Adam(model.parameters(), lr=cfg['model']['learning_rate'])
    opt_B = Adam(model.encoder.parameters(), lr=cfg['model']['learning_rate'])

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
    for iteration in range(cfg['model']['num_iterations']):
        # concat data in batch
        x_batch_source, y_batch_source = next(dl_source.__iter__())
        x_batch_target, y_batch_target = next(dl_target.__iter__())
        x_batch_target_sup, y_batch_target_sup = next(dl_target_sup.__iter__())

        x_batch = torch.cat((x_batch_source, x_batch_target_sup, x_batch_target), dim=0).to(device)
        y_batch = torch.cat((y_batch_source, y_batch_target_sup, y_batch_target), dim=0).to(device)

        model.train()

        triplet_loss = None
        if iteration > 300:
            #opt_B.zero_grad()
            embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch)

            #triplet_loss = triplet_crit(embs[:sup_id], y_batch[:sup_id])
            _, adv_domain_loss = domain_crit(domain_preds, domain_batch)
            #_, cdann_loss_adv = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])
            
            #loss_B = cfg['loss']['class_dann_weight'] * cdann_loss_adv + cfg['loss']['domain_weight'] * adv_domain_loss + triplet_loss
            loss_B = cfg['loss']['domain_weight'] * adv_domain_loss

            # backprop adversarial step
            opt_B.zero_grad()
            loss_B.backward()
            opt_B.step()

            #triplet_loss = round(triplet_loss.item(), 4)

        #opt_A.zero_grad()
        embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch, freeze_gradient=True)

        classify_loss = classify_crit(class_preds[:sup_id], y_batch[:sup_id])
        domain_loss, _ = domain_crit(domain_preds, domain_batch)
        #cdann_loss, _ = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])

        #loss_A = cfg['loss']['classify_weight'] * classify_loss + cfg['loss']['class_dann_weight'] * cdann_loss + cfg['loss']['domain_weight'] * domain_loss
        loss_A = cfg['loss']['classify_weight'] * classify_loss + cfg['loss']['domain_weight'] * domain_loss

        # backprop classification step
        opt_A.zero_grad()        
        loss_A.backward()
        opt_A.step()

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
        #domain_losses.append(-1)
        classify_losses.append(classify_loss.item())
        #cdann_losses.append(cdann_loss.item())
        cdann_losses.append(-1)
        triplet_losses.append(triplet_loss)
        plot_losses.append([domain_losses[-1], classify_losses[-1], cdann_losses[-1], triplet_losses[-1]])
        
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
        # plot loss functions
        plt.figure(figsize=(10, 5))
        plt.plot(range(cfg['model']['num_iterations']), plot_losses)
        plt.legend(['domain_loss', 'classify_loss', 'class_dann_loss', 'triplet_loss'])
        plt.savefig(os.getcwd() + '/results/dirl_digits_losses_dirl' + '-' + 'sim' + '-'+ 'real' + '-' + str(examples_per_class) + '.png', format='png', bbox_inches='tight',
                    pad_inches=2)

    print("Max Training Accuracy: ", max_test_run)
    # evaluate accuracy on source and target domains
    # c1s, c1t = sess.run([source_accuracy_fin, target_accuracy_fin],
    #                     feed_dict={x: combined_test_imgs, y: combined_test_labels, domain: combined_test_domain})
    
    # # domain accuracy
    # domain_accur_both = sess.run(domain_accuracy, feed_dict={x: combined_test_imgs, domain: combined_test_domain})
    # domain_accur_source = sess.run(domain_accuracy,
    #                                feed_dict={x: source_data_test[source_test_indices], domain: np.tile([1, 0.], [num_test, 1])})
    # domain_accur_target = sess.run(domain_accuracy,
    #                                feed_dict={x: target_data_test[target_test_indices], domain: np.tile([0, 1.], [num_test, 1])})

    model = model.eval().cpu()
    with torch.no_grad():
        _, domain_preds_test, class_preds_test, cdann_preds_test = model.forward(combined_test_imgs.cpu())
        embs_source, domain_preds_source, class_preds_source, cdann_preds_source = model.forward(source_data_test[source_test_indices])
        embs_target, domain_preds_target, class_preds_target, cdann_preds_target = model.forward(target_data_test[target_test_indices])

    domain_accur_both = (combined_test_domain.cpu().argmax(1) == domain_preds_test.argmax(1)).float().mean(0).item() 
    domain_accur_source = (torch.tile(torch.tensor([1., 0.]), [num_test, 1]).argmax(1) == domain_preds_source.argmax(1)).float().mean(0).item() 
    domain_accur_target = (torch.tile(torch.tensor([0., 1.]), [num_test, 1]).argmax(1) == domain_preds_source.argmax(1)).float().mean(0).item() 
    print("Domain: both", domain_accur_both, "\nsource:", domain_accur_source, "\ntarget:", domain_accur_target)

    # class accuracy
    interim = (combined_test_labels.cpu().argmax(1) == class_preds_test.argmax(1)).float()
    _size = combined_test_imgs.shape[0]
    class_accur_source = interim[:_size // 2].mean(0).item()
    class_accur_target = interim[_size // 2:].mean(0).item()
    # class_accur_source = sess.run(classify_source_accuracy, feed_dict={x: combined_test_imgs, y: combined_test_labels})
    # class_accur_target = sess.run(classify_target_accuracy, feed_dict={x: combined_test_imgs, y: combined_test_labels})

    print("classification:\nsource:", class_accur_source, "\ntarget:", class_accur_target)

    # KNN accuracy
    neighbors = cfg['model']['k_neighbours']
    # source_emb = sess.run(x_features, feed_dict={x: source_data_test[source_test_indices]})
    # target_emb = sess.run(x_features, feed_dict={x: target_data_test[target_test_indices]})
    kdt = KDTree(embs_source.cpu().numpy(), leaf_size=30, metric='euclidean')
    neighbor_idx = kdt.query(embs_target.cpu().numpy(), k=1 + neighbors, return_distance=False)[:, 1:]
    print(neighbor_idx.shape)
    neighbor_label = source_labels_test[source_test_indices][neighbor_idx]
    neighbor_label_summed = neighbor_label.sum(1)
    knn_accuracy = (target_labels_test[target_test_indices].argmax(1) == neighbor_label_summed.argmax(1)).float().mean().item()
    print(neighbor_label.shape, neighbor_label_summed.shape)
    print("Accuracy of knn on labels in z space (source)(on test)", knn_accuracy)

    # plot t-sne embedding
    num_test = cfg['model']['tsne_test_size']  # tsne
    num_test = min(num_test, source_data_test.shape[0], target_data_test.shape[0])
    source_test_indices = source_random_indices[:num_test]
    target_test_indices = target_random_indices[:num_test]
    combined_test_imgs = torch.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]])
    combined_test_labels = torch.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]])
    combined_test_domain = torch.vstack([torch.tile(torch.tensor([1, 0.]), [num_test, 1]),
                                      torch.tile(torch.tensor([0, 1.]), [num_test, 1])])

    with torch.no_grad():
        dann_emb, _, _, _  = model.forward(combined_test_imgs)
    # dann_emb = sess.run(x_features, feed_dict={x: combined_test_imgs, domain: combined_test_domain})
    dann_tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=400).fit_transform(dann_emb.numpy())
    if save_results:
        plot_embedding(dann_tsne, combined_test_labels.argmax(1).numpy(), combined_test_domain.numpy().reshape(-1), '', 
            save_fig_path=os.getcwd() + '/results/dirl_digits_tsne_plot_dirl' + '-' + 'sim' + '-' + 'real' + '-' + str(examples_per_class) + '.png')
    else:
        plot_embedding(dann_tsne, combined_test_labels.numpy().argmax(1), combined_test_domain.numpy().reshape(-1), '')

    if save_results:
        # Write the accuracy metrics to the training config file
        config_output_file = os.getcwd() + '/results/final_training_config_dirl' + '-' + 'sim' + '-' + \
                             'real' + '-' + str(examples_per_class) + '.yml'

        outfile = open(config_output_file, "a")  # append mode

        outfile.write('\n \n Max Accuracy on target test: \t' + str(max_test_run) + '\n')
        # outfile.write('Final Accuracy on source, target test: \t' + str(class_accur_source) + ',\t' + str(class_accur_target) + '\n')
        outfile.write('Domain Accuracy on both, source, target test: \t' + str(domain_accur_both) + ',\t' + str(domain_accur_source) + ',\t' + str(domain_accur_target) + '\n')
        outfile.write('Classicication Accuracy on source, target test: \t' + str(class_accur_source) + ',\t' + str(class_accur_target) + '\n')
        outfile.write('KNN accuracy on target test: \t' + str(knn_accuracy) + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-examples_per_class', type=int, default=10)
    parser.add_argument('-config_path', type=str, default='../configs/sim2real_dirl.yml')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save_results', type=bool, default=True)
    parser.add_argument('-augment_real', type=int, default=0)

    kwargs = vars(parser.parse_args())
    main(**kwargs)