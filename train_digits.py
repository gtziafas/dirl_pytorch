from typing import *
import os
import yaml
import random
import numpy as np  
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 
from sklearn.neighbors import KDTree
import matplotlib
matplotlib.use('TkAgg')  # Need Tk for interactive plots.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from digits_dataset_loader import load_dataset, few_labels
from digits_model import DigitsDIRL
from dirl_losses import *

# reproducability
SEED = torch.manual_seed(1312)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(1312)
np.random.seed(1312)
random.seed(1312)


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


def test(model: nn.Module, test_data: List[Tensor]) -> Tuple[float, float]:
    combined_test_imgs, combined_test_labels = test_data
    model.eval()
    _size = combined_test_imgs.shape[0]
    out = model.forward(combined_test_imgs)
    interim = (combined_test_labels.argmax(1) == out[2].argmax(1)).float()
    source_accuracy_fin = interim[:_size // 2].mean(0)
    target_accuracy_fin = interim[_size // 2:].mean(0)
    return source_accuracy_fin, target_accuracy_fin


def main(mode: str, 
         source: str, 
         target: str, 
         examples_per_class: int,
         config_path: str,
         save_results: bool,
         device: str
        ):
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    # create directory to save results
    logdir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # load and partition data
    source_data, source_data_test, source_labels, source_labels_test = load_dataset(source)
    target_data, target_data_test, target_labels, target_labels_test = load_dataset(target)

    num_classes = cfg['dataset']['num_classes']
    target_sup_size = examples_per_class * num_classes
    sizing = [target_sup_size * s for s in cfg['dataset']['sizing']]
    sup_id = sum(sizing[0:2])
    batch_size = sum(sizing)

    x_target_sup, y_target_sup = few_labels(target_data, target_labels, examples_per_class, num_classes)

    # batch generation and testing split
    dl_source = DataLoader(list(zip(source_data, source_labels)), batch_size=sizing[0], shuffle=True, drop_last=True)
    dl_target = DataLoader(list(zip(target_data, target_labels)), batch_size=sizing[2], shuffle=True, drop_last=True)

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
    model = DigitsDIRL(num_classes=num_classes,
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
    opt_A =  Adam([ 
                 {'params': model.encoder.parameters(), 'weight_decay': cfg['model']['weight_decay']}, 
                 {'params': model.cls.parameters()}, 
                 {'params': model.dd.parameters()}, 
                 {'params': model.cdd.parameters()},
             ], lr=cfg['model']['learning_rate'])  
    opt_B = Adam(model.encoder.parameters(), lr=cfg['model']['learning_rate'], weight_decay=cfg['model']['weight_decay'])

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
        x_batch = torch.cat((x_batch_source, x_target_sup, x_batch_target), dim=0).to(device)
        y_batch = torch.cat((y_batch_source, y_target_sup, y_batch_target), dim=0).to(device)

        model.train()

        triplet_loss = None
        if iteration > 300:
            #opt_B.zero_grad()
            embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch)

            triplet_loss = triplet_crit(embs[:sup_id], y_batch[:sup_id])
            _, adv_domain_loss = domain_crit(domain_preds, domain_batch)
            _, cdann_loss_adv = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])
            
            loss_B = cfg['loss']['class_dann_weight'] * cdann_loss_adv + cfg['loss']['domain_weight'] * adv_domain_loss + triplet_loss

            # backprop adversarial step
            opt_B.zero_grad()
            loss_B.backward()
            opt_B.step()

            triplet_loss = round(triplet_loss.item(), 4)

        #opt_A.zero_grad()
        embs, domain_preds, class_preds, cdann_preds = model.forward(x_batch, freeze_gradient=True)

        classify_loss = classify_crit(class_preds[:sup_id], y_batch[:sup_id])
        domain_loss, _ = domain_crit(domain_preds, domain_batch)
        cdann_loss, _ = cdann_crit([t[:sup_id] for t in cdann_preds], y_batch[:sup_id], domain_batch[:sup_id], target_start_id=sizing[0])

        loss_A = cfg['loss']['classify_weight'] * classify_loss + cfg['loss']['class_dann_weight'] * cdann_loss + cfg['loss']['domain_weight'] * domain_loss
        
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
        classify_losses.append(classify_loss.item())
        cdann_losses.append(cdann_loss.item())
        #cdann_losses.append(None)
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
        # plot loss functions
        plt.figure(figsize=(10, 5))
        plt.plot(range(cfg['model']['num_iterations']), plot_losses)
        plt.legend(['domain_loss', 'classify_loss', 'class_dann_loss', 'triplet_loss'])
        plt.savefig(os.getcwd() + '/results/dirl_digits_losses_dirl' + '-' + source + '-'+ target + '-' + str(examples_per_class) + '.png', format='png', bbox_inches='tight',
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
            save_fig_path=os.getcwd() + '/results/dirl_digits_tsne_plot_dirl' + '-' + source + '-' + target + '-' + str(examples_per_class) + '.png')
    else:
        plot_embedding(dann_tsne, combined_test_labels.numpy().argmax(1), combined_test_domain.numpy().reshape(-1), '')

    if save_results:
        # Write the accuracy metrics to the training config file
        config_output_file = os.getcwd() + '/results/final_training_config_dirl' + '-' + source + '-' + \
                             target + '-' + str(examples_per_class) + '.yml'

        outfile = open(config_output_file, "a")  # append mode

        outfile.write('\n \n Max Accuracy on target test: \t' + str(max_test_run) + '\n')
        # outfile.write('Final Accuracy on source, target test: \t' + str(class_accur_source) + ',\t' + str(class_accur_target) + '\n')
        outfile.write('Domain Accuracy on both, source, target test: \t' + str(domain_accur_both) + ',\t' + str(domain_accur_source) + ',\t' + str(domain_accur_target) + '\n')
        outfile.write('Classicication Accuracy on source, target test: \t' + str(class_accur_source) + ',\t' + str(class_accur_target) + '\n')
        outfile.write('KNN accuracy on target test: \t' + str(knn_accuracy) + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="dirl", choices=["source_only", "triplet", "dann", "dirl"])
    parser.add_argument('-source', type=str, default='mnist', choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-target', type=str, default='mnistm', choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-examples_per_class', type=int, default=10)
    parser.add_argument('-config_path', type=str, default='./configs/digits_dirl.yml')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save_results', type=bool, default=True)

    kwargs = vars(parser.parse_args())
    main(**kwargs)