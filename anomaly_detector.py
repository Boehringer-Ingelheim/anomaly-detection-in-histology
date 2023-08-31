import numpy as np
import random
from sklearn import svm # must stay here (eval)
import matplotlib
import matplotlib.pyplot as plt
from utils.training_utils import apply_net
import torch
from utils.data_processing import CodesProcessor, set_seed
from torchvision import transforms
from utils.imports import show_configuration, save_configuration, update_configuration
from utils.image_dataset_reader import HistImagesDataset, samples_per_location_from_samples_per_class
from torch.utils.data import DataLoader
import logging
import pickle
from sklearn import manifold
from matplotlib import cm
import pandas as pd
from matplotlib.font_manager import FontProperties
import time
import os
import argparse
from time import perf_counter
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, roc_curve
from shutil import copyfile
import models.pretrained_networks as HistoModel
from utils.timed_input import limited_time_input
matplotlib.use('Qt5Agg')


def anomaly_training(model, clf, data_loader, dev, cache_path=None):

    logging.info('computing codes for training anomaly classifier')
    code_processor = CodesProcessor(cached_codes_path=cache_path)
    apply_net(model, dev, data_loader, verbose=True, code_processor=code_processor)

    logging.info('all codes for training were computed')

    features = code_processor.get_codes()
    logging.info('codes are generated, starting training anomaly classifier')
    clf.fit(features)

    return clf, features


def anomaly_detection(clf, model, data_loader, dev, feat_container=None, cache_path=None):

    code_processor = CodesProcessor(cached_codes_path=cache_path)
    apply_net(model, dev, data_loader, verbose=True, code_processor=code_processor)
    logging.info('all codes were computed')

    # classify feature representations
    features = code_processor.get_codes()
    image_names = code_processor.get_image_names()
    image_labels = code_processor.get_image_labels()

    assert len(features) == len(image_names)

    scores = clf.decision_function(features)

    if feat_container is not None:

        feat_container['labels'].extend(image_labels)

        if len(feat_container['features']) == 0:
            feat_container['features'] = np.zeros((0, features.shape[1]))
        feat_container['features'] = np.concatenate((feat_container['features'], features), axis=0)

    return scores, image_names, image_labels


def save_images_from_excel(path2csv, path2save, n_examples, ex_type='FN'):

    """
    Copies n_examples images with the highest/lowest scores (most normals/anomalous, corresponding to most false negative/positive examples - normal/abnormal detections from anomaly/normal data)
    to newly created subfolder 'ex_type' in path2save, for ex_type='FN'/'FP'. All the data is read from csv file path2csv
    :param path2csv:
    :param path2save:
    :param n_examples: maximal number of extreme examples to be copied (lower number might be copied if thee are less of normal/anomalous images)
    :param ex_type: must be 'FN' or 'FP' - false negatives for anomalous input data or false positives for normal input data
    :return:
    """

    # read csv
    df = pd.read_csv(path2csv)
    image_paths = df['image_path'].to_list()
    image_names = df['image name'].to_list()
    anomaly_scores = df['anomaly scores'].to_list()
    labels = df['original label'].to_list()

    assert len(image_paths) == len(image_names)

    if ex_type == 'FN':
        idx_sorted = np.argsort(-np.array(anomaly_scores))
    elif ex_type == 'FP':
        idx_sorted = np.argsort(np.array(anomaly_scores))
    else:
        raise RuntimeError("argument ex_type must be 'FN' or 'FP'")

    image_path_new = os.path.join(path2save, ex_type)
    if not os.path.exists(image_path_new):
        os.makedirs(image_path_new)

    for n, idx in enumerate(idx_sorted):

        image_path = image_paths[idx]
        image_name = image_names[idx]
        anomaly_score = anomaly_scores[idx]
        label = labels[idx]

        if n == n_examples:
            break

        if ex_type == 'FN':
            if anomaly_score < 0:
                break
        elif ex_type == 'FP':
            if anomaly_score > 0:
                break
        else:
            raise RuntimeError("argument ex_type must be 'FN' or 'FP'")

        image_name_new, ext = os.path.splitext(image_name)
        image_name_new = image_name_new + '_' + label + '_score:' + str(anomaly_score).replace(' ', '_') + ext

        full_path_new = os.path.join(image_path_new, image_name_new)

        full_path_old = os.path.join(image_path, image_name)

        copyfile(full_path_old, full_path_new)


def TSNE_visualization(features, labels, colormap='tab10', n_features=None, scores=None, save_path=None):
    """

    :param features:
    :param labels: data original labels
    :param colormap:
    :param n_features: maximal allowed number of feature vectors to be shown
    :param scores: anomaly scores for the data. To be used only for the second plot to see detection boundary
    :param save_path: path to save TSNE images
    :return:
    """

    assert len(features) == len(labels)
    if scores is not None:
        scores = np.array(scores)
        assert len(labels) == len(scores)

    perplexity = 30
    colormap = cm.get_cmap(colormap)

    me = manifold.TSNE(n_components=2, perplexity=perplexity, verbose=1, method='barnes_hut')
    new_features = me.fit_transform(features)

    # --------plot data TSNE clusters--------
    # plot new embeddings
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    set_labels = set(labels)

    # make the order in which 'normal' or 'NAS_anomaly' labels will be in the beginning of the set_labels list
    n_lab = []
    a_lab = []
    for lab in set_labels:
        if ('normal' in lab) or ('NAS anomaly' in lab):
            n_lab.append(lab)
        else:
            a_lab.append(lab)
    set_labels = n_lab + a_lab

    n_labels = len(set_labels)
    for n, label in enumerate(set_labels):

        if label == 'normal' or label == 'NAS anomaly':
            marker_size = 5
            marker_shape = 'o'
        else:
            marker_size = 5
            marker_shape = 'o'

        # select indexes of current class only
        ind = [labels[i] == label for i in range(len(labels))]

        class_features = new_features[ind, :]
        if scores is not None:
            class_scores = scores[ind]

        # randomly select only part of features
        if n_features:
            ind = np.random.permutation(class_features.shape[0])[:n_features]
            class_features = class_features[ind, :]
            if scores is not None:
                class_scores = class_scores[ind]

        ax.scatter(class_features[:, 0], class_features[:, 1], label=label, marker=marker_shape, s=marker_size, c=np.reshape(colormap(n/(n_labels-1)), (1, 4)), alpha=0.5)

        if scores is not None:
            ind_anomaly = class_scores > 0

            h1 = ax2.scatter(class_features[ind_anomaly, 0], class_features[ind_anomaly, 1], marker='o', label='normal', s=marker_size, c='b', alpha=0.5)
            h2 = ax2.scatter(class_features[~ind_anomaly, 0], class_features[~ind_anomaly, 1], marker='o', label='anomaly', s=marker_size, c='r', alpha=0.5)

    leg = ax.legend(bbox_to_anchor=(0.98, 1), prop=fontP, markerscale=2.5)
    # makes opacity = 100% instead of taking opacity from the points drawn
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    if scores is not None:
        leg2 = ax2.legend(bbox_to_anchor=(0.98, 1), prop=fontP, handles=[h1, h2], markerscale=2.5)
        # makes opacity = 100% instead of taking opacity from the points drawn
        for lh in leg2.legendHandles:
            lh.set_alpha(1)

    if save_path:
        fig.savefig(os.path.join(save_path, 'feature_vectors_TSNE.png'), bbox_inches='tight')
        plt.close()
        fig2.savefig(os.path.join(save_path, 'detections_TSNE.png'), bbox_inches='tight')
        plt.close()


def save_to_excel(scores, image_names, image_labels, csv_path):

    paths = []
    file_names = []
    for i in range(len(image_names)):
        path, file_name = os.path.split(image_names[i])
        paths.append(path)
        file_names.append(file_name)

    data = pd.DataFrame(scores, columns=['anomaly scores'])
    data['original label'] = image_labels
    data['image_path'] = paths
    data['image name'] = file_names

    data.to_csv(csv_path, index=False)

    return


def create_data_loader(paths, cfg, n_patches=None, batch_size=1, augmentation=False):

    if not paths:
        return None

    if n_patches and isinstance(n_patches, int):
        n_patches = samples_per_location_from_samples_per_class(*paths, samples_per_class=n_patches)

    tr_normalize = transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)
    if augmentation:
        hs_jitter = transforms.ColorJitter(saturation=cfg.aug_saturation, hue=cfg.aug_hue)
        bc_jitter = transforms.ColorJitter(brightness=cfg.aug_brightness, contrast=cfg.aug_contrast)

    if not augmentation:
        seq = [transforms.CenterCrop(cfg.patch_size), transforms.ToTensor(), tr_normalize]
    else:
        seq = [transforms.CenterCrop(cfg.patch_size), hs_jitter, bc_jitter, transforms.ToTensor(), tr_normalize]

    transforms_seq = transforms.Compose(seq)

    dataset = HistImagesDataset(*paths, transform=transforms_seq, n_samples=n_patches)

    data_loader = DataLoader(dataset, num_workers=0, batch_size=batch_size)

    return data_loader


def sample_features(embeddings, n_samples, anomaly_scores):

    features_to_visualize = {'features': np.zeros((0, 0)), 'labels': []}
    scores = []

    labs = set(embeddings['labels'])
    lab_intersection = labs.intersection(n_samples.keys())
    assert len(lab_intersection) >= 2

    if len(labs.intersection(n_samples.keys())) != len(labs):
        logging.info('labels from datasets to be visualized in TSNE and requested labels for visualization do not completely coinside')

    for lab in labs:
        if lab in n_samples.keys():
            idx = [i for i in range(len(embeddings['labels'])) if embeddings['labels'][i] == lab]
            if n_samples[lab] <= len(idx):
                idx = random.sample(idx, n_samples[lab])
            else:
                logging.warning('number of requested features {} is larger than was calculated {}, using all calculated features'.format(n_samples[lab], len(idx)))

            sampled_labels = [embeddings['labels'][i] for i in idx]
            sampled_scores = [anomaly_scores[i] for i in idx]
            sampled_features = embeddings['features'][idx, :]

            scores.extend(sampled_scores)
            features_to_visualize['labels'].extend(sampled_labels)
            if len(features_to_visualize['features']) == 0:
                features_to_visualize['features'] = np.zeros((0, sampled_features.shape[1]))
            features_to_visualize['features'] = np.concatenate((features_to_visualize['features'], sampled_features), axis=0)

    return features_to_visualize, scores

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO)
t_start = perf_counter()
string_time = time.strftime("%H%M%S_%d%m%y")
parser = argparse.ArgumentParser(description='anomaly detector')

cfg = update_configuration(parser)

if not cfg.description:
    cfg.description = limited_time_input("Please enter description of an experiment...", 30)

print('\n')
show_configuration(cfg)
print('\n')

set_seed(cfg.seed_number)

fontP = FontProperties()
fontP.set_size('xx-small')

if not os.path.isdir(cfg.output_path):
    os.makedirs(cfg.output_path)

if os.listdir(cfg.output_path):
    raise RuntimeError(f"output folder {cfg.output_path} is not empty")
#assert not os.listdir(cfg.output_path), f"output folder {cfg.output_path} is not empty"

#stream_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler(os.path.join(cfg.output_path, 'anomaly_detector.log'))
logging.root.addHandler(file_handler)

logging.info("staining: {}".format(cfg.data_staining))
logging.info("number of classes: {}".format(cfg.n_trained_classes))
logging.info('output folder: {}'.format(cfg.output_path))

dev = torch.device(cfg.dev)

NetworkModel = getattr(HistoModel, cfg.model_architecture)
model = NetworkModel(path_trained_model=cfg.cnn_model, n_classes=cfg.n_trained_classes, dev=dev)
model.to(dev)

if cfg.cnn_model:  # path to model was defined
    CNN_model_name, _ = os.path.splitext(cfg.cnn_model)
    CNN_model_path, CNN_model_name = os.path.split(CNN_model_name)
    logging.info('CNN model {} will be used'.format(cfg.cnn_model))

    #output_path = os.path.join(CNN_model_path, cfg.anomaly_model_folder + "_" + CNN_model_name)

    #anomaly_model_path = os.path.join(output_path, CNN_model_name + '.pkl')
else:  # path for model, use pre-trained standard model
    CNN_model_name = cfg.model_architecture + '_backbone'
    CNN_model_path = cfg.output_path
    #CNN_model_path = cfg.code_output_path + string_time

    CNN_model_path_full = os.path.join(CNN_model_path, CNN_model_name + '.pt')
    torch.save(model.state_dict(), CNN_model_path_full)
    logging.info('backbone model was saved to {}'.format(CNN_model_path_full))

    #output_path = os.path.join(CNN_model_path, cfg.anomaly_model_folder + "_" + CNN_model_name)
    #anomaly_model_path = os.path.join(output_path, CNN_model_name + '.pkl')


print("")
if cfg.ad_model:  # path to anomaly detection model was defined

    AD_model_name, _ = os.path.splitext(cfg.ad_model)
    AD_model_path, AD_model_name = os.path.split(AD_model_name)

    assert CNN_model_name == AD_model_name[:len(CNN_model_name)], "CNN and AD names must be the same except extention and possibly suffix"

    try:
        clf = pickle.load(open(cfg.ad_model, 'rb'))
    except FileNotFoundError:
        clf = None

else:
    clf = None

# try:
#     clf = pickle.load(open(anomaly_model_path, 'rb'))
# except:
#     new_model = True
#
#     try:
#         os.makedirs(output_path)
#     except FileExistsError:
#         logging.info('{} already exists. Probably training has already been started but then was canceled'.format(output_path))
# else:
#     ans = limited_time_input("Do you want to use already trained one class classifier model [y/n]? ", 30)
#     if ans in ('y', 'Y', 'yes', 'YES', 'Yes'):
#         new_model = False
#
#         logging.info("previousely trained one-class classifier will used")
#
#         # create a new subfolder
#         path_folder, path_subfolder = os.path.split(output_path)
#         path_subfolder = path_subfolder + '_' + string_time
#         output_path = os.path.join(path_folder, path_subfolder)
#     else:
#         new_model = True
#
#         logging.info("new one class classifier will be trained (the previous one will NOT be erased)")
#         # create a new subfolder
#         path_folder, path_subfolder = os.path.split(output_path)
#         path_subfolder = path_subfolder + '_retrained_' + string_time
#         output_path = os.path.join(path_folder, path_subfolder)
#         anomaly_model_path = os.path.join(output_path, CNN_model_name + '.pkl')
#
#     try:
#         os.makedirs(output_path)
#     except FileExistsError:
#         raise RuntimeError('{} already exists'.format(output_path))


# if not cfg.cnn_model: # save backbone model
#     saved_model_path_full = os.path.join(CNN_model_path, CNN_model_name + '.pt')
#     torch.save(model.state_dict(), saved_model_path_full)
#     logging.info('backbone model was saved to {}'.format(saved_model_path_full))

# file_handler = logging.FileHandler(os.path.join(output_path, 'anomaly_detector.log'))
# logging.root.addHandler(file_handler)
#
# logging.info("staining: {}".format(cfg.data_staining))
# logging.info("number of classes: {}".format(cfg.n_trained_classes))

if not clf:

    logging.info('a new model for anomaly detector will be trained')

    AD_model_name = CNN_model_name
    AD_model_path = CNN_model_path
    AD_model_path_full = os.path.join(AD_model_path, AD_model_name + '.pkl')

    # prevent overwriting if the model with same name was already existing
    if os.path.isfile(AD_model_path_full):
        AD_model_name = CNN_model_name + '_' + string_time
        AD_model_path_full = os.path.join(CNN_model_path, AD_model_name + '.pkl')

    logging.info("--------------training---------------------")
    training_data_loader = create_data_loader(cfg.paths_normal, cfg, n_patches=cfg.train_patches_for_train_max, batch_size=cfg.batch_size, augmentation=cfg.augmentation)
    classifier = eval(cfg.clf)
    clf, _ = anomaly_training(model, classifier, training_data_loader, dev, cache_path=cfg.output_path)
    logging.info('-----------training has been finished--------------')

    pickle.dump(clf, open(AD_model_path_full, 'wb'))
    logging.info('trained model was saved to {}'.format(AD_model_path_full))

    t_end_training = perf_counter()
    logging.info('training on normal data took {} sec'.format(t_end_training - t_start))

    #save_configuration(cfg, output_path + '/anomaly_detector_configuration.txt')

    #training_data_loader = create_data_loader(cfg.paths_normal, cfg, n_patches=cfg.train_patches_for_test_max, brightness_factor=cfg.brightness_factor, batch_size=cfg.batch_size)
else:
    logging.info('anomaly detector model {} was found and will be used'.format(cfg.ad_model))

# if new_model:
#
#     logging.info('anomaly detector model {} was not found'.format(anomaly_model_path))
#     logging.info('a new model for anomaly detector will be trained')
#
#     logging.info("--------------training---------------------")
#     training_data_loader = create_data_loader(cfg.paths_normal, cfg, n_patches=cfg.train_patches_for_train_max, batch_size=cfg.batch_size, augmentation=cfg.augmentation)
#     classifier = eval(cfg.clf)
#     clf, _ = anomaly_training(model, classifier, training_data_loader, dev, cache_path=output_path)
#     logging.info('-----------training has been finished--------------')
#
#     pickle.dump(clf, open(anomaly_model_path, 'wb'))
#     logging.info('trained model was saved to {}'.format(anomaly_model_path))
#
# else:
#     logging.info('already saved anomaly detector model {} was found and will be used'.format(anomaly_model_path))

save_configuration(cfg, os.path.join(cfg.output_path, 'anomaly_detector_configuration.txt'))
#logging.info('output folder: {}'.format(output_path))


features_to_visualize = {'features': np.zeros((0, 0)), 'labels': []}
scores = []
scores_normal_test, normal_test_im_names, normal_test_labels = None, None, None
scores_liver_anomaly_test, liver_anomaly_test_im_names, liver_anomaly_test_labels = None, None, None
scores_non_liver_test, non_liver_test_im_names, non_liver_test_labels = None, None, None

print("")
normal_test_data_loader = create_data_loader(cfg.paths_normal_test, cfg, n_patches=cfg.test_normal_patches_max, batch_size=cfg.batch_size)

if normal_test_data_loader:
    logging.info('-----------anomaly detection in normal test data----------------')
    scores_normal_test, normal_test_im_names, normal_test_labels = anomaly_detection(clf, model, normal_test_data_loader, dev, feat_container=features_to_visualize, cache_path=cfg.output_path)
    if scores_normal_test is not None:
        scores.extend(scores_normal_test)
        save_to_excel(scores_normal_test, normal_test_im_names, normal_test_labels, os.path.join(cfg.output_path, cfg.csv_liver_tissue_testnormals))
        save_images_from_excel(os.path.join(cfg.output_path, cfg.csv_liver_tissue_testnormals), cfg.output_path, cfg.save_n_FP, 'FP')

print("")
liver_anomaly_test_data_loader = create_data_loader(cfg.paths_liver_anomaly_test, cfg, n_patches=cfg.test_anomaly_patches_per_class_max, batch_size=cfg.batch_size)

if liver_anomaly_test_data_loader:
    logging.info('-----------anomaly detection in liver with conditions----------------')
    scores_liver_anomaly_test, liver_anomaly_test_im_names, liver_anomaly_test_labels = anomaly_detection(clf, model, liver_anomaly_test_data_loader, dev, feat_container=features_to_visualize, cache_path=cfg.output_path)
    if scores_liver_anomaly_test is not None:
        scores.extend(scores_liver_anomaly_test)
        save_to_excel(scores_liver_anomaly_test, liver_anomaly_test_im_names, liver_anomaly_test_labels, os.path.join(cfg.output_path, cfg.csv_liver_tissue_anomalies))
        save_images_from_excel(os.path.join(cfg.output_path, cfg.csv_liver_tissue_anomalies), cfg.output_path, cfg.save_n_FN, 'FN')


print("")
non_liver_test_data_loader = create_data_loader(cfg.paths_non_liver_tissues_test, cfg, n_patches=cfg.visual_test_auxiliary_patches_per_class, batch_size=cfg.batch_size)

if non_liver_test_data_loader:
    logging.info('-----------anomaly detection in different to liver tissues---------------')
    scores_non_liver_test, non_liver_test_im_names, non_liver_test_labels = anomaly_detection(clf, model, non_liver_test_data_loader, dev, feat_container=features_to_visualize, cache_path=cfg.output_path)
    if scores_non_liver_test is not None:
        scores.extend(scores_non_liver_test)


# t_end = perf_counter()
# logging.info('training on normal data and inference took {} sec'.format(t_end - t_start))

# print to file
if (scores_normal_test is not None) and (scores_liver_anomaly_test is not None):
    with open(os.path.join(cfg.output_path, 'anomaly_detector_results.txt'), 'w') as fh:

        logging.info('----------performance summary----------')
        print('----------performance summary-----------', file=fh)

        all_targets = np.concatenate((np.ones(len(scores_liver_anomaly_test)), np.zeros(len(scores_normal_test))))
        all_scores = - np.concatenate((scores_liver_anomaly_test, scores_normal_test))

        # balanced accuracy
        ba = balanced_accuracy_score(all_targets, all_scores > 0)
        logging.info('Balanced accuracy: {}'.format(ba))
        print('Balanced accuracy: {}'.format(ba), file=fh)

        # calculate AUC
        auc = roc_auc_score(all_targets, all_scores)
        logging.info('AUC: {}'.format(auc))
        print('AUC: {}'.format(auc), file=fh)

        # ROC curve
        fpr, tpr, thr_roc = roc_curve(all_targets, all_scores)
        ind_thr0 = np.argmin(np.abs(thr_roc))
        fpr_thr0 = fpr[ind_thr0]
        tpr_thr0 = tpr[ind_thr0]

        fig, ax = plt.subplots()
        plt.plot(fpr, tpr)
        plt.plot(fpr_thr0, tpr_thr0, '*')
        plt.xlabel("False Positives Rate (1-specificity)")
        plt.ylabel("True Positives Rate (sensitivity)")
        fig.savefig(os.path.join(cfg.output_path, 'ROC.png'), bbox_inches='tight')
        plt.close()

        logging.info('predicted anomalies in normal data: {:.2f}%'.format(fpr_thr0 * 100))
        print('predicted anomalies in normal data: {:.2f}%'.format(fpr_thr0 * 100), file=fh)
        logging.info('predicted normals in anomaly data: {:.2f}%'.format((1 - tpr_thr0) * 100))
        print('predicted normals in anomaly data: {:.2f}%'.format((1 - tpr_thr0) * 100), file=fh)

        # f1 score
        f1 = f1_score(all_targets, all_scores > 0)
        logging.info('F1 score: {}'.format(f1))
        print('F1 score: {}'.format(f1), file=fh)

if scores:
    features_to_visualize_sampled, scores_sampled = sample_features(features_to_visualize, cfg.n_features_visualization, scores)
    TSNE_visualization(features_to_visualize_sampled['features'], features_to_visualize_sampled['labels'], scores=scores_sampled, save_path=cfg.output_path, colormap='tab20')

t_end = perf_counter()
logging.info('it took {} sec'.format(t_end - t_start))

