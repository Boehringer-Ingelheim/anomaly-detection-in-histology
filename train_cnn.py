#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:42:19 2019

@author: zingman
"""

# -----------------------------------------------------------------------------
# classifies histological images into two categories healthy/non-healthy using
# tiled patch images saved on the hard drive
# -----------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from utils.image_dataset_reader import HistImagesDataset, samples_per_location_from_samples_per_class
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil

from time import perf_counter
import os
import datetime
import copy
import time
from utils.data_processing import set_seed

import models.pretrained_networks as HistoModel
from utils.timed_input import limited_time_input
import logging
from models.losses import CenterLoss
import argparse

from utils.imports import show_configuration, save_configuration, \
    pickle_configuraton_as_dictionary, update_configuration

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#plt.ion()

def evaluate_model(val_loader, loss_fun, n_samples_val, model_skel):

    str_labels = val_loader.dataset.str_labels
    # string labels are encoded by sequential integer labels beginning from 0
    integer_labels = list(range(len(str_labels)))

    final_val_loss, final_val_acc, conf_mat = validation(model_skel, iter(val_loader), loss_fun, n_samples=n_samples_val, confusion=True, integer_labels=integer_labels)
    print('validation accuracy of the model: {:.4f}, validation loss of the model: {:.4f}'.format(final_val_acc,
                                                                                                  final_val_loss))
    # visualize confusion matrix
    df_cf = pd.DataFrame(conf_mat, index=str_labels, columns=str_labels)
    plt.figure(figsize=(15, 10))
    sn.heatmap(df_cf, annot=True)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('accuracy: {}'.format(final_val_acc))

def validation(model, data_iter, loss_fun, n_samples=None, centerloss_fun=None, ce_weight=1.0, cl_weight=0.0, confusion=False, integer_labels=None):
    """
    :param model: model to be validated
    :param data_iter: data iterator (use iter(data_set) or iter(data_loader)). The iterator can be infinite.
    If iterator is finite and it is exhausted before n_samples were generated, warning is printed out
    :param n_samples: n_samples to be used for validating the model
    :param confusion: calculate also confusion matrix
    :return: accuracy, loss values, and confusion matrix if requested
    """

    if confusion and integer_labels is None:
        print("Error: when confusion matrix need to be calculated, integer labels must be provided")
        raise

    if not n_samples:
        n_samples = np.inf

    model.eval()

    val_loss = torch.tensor(0.0, device=device)
    n_pred = torch.tensor(0.0, device=device)
    n_processed = 0.0
    conf_mat = None
    with torch.no_grad():
        counter = 0
        # for samples in data_loader:
        while n_processed < n_samples:

            try:
                samples = next(data_iter)
            except StopIteration:
                logging.warning(
                    'finite iterator with {} images was provided, it was exhausted before required {} were generated'.format(n_processed, n_samples))
                break

            images = samples['image'].to(device)
            int_labels = samples['label']
            n_processed += len(int_labels)
            int_labels = int_labels.to(device)

            outputs = model(images)
            pedictions = outputs['categories']
            entropy_loss = loss_fun(pedictions, int_labels)

            if (cl_weight != 0) and (centerloss_fun is not None):
                features = outputs['pooled_codes']
                center_loss = centerloss_fun(features, int_labels)
                loss = ce_weight * entropy_loss + cl_weight * center_loss
            else:
                loss = entropy_loss

            predicted_values = torch.max(pedictions, 1)[1]
            n_pred += torch.sum(predicted_values == int_labels)

            val_loss += loss

            if confusion:

                if counter > 0:
                    conf_mat += confusion_matrix(int_labels.cpu().numpy(), predicted_values.cpu().numpy(), labels=integer_labels)
                else:
                    conf_mat = confusion_matrix(int_labels.cpu().numpy(), predicted_values.cpu().numpy(), labels=integer_labels)

            counter += 1

    loss_value = val_loss.item() / counter
    accuracy = n_pred.item() / n_processed
    return loss_value, accuracy, conf_mat


def train_epoch(model, data_loader, optimizer, loss_fun, tb_writer, iter_step_show=10, centerloss=None, ce_weight=1.0, cl_weight=0.0):

    try:
        len_dataset = len(data_loader)
    except TypeError:
        len_dataset = float("inf")

    model.train()

    train_loss = 0.0
    counter = 0
    progress = tqdm(data_loader, desc="Batch loss: ", total=len_dataset, disable=False)
    for samples in progress:

        try:
            train_epoch.iteration += 1
        except AttributeError:
            train_epoch.iteration = 0

        images = samples['image'].to(device)

        int_labels = samples['label']
        int_labels = int_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        predictions = outputs['categories']
        entropy_loss_criterion = loss_fun(predictions, int_labels)

        if (cl_weight != 0) and (centerloss is not None):
            features = outputs['pooled_codes']
            center_loss_criterion = centerloss(features, int_labels)
            loss = ce_weight * entropy_loss_criterion + cl_weight * center_loss_criterion
        else:
            loss = entropy_loss_criterion

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if counter % iter_step_show == 0:
            progress.set_description("Batch loss: {:.4f}".format(loss.item()))
            tb_writer.add_scalar('Batch_loss', loss.item(), train_epoch.iteration)

        counter += 1

    epoch_loss = train_loss / counter

    return epoch_loss


#-----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------main code---------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

assert torch.cuda.is_available(), "GPU is not available"

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='training cnn')
cfg = update_configuration(parser)

if not cfg.description:
    cfg.description = limited_time_input("Please enter description of an experiment...", 60)

print('\n')
show_configuration(cfg)
print('\n')

device = torch.device(cfg.device_name)

set_seed(cfg.seed_number)

t_start = perf_counter()

# defining paths and creating output directories
output_path = cfg.path_to_results + '/' + cfg.string_time
output_path_tb = cfg.path_to_results + '_tb/' + cfg.string_time
os.makedirs(output_path_tb)
os.makedirs(output_path)
tb_writer = SummaryWriter(output_path_tb)

file_handler = logging.FileHandler(os.path.join(output_path, 'training_cnn.log'))
logging.root.addHandler(file_handler)

tr_resize = transforms.Resize(cfg.patch_size)

tr_normalize = transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)
bc_jitter = transforms.ColorJitter(brightness=cfg.aug_brightness, contrast=cfg.aug_contrast)

transforms_seq_train = transforms.Compose([transforms.CenterCrop(cfg.patch_size), bc_jitter, transforms.ToTensor(), tr_normalize])
transforms_seq_val = transforms.Compose([transforms.CenterCrop(cfg.patch_size), transforms.ToTensor(), tr_normalize])

n_samples_train_per_location = samples_per_location_from_samples_per_class(*cfg.path_to_tissues, samples_per_class=cfg.n_samples_train_per_class)
images_dataset = HistImagesDataset(*cfg.path_to_tissues, n_samples=n_samples_train_per_location, transform=transforms_seq_train, repetition=True)

n_classes = len(images_dataset.str_labels)
assert cfg.number_of_classes == n_classes

images_validation, images_train = images_dataset.split_set(cfg.n_samples_val, transform_validation=transforms_seq_val)

if getattr(cfg, 'mixup_classes', False):
    images_train.prepare_mixup(cfg.mixup_classes)

train_loader = DataLoader(images_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, pin_memory=False)
val_loader = DataLoader(images_validation, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=False)

NetworkModel = getattr(HistoModel, cfg.model_name)
model = NetworkModel(n_classes=n_classes, dev=device).to(device)

try:
    logging.info("training dataset consist of {} images".format(len(images_train)))
    logging.info("validation dataset consist of {} images".format(len(images_validation)))
except TypeError:
    logging.info('iterable dataset is used, the size cannot be determined a priori')


# ---------------------------
# -----------training ------
# ----------------------------

if cfg.centerloss_classes is not None:
    if cfg.centerloss_classes == 'all':
        chosen_class_int = None
    elif isinstance(cfg.centerloss_classes, (list, tuple)):
        chosen_class_int = []
        for lb in cfg.centerloss_classes:
            chosen_class_int.append(images_train.get_int_label(lb))
    elif isinstance(cfg.centerloss_classes, str):
        chosen_class_int = images_train.get_int_label(cfg.centerloss_classes)
    else:
        assert False, 'centerloss_classes in configuration file is not valid'

    metric_loss = CenterLoss(num_classes=n_classes, feat_dim=model.fv_length(), device=device, constrained_classes=chosen_class_int, mu=0.5)
    logging.info('chosen classes in centerloss: {} ({})'.format(cfg.centerloss_classes, chosen_class_int))
else:
    metric_loss = None

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg.model_lr, momentum=cfg.ce_momentum)

best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

n_trained_param = model.count_parameters()

logging.info("number of parameters to be trained is {}".format(n_trained_param))

t_start_epoch = perf_counter()
for epoch in range(cfg.num_epochs):

    logging.info('Epoch {}/{}, training ...'.format(epoch, cfg.num_epochs - 1))
    epoch_training_av_loss = train_epoch(model, train_loader, optimizer, loss_fun, tb_writer, cfg.train_step_show, centerloss=metric_loss, ce_weight=cfg.ce_weight, cl_weight=cfg.cl_weight)
    with torch.cuda.device(cfg.device_name): # by default cuda:0 is used
        torch.cuda.empty_cache()

    logging.info('validating on a separate validation dataset...')
    epoch_val_loss, epoch_val_acc, _ = validation(model, iter(val_loader), loss_fun, n_samples=cfg.n_samples_val, centerloss_fun=metric_loss, ce_weight=cfg.ce_weight, cl_weight=cfg.cl_weight)
    with torch.cuda.device(cfg.device_name): # by default cuda:0 is used
        torch.cuda.empty_cache()

    logging.info('validating on train dataset...')
    epoch_train_loss, epoch_train_acc, _ = validation(model, iter(train_loader), loss_fun, n_samples=cfg.n_samples_val, centerloss_fun=metric_loss, ce_weight=cfg.ce_weight, cl_weight=cfg.cl_weight)
    with torch.cuda.device(cfg.device_name): # by default cuda:0 is used
        torch.cuda.empty_cache()

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    tb_writer.add_scalar('average_training_loss', epoch_training_av_loss, epoch)
    tb_writer.add_scalar('validation_loss', epoch_val_loss, epoch)
    tb_writer.add_scalar('validation_accuracy', epoch_val_acc, epoch)
    tb_writer.add_scalar('training_loss', epoch_train_loss, epoch)
    tb_writer.add_scalar('training_accuracy', epoch_train_acc, epoch)

    logging.info('Training loss: {:.4f}'.format(epoch_train_loss))
    logging.info('Training accuracy: {:.4f}'.format(epoch_train_acc))
    logging.info('Validation loss: {:.4f}'.format(epoch_val_loss))
    logging.info('Validation accuracy: {:.4f}'.format(epoch_val_acc))
    logging.info('-' * 10)

    if epoch == 0:
        save_configuration(cfg, os.path.join(output_path, 'training_cnn_configuration.txt'))
        with open(os.path.join(output_path, 'training_cnn_configuration.txt'), 'a') as fh:
            print(model, file=fh)

    # saving the resulted model
    model_file_name = cfg.model_name + '_epoch: {}_acc{:.4f}'.format(epoch, epoch_val_acc)
    saved_model_path_full = os.path.join(output_path, model_file_name + '.pt')
    torch.save(model.state_dict(), saved_model_path_full)
    logging.info('current model was saved to {}'.format(saved_model_path_full))

    t_end_epoch = perf_counter()
    logging.info('training the epoch took {} sec'.format(t_end_epoch - t_start_epoch))
    t_start_epoch = t_end_epoch

tb_writer.close()

t_end = perf_counter()
logging.info('training took {} sec'.format(t_end - t_start))

# saving the best model
model_file_name = cfg.model_name + '_' + cfg.data_staining + '_' + cfg.organ + '_' + cfg.animal + '_' + cfg.string_time + '_acc{:.4f}'.format(best_val_acc)
saved_model_path_full = os.path.join(output_path, model_file_name + '.pt')
torch.save(best_model_wts, saved_model_path_full)
logging.info('best model was saved to {}'.format(saved_model_path_full))
model_for_evaluation = NetworkModel(n_classes=n_classes, path_trained_model=saved_model_path_full, dev=device).to(device)
evaluate_model(val_loader, loss_fun, cfg.n_samples_val, model_for_evaluation)
plt.savefig(os.path.join(output_path, 'confusion_matrix_best_model.png'))

logging.info('tensorboard log was saved to {}'.format(output_path_tb))

# save configuration that can be read together with the models it was used to train
saved_configuration_path_full = os.path.join(output_path, model_file_name + '_training_configuration.pkl')
pickle_configuraton_as_dictionary(cfg, saved_configuration_path_full )
logging.info('pickle configuration file was saved to {}'.format(saved_configuration_path_full ))

if hasattr(cfg, 'test_run') and cfg.test_run is True:
    print('output stored in {} and {} will be removed'.format(output_path, output_path_tb))
    answer = input('Do you really want to remove these folders?')
    if answer in ('yes', 'y', 'Y', 'YES', 'Yes'):
        print('output stored in {} and {} are being removed'.format(output_path, output_path_tb))
        time.sleep(5)
        shutil.rmtree(output_path_tb)
        shutil.rmtree(output_path)
        print('output stored in {} and {} were removed'.format(output_path, output_path_tb))
