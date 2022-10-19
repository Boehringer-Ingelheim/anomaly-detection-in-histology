#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:11:10 2019

@author: zingman
"""
from PIL import Image
import glob
import random
from torch.utils.data import Dataset

import numpy as np
import copy

import logging


import time

Image.MAX_IMAGE_PIXELS =  5000000000
from tqdm import tqdm


class color_transform_skimage():

    def __init__(self, int_type_precision=np.uint8):
        """
        Envelopes skimage color transforms such that they will output integer images suitable for LUT transformation.
        Below there are definitions of parameters that should correspond to the skimage color transform used.
        For color transformations use forward() and backward functions()

        :param int_type_precision: precision with which LUT will be performed. e.g. for np.uin8 it will be LUT 256 values
        """

        # -------these values should be adapted to the color skimage transform used--------
        self.tr_type = np.float  # type of output of color skimage transformation

        # self.tr_min = np.array([16, 16, 16], dtype=self.tr_type) # min values of the output of color transform (YCbCr)
        # self.tr_max = np.array([235, 240, 240], dtype=self.tr_type)  # max values of the output of color transform (YCbCr)

        self.tr_min = np.array([0, 0, 0], dtype=self.tr_type)  # min values of the output of color transform
        self.tr_max = np.array([255, 255, 255], dtype=self.tr_type)  # max values of the output of color transform

        # color transforms used
        # self.forward_transform = rgb2ycbcr  # forward color transform
        # self.backward_transform = ycbcr2rgb  # backward color transform

        self.forward_transform = lambda im_in: np.array(im_in, dtype=self.tr_type) # identity transform (direct RGB use)
        self.backward_transform = lambda im_in: im_in/np.expand_dims(np.expand_dims(self.tr_max, axis=0), axis=0)  # identity transform (direct RGB use), skiimage transforms output in [0, 1] range

        #--------------

        self.dig_type = int_type_precision  # integer type where mixup up transform (LUT) happens
        self.max_intensity = np.iinfo(self.dig_type).max  # number of values - 1 used for LUT and color histograms

        self.input_type = None # type of the input image to be processed, also the output image processed
        self.max_input_type = None # maximal value of the input image being processed

    def get_n_intensity_levels(self):
        return self.max_intensity + 1

    def _digitize(self, im):
        """
        Transforms the input image to the integer type (self.dig_type) rescaling according the input type of the image
        :param im:
        :return:
        """

        im_out = np.array(im, dtype=self.dig_type)
        im = np.moveaxis(im, [0, 1, 2], [1, 2, 0])
        for i, im_channel in enumerate(im):

            if im_channel.max() < self.tr_min[i]:
                self.logger.warning("Warning: skimage tranform can get values that are smaller than min defined in the color_transform_skimage class")
            if im_channel.min() > self.tr_max[i]:
                self.logger.warning("Warning: skimage transform can get values that are larger than max defined in the color_transform_skimage class")

            factor = np.float(self.max_intensity) / (self.tr_max[i] - self.tr_min[i])
            im_channel = (im_channel - self.tr_min[i]) * factor
            im_channel = np.clip(im_channel, 0, self.max_intensity)
            im_channel = np.array(im_channel, dtype=self.dig_type)
            im_out[:, :, i] = im_channel

        return im_out

    def _undigitize(self, im):
        """
        Transforms from the integer type used by LUT to the skimage transform output type (self.tr_type).
        :param im:
        :return:
        """

        im_out = np.array(im, dtype=self.tr_type)
        im = np.moveaxis(im, [0, 1, 2], [1, 2, 0])
        for i, im_channel in enumerate(im):
            factor = (self.tr_max[i] - self.tr_min[i]) / np.float(self.max_intensity)
            im_channel = im_channel * factor + self.tr_min[i]
            im_channel = np.clip(im_channel, self.tr_min[i], self.tr_max[i])
            im_channel = np.array(im_channel, dtype=self.tr_type)
            im_out[:, :, i] = im_channel

        return im_out

    def forward(self, im):
        '''
        im: input ndarray image (can be float or uint)
        return: image integer type (self.dig_type)

        '''
        self.input_type = im.dtype
        self.max_input_type = np.iinfo(self.input_type).max

        im2 = self.forward_transform(im)
        assert im2.dtype == self.tr_type

        im3 = self._digitize(im2)

        return im3

    def backward(self, im):
        """
        Backward color transformation from self.dig_type to the type of the original input image that passed forward() transformation
        :param im:
        :return:
        """

        im1 = self._undigitize(im)
        im2 = self.backward_transform(im1)

        # output of skimage.color transforms are floats in [0, 1] range
        im2 = np.clip(im2, 0.0, 1.0)  # however, values may in practice be higher than 1
        im3 = np.array(im2 * self.max_input_type, dtype=self.input_type)

        return im3


class HistImagesDataset(Dataset):

    def __init__(self, *imsets, n_samples=None, repetition=False, transform=None, dataset_name=''):

        """
                    The dataset is created from any number of image sets (imsets) defined with a class label
                    and a path (location) to images (see example of use in check_up() function). Several paths can belong to the same class.
                    Images and locations are chosen to be read in a random order. The sampling rate for each location is determined
                    by the number of images relative to the other locations or n_samples if given.
                    Output images are PIL images (unless 'transform' changes the type).
                    Images with '_empty' suffix are ignored.
                    Images are read e,g. by indexing as dataset[i], which outputs a dictionary with 'image', 'label', 'string_label', and 'image_name'
                    fields. 'label' field contains a numeric value of a label, while 'string_label' is an string label taken from 'imsets'.
                    The string_label provided with the input imsets can also be accessed with instance.get_str_label(label)

                    :param imsets: sequence of dictionaries with fields 'folder', 'label, 'ext', and optionally pattern
                    'key' (rules used by the UNIX shell e.g. '*67?', '10[1-2,5-7]') that should be a part of a file name of all files of interest.
                    :param n_samples: a number of images to read from each folder (dictionary, location) or a list with different number of samples for each folder (dictionary, location).
                    It is used to restrict the number of images in the case of too large or unbalanced datasets.
                    You can use 'samples_per_location_from_samples_per_class' function defined in this module for
                    defining the same number of images for each class (when number of locations is larger than the number of classes)
                    :param repetition: effective only when n_samples is larger than the number of images for in a folder (location).
                    When False n_samples is reduced to the number of images in the folder, if True n_samples will be sampled from the folder with repetitions.
                    :param transform: pytorch transform to be applied on each read PIL image
                    :param dataset_name: the name of the dataset that will be used for logging

                    split_set(): method can be used to divide the dataset to two disjoints datasets
                    (see example of use in check_up() function). Alternatively, torch.utils.data.random_split
                    can be used.

                    create_subset(): Creates a subset of dataset. The original set is not changed.

                    prepare_mixup(): Computes distribution for each class and switches on mix-up normalization -
                    transfer of color appearance of images to randomly chosen class

        """
        if dataset_name:
            self.logger = logging.getLogger('HistImagesDataset' + ':' + dataset_name)
        else:
            self.logger = logging.getLogger('HistImagesDataset')

        # if given n_samples is integer create list with repeated 'n_samples' for each location
        if n_samples:       
            if isinstance(n_samples, int):
                n_samples = [n_samples] * len(imsets)
            else:
                n_samples = list(n_samples[:]) # copies values in order to avoid changing argument of calling function. List command allows an argument to be either list or tuple
            
            assert isinstance(n_samples, list), 'n_samples keyward should be either integer or a list of numbers'
            assert len(n_samples) == len(imsets), 'n_samples should be of a length equal to the number of image sets'

        self.str_labels = [] # set of used string labels in the dataset (the length equals to the number of classes, or number of different labels)
        self.file_names = []
        self.labels = [] # integer labels for every image file
        n_samples_per_class = []
        for i, imset in enumerate(imsets, 0):

            if 'key' in imset.keys():
                key = imset['key']
                if key is None or len(key) == 0:
                    key = '*'
            else:
                key = '*'

            # reading file names for every location with appropriate file extensions and matching key
            pattern = imset['folder'] + '/' + key + '.' + imset['ext']
            # alphabetic sorting is for reproducability purposes only
            file_names = sorted(glob.glob(pattern))

            # exclude those that do not include '_empty' suffix
            file_names = [x for x in file_names if ('_empty.' + imset['ext']) not in x]

            if len(file_names) == 0:
                self.logger.error('no files were found in: {}'.format(imset['folder']))
            else:
                self.logger.debug('file names in {} were successfully read'.format(imset['folder']))

            # random sampling of file names (useful in the case of large or unbalanced datasets)
            if n_samples:
                if n_samples[i] > len(file_names):
                    if not repetition:
                        self.logger.warning('number of requested samples ({}) is larger then the size of the dataset ({}) in: {}. Using all available samples.'.format(n_samples[i], len(file_names), imset['folder']))
                        n_samples[i] = len(file_names)
                    else:
                        self.logger.warning('number of requested samples ({}) is larger then the size of the dataset ({}) in: {}. Mupliple copies of the same images will be used.'.format(n_samples[i], len(file_names), imset['folder']))

                file_names = sample_with_possible_repetition(file_names, n_samples[i])

            self.file_names += file_names

            if imset['label'] in self.str_labels: # label already appeared
                idx_class = self.str_labels.index(imset['label'])
                n_samples_per_class[idx_class] += len(file_names)
            else: # new label
                self.str_labels.append(imset['label'])
                idx_class = len(self.str_labels) - 1
                assert idx_class == self.str_labels.index(imset['label'])
                n_samples_per_class.append(len(file_names))

            self.labels += [idx_class] * len(file_names)
                
        self.len = len(self.labels)
        
        # randomization of the order of the samples from different locations to be used in __getitem__()
        self.idx = self._random_index()

        self.transform = transform

        self.transform_lut = None # transformation between class intensities is not defined

        self.ImReader = Image.open

        self.n_channels_mixup = [True, True, True]  # for mixup augmentaiton make transform (adapt appearance) for channels with True value only

        self.color_transform = color_transform_skimage()

        self.logger.info(
            'mapping dataset was initialized. {} classes, {} images per class'.format(self.str_labels, n_samples_per_class))

        assert self.get_number_samples_per_class() == n_samples_per_class


    def _random_index(self): # generates random numbers up to the length of the dataset self.len
        return random.sample(range(self.len), self.len)

    def get_str_label(self, int_label):
        return self.str_labels[int_label]
    
    def get_int_label(self, string):
        return self.str_labels.index(string)

    def get_number_samples_per_class(self):

        num_samples_per_class = []
        for str_label in self.str_labels:
            num_samples_per_class.append(sum([self.get_str_label(int_label) == str_label for int_label in self.labels]))

        return num_samples_per_class

    def compute_statistics(self):

        means = 0
        stds = 0
        n_patches_read = 0
        number_of_classes = len(self.str_labels)
        class_mean = [np.zeros((3)) for label in range(number_of_classes)]
        class_std = [np.zeros((3)) for label in range(number_of_classes)]
        n_class_patches_read = [0 for label in range(number_of_classes)]
        self.logger.info("computing statistics of the train dataset")
        for i in tqdm(range(len(self))):

            im, label, _ = self._get_raw_image(i)

            # From PIL to ndarray
            im = np.array(im)

            mean = np.mean(im, axis=(0, 1))
            std = np.std(im, axis=(0, 1))

            means  += mean
            stds += std
            n_patches_read += 1

            class_mean[label] += mean
            class_std[label] += std
            n_class_patches_read[label] += 1

        means = means / n_patches_read
        stds = stds / n_patches_read

        class_means = {}
        class_stds = {}
        for label in range(number_of_classes):
            str_label = self.get_str_label(label)
            class_means[str_label] = class_mean[label] / n_class_patches_read[label]
            class_stds[str_label] = class_std[label] / n_class_patches_read[label]
            self.logger.info('average of {} class is {}, average of std is {}'.format(str_label, class_means[str_label], class_stds[str_label]))

        self.logger.info('average value: {}, std value: {}, based on {} images'.format(means, stds, n_patches_read))

        return means, stds, class_means, class_stds


    def _mixup_normalization(self, im, src_cls):
        """
        :param im: PIL image to be normalized to appearance of a random class
        :param src_cls:  class (int) of the input image
        :return: transformed_im: normalized PIL image
        :return: dst_class: class (int) the image appearance was transformed to
        """

        n_cls = len(self.str_labels)

        # ranomly chose class for allowed destination class appearance
        allowed_dst_class_labs = self._allowed_dst_class(src_cls)
        dst_class_idx = random.randrange(len(allowed_dst_class_labs))
        dst_class = allowed_dst_class_labs[dst_class_idx]

        im = np.array(im) # transform PIL to ndarray

        # use color transformation to get channels that will be normalized to the channels of destination class
        im = self.color_transform.forward(im)

        n_channels = im.shape[2]
        assert n_channels == len(self.n_channels_mixup)
        assert n_channels == self.transform_lut.shape[2]


        transformed_im = np.array(im, copy=True) # copy of an array

        # make per channel normalizations
        for channel in range(n_channels):
            if self.n_channels_mixup[channel]:
                im_channel = im[:,:,channel]
                transformed_im[:,:,channel] = self.transform_lut[src_cls, dst_class, channel][im_channel]

        # transform back normalized channels
        transformed_im = self.color_transform.backward(transformed_im)
        # output PIL image
        transformed_im = Image.fromarray(transformed_im)
        return transformed_im, dst_class

    def _allowed_dst_class(self, src_cls):
        """
        Outputs allowed integer class labels for given input integer label for mix-up augmentation
        (based on label groups in self.mixup_classes)
        :param src_cls: integer class label
        :return: dst_cls: list of integer class labels
        """

        dst_cls = None
        for classes in self.mixup_classes:
            if src_cls in classes:
                dst_cls = classes
                break

        assert dst_cls is not None
        return dst_cls


    def _list_of_lists_strlab_2_list_of_lists_intlab(self, classes_str):
        """
        :param classes_str: list of lists that contains string class labels.
        :return: list of lists that contains ineger class labels
        """
        classes_int = []
        for lst in classes_str:
            classes_int.append([])
            for cls in lst:
                classes_int[-1].append(self.get_int_label(cls))

        return classes_int

    def prepare_mixup(self, mixup_classes=None):

        """
        Prepares normalization transformation between every pair of classes, per channel

        :param mixup_classes: list of lists, each of which contains string labels of classes between which mixup will be done

        """

        # set groups of mixup classes
        if mixup_classes is None:
            self.mixup_classes = self._list_of_lists_strlab_2_list_of_lists_intlab([self.str_labels])
        else:
            self.mixup_classes = self._list_of_lists_strlab_2_list_of_lists_intlab(mixup_classes)

        flattened_mixup_classes = [num for sublist in self.mixup_classes for num in sublist]
        assert len(flattened_mixup_classes) == len(self.str_labels)
        assert len(set(flattened_mixup_classes)) == len(self.str_labels)

        # compute cumulative distribution functions for all classes
        cdf = self._compute_cdfs()

        n_cls = len(cdf)
        n_intensities = len(cdf[0, 0])
        n_channels = len(cdf[0])

        # compute transformation-normalization - look up table per channel, for every pair of classes
        self.transform_lut = np.empty((n_cls, n_cls, n_channels, n_intensities))
        for source_cls in range(n_cls):
            for dist_cls in range(n_cls):
                for channel in range(n_channels):
                    self.transform_lut[source_cls, dist_cls, channel] = self._compute_cdfs_lut(cdf[source_cls, channel], cdf[dist_cls, channel])

        self.logger.info('mix-up is prepared, all images will be transformed with mix-up')

    @staticmethod
    def _compute_cdfs_lut(cdf_source, cdf_target):
        """
        Param: cdf_source, cdf_target cumulative distribution functions of classes to be transformed from and to
        Return:  look up table for normalized source intensities to the target intensities
        """

        assert len(cdf_source) == len(cdf_target)
        cdf_target_intensities = np.arange(len(cdf_target))
        cdf_source_intensities = np.interp(cdf_source, cdf_target, cdf_target_intensities)

        return cdf_source_intensities


    def _compute_cdfs(self):
        """
        Computes cumulative distribution function for each class and each channel
        """

        n_classes = len(self.str_labels)
        n_channels = len(self.n_channels_mixup)
        n_intensity_levels = self.color_transform.get_n_intensity_levels()

        cdf = np.zeros((n_classes, n_channels, n_intensity_levels))

        samples_per_class = np.zeros((n_classes, 1, 1))

        self.logger.info('generation of set of class distributions for mix-up augmentaiton')
        # calculation average cumulative histogram over all images, per class, per channel
        for i in tqdm(range(len(self))):

            im, label, _ = self._get_raw_image(i)
            if im.mode != 'RGB':
                self.logger.warning("not RGB with 8bit depth were not yet tested for mix-up")

            # From PIL to ndarray
            im = np.array(im)
            im = self.color_transform.forward(im)
            assert np.iinfo(im.dtype).max + 1 == n_intensity_levels
            assert np.iinfo(im.dtype).min == 0

            samples_per_class[label] += 1

            im_size = np.prod(im.shape[:2])

            for i in range(n_channels):
                histogram, _ = np.histogram(im[:,:, i].ravel(), bins=n_intensity_levels, range=(0, np.iinfo(im.dtype).max+1))
                cumulative_histogram = np.cumsum(histogram) / im_size
                cdf[label, i] += cumulative_histogram


        # get average cumulative histogram (from sum histogram) over all images
        cdf = cdf / np.repeat(np.repeat(samples_per_class, n_channels, 1), n_intensity_levels, 2)
        assert np.all(np.squeeze(samples_per_class) == np.array(self.get_number_samples_per_class()))

        return cdf


    def __len__(self):
        return self.len

    def _try_read_image(self, path, trials=10):

        pause = 5.0

        for i in range(trials):  # try to read an image a few times for the case of lost connection to a linked folder

            try:
                img = self.ImReader(path)
            except FileNotFoundError:
                if trials != 1:
                    self.logger.warning('file {} was not found, probably lost connection to the linked data folder'.format(path))
                    self.logger.warning('{}/{} read trial was unsuccessful'.format(i + 1, trials))
                    self.logger.warning('Waiting for {} sec'.format(pause))
                    time.sleep(pause)
                    self.logger.warning('resuming')
                img = None
            except RuntimeError as er:
                if trials != 1:
                    self.logger.warning('{}/{} read trial was unsuccessful for {}'.format(i + 1, trials, path))
                    self.logger.warning('unknown error: {}'.format(er))
                    self.logger.warning('Waiting for {} sec'.format(pause))
                    time.sleep(pause)
                    self.logger.warning('resuming')
                img = None
            else:
                break

        return img

    def _get_raw_image(self, n):

        idx = self.idx[n]
        img_path = self.file_names[idx]
        img_label = self.labels[idx]

        img = self._try_read_image(img_path)

        return img, img_label, img_path


    def __getitem__(self, n):

        img, img_label, img_path = self._get_raw_image(n)

        if self.transform_lut is not None:
            img, aug_dst = self._mixup_normalization(img, img_label)
        else:
            aug_dst = None
        
        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img, 'label': img_label, 'string_label': self.get_str_label(img_label), 'image_name': img_path}

        # this is for debugging only
        if self.transform_lut is not None:
            sample['_debug_mixup_dst'] = aug_dst

        
        return sample
    
    def shuffle(self):
        random.shuffle(self.idx)

    def _get_shuffled_idx(self):
        idx = random.sample(self.idx, len(self.idx))

        return idx



    def split_set(self, n_val, transform_validation='same', val_dataset_name='', train_dataset_name='', shuffle=False):

        """
        Separates the dataset into two for training and validation.
        A separate part of the images in the original dataset will be taken for newly created (validation) dataset,
        the other part will be taken by the second created dataset (training).
        The original dataset is not changed.

        :param transform_validation:  transformation to be applied on the created validaiton dataset. If not supplied or if 'same'
        the same transformation as for the original set is applied, if None no transformation will be applied
        :param n_val: the number of images in the validation dataset (overall number from all classes).
        :param val_dataset_name: name of validaiton dataset that willl be used for logging
        :param train_dataset_name: name of training dataset that will be used logging
        :param shuffle: If True, randomizes the choice of images taken for newly created datasets.
         If False, the first part of images will be taken from the dataset for created validation dataset, while the second part for the training datset.
         Ordering of images in the created datasets is also randomized if shuffle is True. shuffle=True might be useful if several different splitted datasets are required
        :return cls2:  newly created separate dataset for validation
        :return cls1:  newly created separate dataset for training
        """

        assert n_val < len(self.idx), 'not enough images for validation'

        # creates two new identical datasets
        cls2 = copy.deepcopy(self) # validation dataset
        cls1 = copy.deepcopy(self) # training dataset

        # set transformation to validation dataset
        if transform_validation:
            if transform_validation == 'same':
                cls2.transform = self.transform
            else:
                cls2.transform = transform_validation
        else:
            cls2.transform = None

        #-------
        if shuffle:
            idx = self._get_shuffled_idx()
        else:
            idx = self.idx

        idx2 = idx[:n_val]
        idx1 = idx[n_val:]

        cls2.file_names = [cls2.file_names[i] for i in idx2]
        cls2.labels = [cls2.labels[i] for i in idx2]
        cls2.len = len(cls2.file_names)
        if shuffle:
            cls2.idx = cls2._random_index()
        else:
            cls2.idx = list(range(cls2.len))

        cls1.file_names = [cls1.file_names[i] for i in idx1]
        cls1.labels = [cls1.labels[i] for i in idx1]
        cls1.len = len(cls1.file_names)
        if shuffle:
            cls1.idx = cls1._random_index()
        else:
            cls1.idx = list(range(cls1.len))

        if val_dataset_name:
            cls2.logger = logging.getLogger('HistImagesDataset' + ':' + val_dataset_name)
        else:
            cls2.logger = logging.getLogger('HistImagesDataset')

        if train_dataset_name:
            cls1.logger = logging.getLogger('HistImagesDataset' + ':' + train_dataset_name)
        else:
            cls1.logger = logging.getLogger('HistImagesDataset')

        cls1.transform_lut = None
        cls2.transform_lut = None

        # # adding to all labels suffix: val
        # for i in range(len(self.str_labels)):
        #     cls2.str_labels[i] += '_val'

        self.logger.info('dataset was splitted {} images in the validation and {} in the train datasets'.format(cls2.len, cls1.len))

        cls1.logger.info('splitted training set was initialized. {} classes, {} images per class'.format(cls1.str_labels, cls1.get_number_samples_per_class()))
        cls2.logger.info('splitted validation set was initialized. {} classes, {} images per class'.format(cls2.str_labels, cls2.get_number_samples_per_class()))

        return cls2, cls1


def samples_per_location_from_samples_per_class(*imsets, samples_per_class: int) -> list:
    """
    :param imsets: sequence of dictionaries with fields 'folder' (locations/folders with the data) 'label' (for each location/folder)
    like in inititalizers of of the  HistImagesDataset or WSIiterableDataset classes
    :param samples_per_class: number of samples to be taken for each class (label)
    :return: list of number of samples to be taken for each location/folder with input data
    """

    assert isinstance(samples_per_class, int), "samples per_class must be integer"
    str_labels = []
    counter = {}
    # counting the number of locations for each class
    for location in imsets:

        str_label = location['label']
        if str_label not in str_labels:
            str_labels.append(str_label)
            counter[str_label] = 1
        else:
            counter[str_label] += 1

    samples_per_location = []
    for location in imsets:
        str_label = location['label']
        samples_per_location.append(round(samples_per_class / counter[str_label]))

    return samples_per_location


def sample_with_possible_repetition(file_names: list, n_samples: int) -> list:
    """
    :param file_names: list
    :param n_samples: number of samples from list to be taken. If the number of samples 'n_samples' is larger than the
     length of the list, output list will have multiple copies of elements from the input 'file_names' list
    :return: list of sampled elements from 'file_names' list
    """

    n_elements = len(file_names)
    n_cycles = n_samples // n_elements
    n_samples_last = n_samples - n_cycles * n_elements
    res = []
    for i in range(n_cycles):
        res += random.sample(file_names, n_elements)

    res += random.sample(file_names, n_samples_last)

    return res












    



            
        
        
        




