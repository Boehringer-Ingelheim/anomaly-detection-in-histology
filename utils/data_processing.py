import numpy as np
import torch
import random
import os
import pickle
import time
import logging


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CodesProcessor:

    def __init__(self, cached_codes_path=None, cache_thr=3.0):

        logging.info('code processor is initialized with {} GB maximum GPU memmory threshold'.format(cache_thr))
        self.codes = None
        self.patch_coordinates = {'x': [], 'y': [], 'w': [], 'h': []}
        self.image_names = []
        self.image_labels = []

        if cached_codes_path is not None:
            self.cached_codes_path = cached_codes_path + '/cached_codes_temp' + time.strftime("%H%M%S_%d%m%y") + '.pt'
        else:
            self.cached_codes_path = None

        self.cache_status = False
        self.cache_thr = cache_thr
        #self.avg_pool = nn.AvgPool2d(kernel_size=average_pooling_size)


    def __call__(self, codes, image_names, image_labels, coordinates=None):

        #codes = self.avg_pool(codes)

        if self.codes is None:
            self.codes = codes
        else:
            self.codes = torch.cat((self.codes, codes), dim=0)

        if coordinates is not None:
            self.patch_coordinates['x'].extend(coordinates['x'])
            self.patch_coordinates['y'].extend(coordinates['y'])
            self.patch_coordinates['w'].extend(coordinates['w'])
            self.patch_coordinates['h'].extend(coordinates['h'])
        elif self.patch_coordinates is not None:
            self.patch_coordinates = None # first call
            assert len(self.image_names) == 0

        self.image_names.extend(image_names)
        self.image_labels.extend(image_labels)

        gb_taken = self.codes.element_size() * self.codes.nelement() / 1024 / 1024 / 1024
        if self.cached_codes_path is not None and gb_taken > self.cache_thr:
            logging.info('{} GB memmory taken, {} patch codes gathered, saving to disk to prevent GPU crash'.format(gb_taken, self.codes.shape[0]))
            self.backup_codes()

    def backup_codes(self):
        logging.info('saving codes')

        if self.cache_status:
            cached_codes = torch.load(self.cached_codes_path, map_location='cpu')
            cached_codes = torch.cat((self.codes.cpu(), cached_codes), dim=0)
            torch.save(cached_codes, self.cached_codes_path)
        else:
            torch.save(self.codes.cpu(), self.cached_codes_path)

        self.codes = None
        self.cache_status = True

        logging.info('codes were saved')

    def get_codes(self):

        if self.cache_status:

            codes = torch.load(self.cached_codes_path, map_location='cpu')
            codes = torch.cat((self.codes.cpu(), codes), dim=0)
        else:
            codes = self.codes.cpu()

        codes = codes.numpy()
        codes = np.squeeze(codes)

        return codes

    def get_patch_coordinates(self):
        return self.patch_coordinates

    def get_image_names(self):
        return self.image_names

    def get_image_labels(self):
        return self.image_labels



def read_features(file_name, paths, label_names, n_codes=None, group_method=None, group_size=None, arrange=None):
    """
    Reads and groups features in different ways.

    :param file_name: name of the file with extension that contains latent codes
    :param paths: full paths to the files, each of which contains codes for a specific category
    :param n_codes: number of codes to be sampled for every category - makes the number of samples equal for each category (even when it is not so in the data)
    :param group_method: 'mean', 'concat', or 'max' - a way to combine the data from patch codes to image codes - requires group_size parameter
    :param group_size: the number of codes to be grouped together
    :param label_names: list of string labels corresponding to the data in paths
    :param arrange: If provided, rearranges the feature vectors, such that they will be continuous sequence for particular spatial windows in the original tensor.
    'arrange' is a dictionary with fields win_size and tens_size. win_size is a spatial side size of square patch with channel features.
    tens_size is a spatial side size of the whole tensor of features
    :return features (n_samples x n_features) and sample labels - indexes of 'paths' or 'label_names'
    """

    assert len(paths) == len(label_names), "length of paths and label_names lists must be same"

    def rearrange(features, win_size, tensor_size):
        assert tensor_size % win_size == 0, "tensor_size must be multiple of win_size"
        assert features.shape[0] % (tensor_size*tensor_size) == 0, "number of feature vectors maust be a multiple of tensor_size squared"

        n_tensors = features.shape[0] / tensor_size
        #n_win = np.int(tensor_size / win_size)

        cur_index = 0
        #check_up = []
        rearranged_features = np.zeros_like(features)
        for tens in range(0, features.shape[0], tensor_size*tensor_size):
            for y_block in range(0, tensor_size, win_size):
                for x in range(tensor_size):
                    for y in range(y_block, win_size + y_block):
                        ind = x *tensor_size + y + tens
                        rearranged_features[cur_index] = features[ind, :]
                        #check_up.append(ind)

                        cur_index += 1

        assert features.shape[0] == cur_index, "tensor_size does not match the number of feature vectors"
        #return rearranged_features, check_up
        return rearranged_features



    features = None
    labels = []
    for n, path in enumerate(paths):

        path = os.path.join(path, file_name)
        with open(path, 'rb') as f:
            cur_features = pickle.load(f)

        if arrange:
            cur_features = rearrange(cur_features, arrange['win_size'], arrange['tens_size'])

        feat_vec_len = cur_features.shape[-1]
        if group_method == 'concat':
            cur_features = np.reshape(cur_features, (-1, group_size*feat_vec_len), order='C')
        if group_method == 'concatabs':
            cur_features = np.reshape(np.abs(cur_features), (-1, group_size*feat_vec_len), order='C')
        elif group_method == 'abs':
            cur_features = np.abs(cur_features)
        elif group_method == 'meanabs':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_features = np.mean(np.abs(cur_features), axis=1)
        elif group_method == 'mean':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_features = np.mean(cur_features, axis=1)
        elif group_method == 'maxabs':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_features = np.max(np.abs(cur_features), axis=1)
        elif group_method == 'minabs':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_features = np.min(np.abs(cur_features), axis=1)
        elif group_method == 'norm':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_features = np.linalg.norm(cur_features, axis=1)
        elif group_method == 'meanstd':
            cur_features = np.reshape(cur_features, (-1, group_size, feat_vec_len), order='C')
            cur_1 = np.mean(cur_features, axis=1)
            cur_2 = np.std(cur_features, axis=1)
            cur_features = np.concatenate((cur_1, cur_2), axis=1)

        # use only part of feature vectors to reduce running time
        if n_codes:
            if n_codes > cur_features.shape[0]:
                logging.info('the number of requested feature vectors for {} does not exist'.format(label_names[n]))
                logging.info('using only {} codes'.format(cur_features.shape[0]))
                n_codes_cur = cur_features.shape[0]
            else:
                logging.info('using all {} requested codes for {}'.format(n_codes, label_names[n]))
                n_codes_cur = n_codes

            #ind = np.random.randint(0, high=n_codes_cur, size=n_codes_cur)
            ind = np.random.permutation(cur_features.shape[0])[:n_codes_cur]
            cur_features = cur_features[ind, :]

        else:
            logging.info('using all {} codes for {}'.format(cur_features.shape[0], label_names[n]))

        cur_labels = [n] * cur_features.shape[0]

        if features is not None:
            features = np.concatenate((features, cur_features), axis=0)
        else:  # first run
            features = cur_features

        labels.extend(cur_labels)

    return features, labels






