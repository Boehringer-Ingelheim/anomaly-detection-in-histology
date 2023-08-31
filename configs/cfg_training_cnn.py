import numpy as _np
import time as _time
import os as _os

string_time = _time.strftime("%y%m%d_%H%M%S")

test_run = False # set to True if this is the fast test run, no results are saved. Allows checking for no run-time errors
seed_number = 500

# root folder for inputs (data) and outputs  - defined here as the root folder of the code repository.
# You can define your own arbitrary root path for data (outside of the project code)
_prj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
#_prj_root = '/your root path to input and output data/'

# path to trained model, tensorboard logs etc. You can define your own arbitrary path.
path_to_results = _os.path.join(_prj_root, "train_results")

# path to the training data. You can define your own arbitrary path.
path_to_data = _os.path.join(_prj_root, "data/train")

# Target combination of specie, organ, and staining, the tissue where anomalies should be found
data_staining = "Masson"
#data_staining = "HE"
organ = "Liver"
animal = "Mouse"
#animal = "Rat"

# training data - all healthy, but might be cluttered with low quality examples
path_to_tissues = (
    {'folder': path_to_data + "mt_mouse_brain", 'label': 'brain', 'ext': 'png'},
    {'folder': path_to_data + "mt_mouse_heart", 'label': 'heart', 'ext': 'png'},
    {'folder': path_to_data + "mt_mouse_kidney", 'label': 'kidney', 'ext': 'png'},
    {'folder': path_to_data + "mt_mouse_lung", 'label': 'lung', 'ext': 'png'},
    {'folder': path_to_data + "mt_mouse_pancreas", 'label': 'pancreas', 'ext': 'png'},
    {'folder': path_to_data + "mt_mouse_spleen", 'label': 'spleen', 'ext': 'png'},

    {'folder': path_to_data + "mt_mouse_liver", 'label': 'liver', 'ext': 'png'},

    {'folder': path_to_data + "mt_rat_liver", 'label': 'liver_rat', 'ext': 'png'},

    {'folder': path_to_data + "he_mouse_brain", 'label': 'he_brain', 'ext': 'png'},
    {'folder': path_to_data + "he_mouse_kidney", 'label': 'he_kidney', 'ext': 'png'},
    {'folder': path_to_data + "he_mouse_spleen", 'label': 'he_spleen', 'ext': 'png'},
    {'folder': path_to_data + "he_mouse_pancreas", 'label': 'he_pancreas', 'ext': 'png'},
    {'folder': path_to_data + "he_mouse_heart", 'label': 'he_heart', 'ext': 'png'},
    {'folder': path_to_data + "he_mouse_lung", 'label': 'he_lung', 'ext': 'png'},

    {'folder': path_to_data + "he_mouse_liver", 'label': 'he_liver', 'ext': 'png'},

    {'folder': path_to_data + "he_rat_liver", 'label': 'he_liver_rat', 'ext': 'png'},
    )

number_of_classes = len(set([loc['label'] for loc in path_to_tissues]))

centerloss_classes = 'derived'
#centerloss_classes = None # do not use centerloss
#centerloss_classes = ('liver', 'heart')
#centerloss_classes = 'liver'
#centerloss_classes = 'he_liver'
#centerloss_classes = ('liver', 'liver_rat')
#centerloss_classes = 'all' # tighten distribution for all the classes
if centerloss_classes == 'derived':
    if data_staining == "HE" and animal == "Mouse" and organ == "Liver":
        centerloss_classes = 'he_liver'
    elif data_staining == "Masson" and animal == "Mouse" and organ == "Liver":
        centerloss_classes = 'liver'
    elif data_staining == "HE" and animal == "Rat" and organ == "Liver":
        centerloss_classes = 'he_liver_rat'
    elif data_staining == "Masson" and animal == "Rat" and organ == "Liver":
        centerloss_classes = 'liver_rat'
    else:
        raise RuntimeError("such a combination of animal staining and organ is not yet implemented")

mixup_classes = [['brain', 'heart', 'kidney', 'lung', 'pancreas', 'spleen', 'liver', 'liver_rat'],
                 ['he_brain', 'he_kidney', 'he_spleen','he_pancreas', 'he_heart', 'he_liver', 'he_liver_rat', 'he_lung']]

#mixup_classes = False # False or comment out


#model_name = 'VGG_11' # class defined in pretrained_networks.py
#model_name = 'DenseNet_121' # class defined in pretrained_networks.py
#model_name = 'ResNet_18'
#model_name = 'DenseNet_121_512'
model_name = 'EfficientNet_B0_320'
#model_name = 'EfficientNet_B0'
#model_name = 'EfficientNet_B2_352'
#model_name = 'EfficientNet_B2'
#model_name = 'ConvNeXt'
#model_name = 'VT_B_32'

# arbitrary description of th experiment
description = 'test'

device_name = "cuda:0"

num_workers = 3 # should be at least 4 times the number of GPUs used. Beyond that almost no speed gain

# image transformations
normalize_mean = (0.5788, 0.3551, 0.5655)
normalize_std = (1, 1, 1)

aug_brightness=(0.8, 1.2)
aug_contrast=(0.8, 1.2)

n_samples_train_per_class = 6920 # number of samples to be taken for each class for training + validation
n_samples_val = _np.int(n_samples_train_per_class / 10 * number_of_classes)

batch_size = 64
num_epochs = 15
patch_size = (256, 256)
#patch_size = (224, 224)
cl_weight = 1.0 # center-loss weight
ce_weight = 1.0 # cross entropy weight
model_lr= 0.001 # learning rate
ce_momentum= 0.9 # cross entropy loss, momentum

train_step_show = 10

if test_run:
    mixup_classes = False
    n_samples_train_per_class = 500
    n_samples_val = _np.int(n_samples_train_per_class / 10 * number_of_classes)
    num_epochs = 2


