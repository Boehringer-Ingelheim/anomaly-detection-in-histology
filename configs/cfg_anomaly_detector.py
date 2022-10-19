import os as _os
import pickle as _pickle
import os

show_non_liver = True # used for t-sne visualization of non liver data from auxiliary task
seed_number = 500

# root folder for inputs and outputs
_prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#_prj_root = '/you root input and output data/'


_code_output_path = os.path.join(_prj_root,"results")
_train_data = os.path.join(_prj_root, "data/train")
_test_data = os.path.join(_prj_root, "data/test")

aug_saturation=(0.4, 1.6)
aug_hue= (-0.05, 0.05) #(-0.01, 0.01)

augmentation = True #True  # augment (one class) training examples (useful for larger models like densenet)

csv_liver_tissue_anomalies = 'liver_tissue_anomalies.csv'
csv_liver_tissue_testnormals = 'liver_tissue_testnormals.csv'

dev = "cuda:0"

batch_size = 32

#n_features_visualization = {'normal': 1000, 'NAS anomaly': 1000} # labels should correspond to the labels of datasets
n_features_visualization = {'brain': 1000, 'heart': 1000, 'kidney': 1000, 'lung': 1000, 'pancreas': 1000, 'spleen': 1000,
                            'liver': 1000, 'liver_rat': 1000, 'he_brain': 1000, 'he_kidney': 1000, 'he_spleen': 1000,
                            'he_pancreas': 1000, 'he_heart': 1000, 'he_liver': 1000, 'he_lung': 1000, 'he_liver_rat': 1000,
                            'normal': 1000, 'NAS anomaly': 300} # labels should correspond to the labels of datasets

test_normal_patches_max = None # None - use all
test_anomaly_patches_per_class_max = None # None - use all
visual_test_auxiliary_patches_per_class = 1000
train_patches_for_train_max = None # None - use all

# one-class classifier to be used
clf = "svm.OneClassSVM(nu=0.1, kernel='rbf')"
#clf = "svm.OneClassSVM(nu=0.05, kernel='rbf')"
#clf = "LocalOutlierFactor(n_neighbors=30, novelty=True, metric='minkowski')"

ext2save = 'png'

save_images = True  # write png images in addition to HALO annotations for all required WSI
save_n_FN = 100 # number of falsely classified anomaly patch images to be saved
save_n_FP = 100 # number of falsely classified normal patch examples to be saved

description = "cnn trained on sampled data"
anomaly_model_folder = 'anomaly_detection' # prefix of subfolder where results will be saved


#cnn_model = '' # ImageNet pretrained
cnn_model = _code_output_path + "221002_173403/EfficientNet_B0_320_best_HE_Liver_Mouse_2022-10-02_20:22:16.659299_acc0.9754.pt" # auxiliary task trained on sampled data (also liver sampled folder), seed 500
#cnn_model = _code_output_path + "221002_173134/EfficientNet_B0_320_best_Masson_Liver_Mouse_2022-10-02_20:17:16.212337_acc0.9765.pt" # auxiliary task trained on sampled data (also liver sampled folder), seed 500
#cnn_model = _code_output_path + "221001_234242/EfficientNet_B0_320_best_HE_Liver_Mouse_2022-10-02_02:04:49.050910_acc0.9779.pt" # auxiliary task trained on sampled data, seed 500
#cnn_model = _code_output_path + "221001_234155/EfficientNet_B0_320_best_Masson_Liver_Mouse_2022-10-02_02:04:49.027924_acc0.9784.pt" # auxiliary task trained on sampled data, seed 500
#cnn_model = _code_output_path + "220930_154739/EfficientNet_B0_320_best_Masson_Liver_Mouse_2022-09-30_18:21:12.188338_acc0.9758.pt" # # auxiliary task trained on sampled data, seed 300
#cnn_model = _code_output_path + "220930_161747/EfficientNet_B0_320_best_HE_Liver_Mouse_2022-09-30_18:44:50.936995_acc0.9765.pt" # auxiliary task trained on sampled data, seed 300
#cnn_model = _code_output_path + "220724_000714/EfficientNet_B0_320_best_Masson_Liver_Mouse_2022-07-24_03:07:38.540309_acc0.9755.pt" # used for paper - comparison to NAS study
#cnn_model = _code_output_path + "220724_000900/EfficientNet_B0_320_best_HE_Liver_Mouse_2022-07-24_03:06:00.582888_acc0.9762.pt" # used for paper - tox study


# getting parameters from training_cnn configuration file (if was saved)
_path_to_configuration_pkl = _os.path.split(cnn_model)[0]

_path_to_configuration_pkl = _os.path.join(_path_to_configuration_pkl, 'training_configuration.pkl')
try:
    _pkl_dic = _pickle.load(open(_path_to_configuration_pkl, 'rb'))
except FileNotFoundError:
    print("Meta data cannot be read. Probable reason: the old version of the trained model did not include meta data, or no model was given")
    print("Using parameters from testing configuration file if defined")
    pass
else:
    print('taking parameters from saved along the model meta data from training organs classifier')
    n_trained_classes = int(_pkl_dic['number_of_classes'])

    normalize_mean = _pkl_dic['normalize_mean']
    normalize_std = _pkl_dic['normalize_std']

    aug_brightness = _pkl_dic['aug_brightness']
    aug_contrast = _pkl_dic['aug_contrast']

    model_architecture = _pkl_dic['model_name']

    patch_size = _pkl_dic['patch_size']

    try:
        animal = _pkl_dic['animal']
    except:
        pass

    try:
        data_staining = _pkl_dic['data_staining']
    except:
        if _pkl_dic['centerloss_classes'] == 'liver':
            data_staining = "Masson"
        elif _pkl_dic['centerloss_classes'] == 'he_liver':
            data_staining = "HE"
        elif _pkl_dic['centerloss_classes'] == 'he_liver_rat':
            data_staining = "HE"
        elif _pkl_dic['centerloss_classes'] == 'liver_rat':
            data_staining = "Masson"
        else:
            raise RuntimeError("Warning: trained center loss class is not valid")

# ----data--------
paths_non_liver_tissues_test = ()
if show_non_liver:
    paths_non_liver_tissues_test = (
        {'folder': _train_data + "mt_mouse_brain", 'label': 'brain', 'ext': 'png'},
        {'folder': _train_data + "mt_mouse_heart", 'label': 'heart', 'ext': 'png'},
        {'folder': _train_data + "mt_mouse_kidney", 'label': 'kidney', 'ext': 'png'},
        {'folder': _train_data + "mt_mouse_lung", 'label': 'lung', 'ext': 'png'},
        {'folder': _train_data + "mt_mouse_pancreas", 'label': 'pancreas', 'ext': 'png'},
        {'folder': _train_data + "mt_mouse_spleen", 'label': 'spleen', 'ext': 'png'},

        {'folder': _train_data + "mt_mouse_liver", 'label': 'liver', 'ext': 'png'},

        {'folder': _train_data + "mt_rat_liver", 'label': 'liver_rat', 'ext': 'png'},

        {'folder': _train_data + "he_mouse_brain", 'label': 'he_brain', 'ext': 'png'},
        {'folder': _train_data + "he_mouse_kidney", 'label': 'he_kidney', 'ext': 'png'},
        {'folder': _train_data + "he_mouse_spleen", 'label': 'he_spleen', 'ext': 'png'},
        {'folder': _train_data + "he_mouse_pancreas", 'label': 'he_pancreas', 'ext': 'png'},
        {'folder': _train_data + "he_mouse_heart", 'label': 'he_heart', 'ext': 'png'},
        {'folder': _train_data + "he_mouse_lung", 'label': 'he_lung', 'ext': 'png'},

        {'folder': _train_data + "he_mouse_liver", 'label': 'he_liver', 'ext': 'png'},

        {'folder': _train_data + "he_rat_liver", 'label': 'he_liver_rat', 'ext': 'png'},

        )


paths_liver_anomaly_test = () # anomalies for quantitative test (labeled png)
if data_staining == "Masson":
    paths_liver_anomaly_test = (

        {'folder': _test_data + "/NAFLD_anomaly_mt_mouse_liver/", 'label': 'NAS anomaly', 'ext': 'png'},

    )
elif data_staining == "HE":
    paths_liver_anomaly_test = (

        {'folder': _test_data + "/NAFLD_anomaly_he_mouse_liver/", 'label': 'NAS anomaly', 'ext': 'png'},

    )


paths_normal = () # healthy for training one class classifier
if data_staining == "Masson" and animal == "Mouse":
    paths_normal = (

        {'folder': _train_data + "mt_mouse_liver", 'label': 'normal_train', 'ext': 'png'},

     )
elif data_staining == "HE" and animal == "Mouse":
    paths_normal = (

        {'folder': _train_data + "he_mouse_liver", 'label': 'normal_train', 'ext': 'png'},

    )
elif data_staining == "Masson" and animal == "Rat":
    paths_normal = (
    {'folder': _train_data + "mt_rat_liver", 'label': 'normal_train', 'ext': 'png'},
    )
elif data_staining == "HE" and animal == "Rat":
    paths_normal = (
        {'folder': _train_data + "he_rat_liver", 'label': 'normal_train', 'ext': 'png'},
    )

paths_normal_test = () # healthy for quantitative tests and visual test, those were not used for training
if data_staining == "Masson":
    paths_normal_test = (

        {'folder': _test_data + "/normal_mt_mouse_liver/", 'label': 'normal', 'ext': 'png'},

    )
elif data_staining == "HE":
    paths_normal_test = (

        {'folder': _test_data + "/normal_he_mouse_liver/", 'label': 'normal', 'ext': 'png'},

    )

