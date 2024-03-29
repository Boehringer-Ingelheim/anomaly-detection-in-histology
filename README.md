# Learning Image Representations for Anomaly Detection

-------

This repository contains Pytorch implementation of **training image representations** and **performance evaluation** of the approach introduced in
*I. Zingman, B. Stierstorfer, C. Lempp, F. Heinemann. ["Learning image representations for anomaly detection: application to discovery of
histological alterations in drug development", Medical Image Analysis, 2024.](https://www.sciencedirect.com/science/article/pii/S1361841523003274#da1)*
It is also available on [ArXiv](https://arxiv.org/abs/2210.07675) and temporary has a free access on [Elsevier](https://authors.elsevier.com/a/1iIdF4rfPmE0%7EA).

The paper develops a method for anomaly detection in whole slide images of stained tissue samples in order to routinely screen histopathological data for abnormal alterations in tissue. 

![GitHub Logo](docs/tox_pattern.png)

**Figure** above shows detection of adverse drug reactions by the Boehinger Ingelheim Histological Network (BIHN) based anomaly detection. **A:** The developed Anomaly Detection (AD) method detects induced tissue alterations in
the liver of mouse after administration an experimental compound. The fraction of abnormal tiles increases with the the dosage of the compound. The
compound was previously found to have toxic side effects in toxicological screening by pathologists. Each dot corresponds to a single Whole Slide Image (WSI). Three arrows
correspond to three WSI examples given in **B**. Stars on the top of the graph show statistical significance of the change compared to the mean of control
group. **B:** Examples of detected anomalies. In the control group (left image) blood and a few other not pathological structures result in a low level of false
positives. Detections in compound treated groups (two right images) correspond to pathological alterations and were confirmed by a pathologist.

------
**Requirements**

```PyTorch```, ```NumPy```, ```Pillow```, ```scikit-learn```

The code in the repository was tested under ```Python 3.9``` with GPU 11GB and packages' listed in the ```requirements.txt```.
It, however, should also run with earlier Python versions and smaller GPU memory.

**Experiments** (training image representations and performance evaluation)

![GitHub Logo](docs/Scheme_extended.png)

**Setting up dataset**

* The training dataset with normal tissue of different species, organs, and staining can be downloaded from ```data/train/``` folder from https://osf.io/gqutd/.
This dataset was used for training image representations.

* The evaluation dataset with normal mouse liver tissue and mouse tissue with Non-Alcoholic Fatty Liver Disease (NAFLD) can
be downloaded from ```data/test/``` folder from https://osf.io/gqutd/

* Due to large sizes of zip files it is recommended to download each zip file separately.

* Create the folder structure shown below under the root folder of your repository with the cloned code or in any other location. 
In the last case set ```_prj_root``` variable to the chosen location in ```configs/cfg_training_cnn.py``` and ```configs/cfg_anomaly_detector.py``` configuration files.
We use *.py configuration files, not e.g. yaml, which allows more flexibility and is convenient for prototyping.

* Unzip downloaded data files to the corresponding folders within the created folders structure

* If you want to use pre-trained models (instead of training yourself)
  * download them from ```trained models/``` folder from https://osf.io/gqutd/.
  * unzip and save ```EfficientNet_B0_320_HE_Liver_Mouse_acc0.9762.pt```, ```EfficientNet_B0_320_Masson_Liver_Mouse_acc0.9755.pt``` CNN models, 
  the corresponding ```EfficientNet_B0_320_Masson_Liver_Mouse_acc0.9755.pkl``` and ```EfficientNet_B0_320_HE_Liver_Mouse_acc0.9762.pkl``` 
  anomaly detection models (One-cass SVM classifiers), and the corresponding ```EfficientNet_B0_320_HE_Liver_Mouse_acc0.9762_training_configuration.pkl```
  and ```EfficientNet_B0_320_Masson_Liver_Mouse_acc0.9755_training_configuration.pkl```
  configuration files into e.g. ```BIHN_models``` folder under the project root.
   

**Folders structure for project's input**  
```
 .
 ├── data
     ├── test
     │   ├── NAFLD_anomaly_he_mouse_liver
     │   ├── NAFLD_anomaly_mt_mouse_liver
     │   ├── normal_he_mouse_liver
     │   └── normal_mt_mouse_liver
     └── train
         ├── he_mouse_brain
         ├── he_mouse_heart
         ├── he_mouse_kidney
         ├── he_mouse_liver
         ├── he_mouse_lung
         ├── he_mouse_pancreas
         ├── he_mouse_spleen
         ├── he_rat_liver
         ├── mt_mouse_brain 
         ├── mt_mouse_heart
         ├── mt_mouse_kidney
         ├── mt_mouse_liver
         ├── mt_mouse_lung
         ├── mt_mouse_pancreas
         ├── mt_mouse_spleen
         └── mt_rat_liver
 

```

**Training**

* Set variable ```data_staining``` in ```configs/cfg_training_cnn.py``` to either ```Masson``` (Massosn's Trichrome staining) or ```HE```(H&E staining) values, which will
adjust training image representations for anomaly detection in images of tissue stained correspondingly. If you store the training data in your own location,  update
 ```path_to_data``` accordingly.
* Run ```python train_cnn.py --config configs/cfg_training_cnn.py```
    * The code generates ```train_results/stamp``` folder with trained models (models for each epoch and the best one), confusion matrix, configuration
    and log files, where *stamp* is a unique number that is set for each run. You can redefine the output 
    folder in the configuration file ```configs/cfg_training_cnn.py```, if needed, by updating ```path_to_results```.
 
**Evaluation**

* Set ```cnn_model``` variable in ```configs/cfg_anomaly_detector.py``` to the relative to root path to the trained CNN model, which was 
generated in folder ```train_results/stamp/model_name.pt``` during the training step above. Alternatively, you can set an arbitrary path to the downloaded from https://osf.io/gqutd pre-trained CNN model ```*.pt```.
* If you've downloaded an anomaly model ```*.pkl``` from https://osf.io/gqutd/, set ```ad_model``` to its location. Alternatively, if you want to train anomaly model on your own (once-class classifier), set ```ad_model``` to empty string ```""``` or to ```"CNN_location"```.
* Run ```python anomaly_detector.py --config configs/cfg_anomaly_detector.py```. The code will output evaluation results to ```test_results``` folder.
If anomaly model (once-class classfier) was trained, it will be saved to the folder where CNN model is.  

*Expected performance of anomaly detection with BIHN models*

| Staining         |  Balanced accuracy  |  AU-ROC  |  F<sub>1</sub> score  |
|------------------|:-------------------:|:--------:|:---------------------:|
| H&E              |       94.20%        |  97.33%  |        94.09%         |
| Masson Trichrome |       97.51%        |  99.03%  |        97.51%         |

* To evaluate other algorithms from [Anomalib library](https://github.com/openvinotoolkit/anomalib) on our dataset with NAFLD pathology,
please consult Anomalib section *Custom Dataset*. Particularly, one needs to set appropriate paths in yaml configuration files of the chosen method located at ```anomalib_root/anomalib/models/method/config_file.yaml```.
The paths fields to be set in yaml are ```normal_dir```, ```abnormal_dir```, ```normal_test_dir```, which should point to ```./data/train/*mouse_liver/```, ```./data/test/NAFLD_anomaly_*_mouse_liver```,  ```./data/test/normal_*_mouse_liver``` data paths correspondingly.
The star in paths refers to a particular staining type, ```mt``` or ```he``` you want to experiment with. The ```task``` field should be set to "classification".
* To evaluate [DPA](https://github.com/ninatu/anomaly_detection) appraoch we adapted ```Camalyon16Dataset``` class, reading images from NAFLD dataset.
We obtained our best results for DPA using ```camelyon16``` ```wo_pg_unsupervsed``` default configuration with the following parameters tuned ```inner_dims: 16, latent_dim:16``` (for both decoder and encoder, same values for all layers as in the default configuration), ```initial_image_res:256, max_image_res:256, crop_size: 256```.
Batch size was reduced to 64 to be able to run on 256x256 size images.


**Use of pretrained BIHN models in your own projects**

In oder to use pretrained BIHN models (*.pt files that can be downloaded from https://osf.io/gqutd/) to generate 
feature representations of histopathological images (Masson or H&E) for your own tasks, you can consult the code example in ```model_use_example.py```. 

**Citing**
```markdown
@article{zingman2022anomaly,
      title={Learning image representations for anomaly detection: application to discovery of histological alterations in drug development},
      author={Igor Zingman and Birgit Stierstorfer and Charlotte Lempp and Fabian Heinemann},
      year={2022},
      journal={CoRR},
      volume={abs/2210.07675},    
      eprinttype = {arXiv},
      url = {https://arxiv.org/abs/2210.07675}
}
```
```markdown
@online{NAFLD_dataset,
  author    = {Igor Zingman and Birgit Stierstofer and Fabian Heinemann},
  title     = {{NAFLD} pathology and healthy tissue samples},  
  year      = {2022},
  url       = {https://osf.io/gqutd/},   
}
```










