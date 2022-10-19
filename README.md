# Learning Image Representations for Anomaly Detection

-------

This repository contains Pytorch implementation of **training image representations** and **performance evaluation** of the approach introduced in
*Zingman et al. ["Learning image representations for anomaly detection: application to discovery of
histological alterations in drug development", CoRR, 2022.](https://arxiv.org/abs/2210.07675)*

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

Repository was tested under ```Python 3.9```, using GPU 11GB, and packages versions are listed in the ```requirements.txt```,
but should also run with earlier Python versions and smaller GPU memory.

**Experiments** (training image representations and performance evaluation)

![GitHub Logo](docs/Scheme_extended.png)

**Setting up dataset**

* The training dataset with normal tissue of different species, organs, and staining can be downloaded from ```data/train/``` folder from here https://osf.io/gqutd/.
This dataset is used for training image representations.

* The evaluation dataset with normal mouse liver tissue and tissue with Non-Alcoholic Fatty Liver Disease (NAFLD) can
be downloaded from ```data/test/``` and ```trained models/``` folders from here: https://osf.io/gqutd/

* Due to large zip file sizes it is recommended to download each zip file separately.

* Create the folder structure shown below under the root folder of your repository with the cloned code or in any other location. 
In the last case set ```_prj_root``` variable to the chosen location in ```configs/cfg_training_cnn.py``` and  ```configs/cfg_anomaly_detector.py```.

* Unzip downloaded data files to the corresponding folders within the created structure

* If you want to use pre-trained models (instead of training yourself)
  * download them from ```trained models/``` folders from here: https://osf.io/gqutd/.
  * unzip and save ```*HE*.pt``` model in ```BIHN_models_HE``` folder, ```*MT*.pt``` model in ```BIHN_models_MT```,
  and the corresponding ```*.pkl``` models (SVM classifiers) in the corresponding ```anomaly_detection_*``` subfolders. 

**Folders structure for project's input and output**  
```
 .
 ├── data
 │   ├── test
 │   │   ├── NAFLD_anomaly_he_mouse_liver
 │   │   ├── NAFLD_anomaly_mt_mouse_liver
 │   │   ├── normal_he_mouse_liver
 │   │   └── normal_mt_mouse_liver
 │   └── train
 │       ├── he_mouse_brain
 │       ├── he_mouse_heart
 │       ├── he_mouse_kidney
 │       ├── he_mouse_liver
 │       ├── he_mouse_lung
 │       ├── he_mouse_pancreas
 │       ├── he_mouse_spleen
 │       ├── he_rat_liver
 │       ├── mt_mouse_brain 
 │       ├── mt_mouse_heart
 │       ├── mt_mouse_kidney
 │       ├── mt_mouse_liver
 │       ├── mt_mouse_lung
 │       ├── mt_mouse_pancreas
 │       ├── mt_mouse_spleen
 │       └── mt_rat_liver
 └── results
     ├── BIHN_models_HE
     │   └── anomaly_detection_EfficientNet_B0_320_HE_Liver_Mouse_acc0.9762
     └── BIHN_models_MT
         └── anomaly_detection_EfficientNet_B0_320_Masson_Liver_Mouse_acc0.9755

```

**Training**

* Set variable ```data_staining``` in ```configs/cfg_training_cnn.py``` to either ```Masson``` (Massosn's Trichrome staining) or ```HE```(H&E staining) values, which will
adjust training image representations for anomaly detection in images of tissue stained correspondingly. 
* Run ```python train_cnn.py --config configs/cfg_training_cnn.py```
 
**Evaluation**

* Set ```cnn_model``` variable in ```configs/cfg_anomaly_detector.py``` to the path to existing trained PyTorch model, which was 
either automatically generated in folder ```results/stamp/model_name.pt``` during the training step above, or to the path to copied pre-trained model in 
folder ```results/BIHN_models_staining/model_name.pt```  (see **setting up dataset** section above). 
* Run ```python anomaly_detector.py --config configs/cfg_anomaly_detector.py```. 

*Expected performance of anomaly detection with BIHN models*

Staining | Balanced accuracy | AU-ROC | F<sub>1</sub> score
--- |-------------------|--------| -------
 H&E               | 94.20% | 97.33% | 94.09%
 Masson Trichrome  | 97.51% | 99.03% | 97.51%
 
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










