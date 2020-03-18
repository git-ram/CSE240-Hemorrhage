# CSE240 RSNA Intercranial Hemorrhage Detection
Intracranial hemorrhage is when bleeding occurs inside the cranium and it is one of the top causes of stroke and death in the United States. Since time is a critical factor in the diagnosis of hemorrhage, it is vital to review the symptoms in the images of a patient’s cranium and identify the presence and type of hemorrhage as quickly as possible. In this project, we propose a deep learning model to detect the presence of hemorrhage. 

### Details of the files organised. 
1. The git repo contains both types of experiments, ones run on kaggle as well as the ones run on PRP. 
2. For the experiments run on PRP, the details of the models are in the models folder.(basic_model_binary.py, basic_multilabel.py, convolution_multimodel.py, and cnn_model2_multilabel.py ) The helper folder contains classes used to perform training as well as reading and preprocessing the dicom data. 
3. For the inital experiments(phase 1) which was carried out completely on kaggle, the code can be found in the ipython notebooks imported. 


Note: if running on PRP, make sure the paths are changed to the following:

### Paths to training files        
input_filepath = "../../rsna-intracranial-hemorrhage-detection/"
train_image_filepath = "../../rsna-intracranial-hemorrhage-detection/stage_2_train/"

### Paths to csv files
csv_file_path = "./CSV/down_sampled_positive_data.csv"
csv_file_path_test = "./CSV/hem_positive_test_set.csv"
