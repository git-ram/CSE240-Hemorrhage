# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from keras.utils import multi_gpu_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
from sklearn.utils import resample
from keras.models import save_model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Concatenate
from keras.applications.densenet import *
from numbers import Number
from keras.utils import to_categorical
import gc
from cachetools import TTLCache
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU






# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


        
input_filepath = "../../rsna-intracranial-hemorrhage-detection/"
train_image_filepath = "../../rsna-intracranial-hemorrhage-detection/stage_2_train/"


# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files

print(__doc__)

filename = train_image_filepath + "ID_00019828f.dcm"
dataset = pydicom.dcmread(filename)

# Normal mode:
print()
print("Filename.........:", filename)
print()

print("Modality.........:", dataset.Modality)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

# use .get() if not sure the item exists, and want a default value if missing
print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

# plot the image using matplotlib
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.title('Before Windowing', y=-0.17)
plt.savefig('before-windowing.png')

plt.show()





def brain_window(img):
    
    window_center =  img.WindowCenter if isinstance(img.WindowCenter, Number) else img.WindowCenter[0] 
    window_width = img.WindowWidth if isinstance(img.WindowWidth, Number) else img.WindowWidth[0] 
    slope, intercept  =  img.RescaleSlope, img.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = img.pixel_array
    img = img * dataset.RescaleSlope + intercept
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    # Normalize
    img = (img - img_min) / (img_max - img_min)
    return img
    
    
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = [a for a,b in zip(x,y) if b == yi ]
        class_xs.append((yi, elems))
        if min_elems == None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.extend(x_)
        ys.extend(y_)

    print (xs[:10], ys[:10])
    return xs,ys
    
    
    
    
    
#########making y with all labels ################
#####multilabel###############
filename = train_image_filepath + "ID_00019828f.dcm"
file_dcm = pydicom.dcmread(filename)
print(file_dcm.pixel_array.shape)
def get_two_class_labels(csv_file_path, stratify_percentage=1):
    """returns a list of tuples where the first value is the file id and the second is the label
    [('ID_00019828f', 0)]
    """
    
    input_dataframe = pd.read_csv(csv_file_path)
    #filtered_input_dataframe = input_dataframe[input_dataframe['ID'].apply(lambda x : 'any' in x) ]
    files_with_ids = []

    
   # print(input_dataframe.columns.values)
    X = list(input_dataframe['ID'])
    y_dataframe = input_dataframe.drop(input_dataframe.columns[[0,1,7]], axis = 1)

    #print(y_dataframe.head)
  
    #y = [y_dataframe.columns.values.tolist()] + y_dataframe.values.tolist()
    y =  y_dataframe.values.tolist()
    #print (y[0])
    #print(len(X))
    #print(len(y))
    
    num_samples = int(stratify_percentage * len(X))
    print("Num Samples :", num_samples)
    
    for k,v in list(zip(X, y)) :
        files_with_ids.append( ("_".join(k.split('_')[:2]), v))
        
    return files_with_ids
        
    

def get_images(image_folder_root, image_ids):
    X = []
    for file_name in image_ids:
        try:
            current_file = pydicom.dcmread(image_folder_root + file_name + '.dcm')
            pixel_array = current_file.pixel_array
            if (pixel_array.shape != (512,512)):
                continue
            X.append(pixel_array)
        except ValueError:
            continue
    return np.asarray(X).reshape(len(X), 512, 512, 1)
    
    
    
#########making y with all labels ################
#####multilabel###############
def get_two_class_labels_fortest(csv_file_path_test, stratify_percentage=1):
    """returns a list of tuples where the first value is the file id and the second is the label
    [('ID_00019828f', 0)]
    """
    
    test_dataframe = pd.read_csv(csv_file_path_test)
    #filtered_input_dataframe = input_dataframe[input_dataframe['ID'].apply(lambda x : 'any' in x) ]
    files_with_ids_fortest = []
    
   # print(input_dataframe.columns.values)
    X_test = list(test_dataframe['ID'])
    
    print("Testing sample",X_test[0])
    
    y_test_df = test_dataframe.drop(test_dataframe.columns[[0,6]], axis = 1)
                  
    print("testing y samples")
    
  
    #y = [y_dataframe.columns.values.tolist()] + y_dataframe.values.tolist()
    y_test = y_test_df.values.tolist()
    #print (y[0])
    #print(len(X))
    #print(len(y))
    
    num_samples_train = int(stratify_percentage * len(X))
    num_samples_test = int(stratify_percentage * len(X_test))
    print("Num Samples in Training :", num_samples_train)
    print("Num Samples in Testing :", num_samples_test)
    
    for k,v in list(zip(X_test, y_test)) :
        files_with_ids_fortest.append( ("_".join(k.split('_')[:2]), v))
        
    return files_with_ids_fortest

def get_images(image_folder_root, image_label_list):
    """returns a list of tuples with ('ID',label,file) where file is the ndarray (with a readable shape )"""
    file_dcm=[]
    X_test = []
    y_test = []
    for file_name,label in image_label_list:
        try:
            current_file = pydicom.dcmread(image_folder_root + file_name + '.dcm')
            pixel_array = current_file.pixel_array
            if (pixel_array.shape != (512,512)):
                continue
            file_dcm.append((file_name,label,brain_window(current_file)))
            y_test.append(label)
            X_test.append(pydicom.dcmread(image_folder_root + file_name + '.dcm').pixel_array)
        except ValueError:
            continue
    return X_test,y_test
    
    
    
    
###using multilabel dataset
csv_file_path = "./CSV/down_sampled_positive_data.csv"
csv_file_path_test = "./CSV/hem_positive_test_set.csv"



image_folder_root = train_image_filepath
files_with_ids = get_two_class_labels(csv_file_path,stratify_percentage=1)
X,y = [ x for x,y in files_with_ids], [y for x,y in files_with_ids]
print (len(files_with_ids))
print(type(y))
print(type(X))

files_with_ids_fortest = get_two_class_labels_fortest(csv_file_path_test,stratify_percentage=1)
X_test,y_test = [ x_test for x_test,y_test in files_with_ids_fortest], [y_test for x_test,y_test in files_with_ids_fortest]
print (len(files_with_ids_fortest))
print((y_test[0]))
print(type(X_test[0]))




class Model():
    
    
    def fit(self,X,y):
        raise NotImplemetedError()
    def predict(self, X):
        """Takes test data and returns the label probabilities """
        raise NotImplemetedError()

class Basic(Model):
    """intput dimension is the shape of the input"""
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(400,input_shape=(512,512)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dense(50))
        self.model.add(Activation('relu'))
        self.model.add(Dense(25))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        self.model.add(Dense(5))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

 
        
    
    def fit(self, X,y):
        self.model.fit(x=X,y=y,epochs=1,batch_size=8)
    def predict(self, X):
        self.model.predict(X)
    def save(self,filename):
        self.model.save(filename)




#################################### INCEPTION_RESNET_V2 ############################
from keras.layers import Input, Concatenate
from keras.applications.inception_resnet_v2 import InceptionResNetV2

img_input = Input(shape=(512,512,1))
img_conc = Concatenate()([img_input, img_input, img_input])

class Incept_ResNetV2(Model):
    """intput dimension is in the format (channels, height, width)"""
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.model = InceptionResNetV2(include_top=False,   # whether to have fully connected layer or not
                        weights='imagenet',                # pretrained on imagenet, or None for random weights
                        input_tensor=img_conc,
                        input_shape=(input_dimension[0],input_dimension[1], 3))
        
        self.my_model = Sequential()
        self.my_model.add(self.model)
        self.my_model.add(Flatten())
        self.my_model.add(Dense(5, activation='sigmoid'))
        
        
        self.my_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    
    def fit(self, X,y):
        self.my_model.fit(x=X,y=y,batch_size=8, epochs=1)
    def predict(self, X):
        self.my_model.predict(X)





#Has an image loader for providing the image for a given id in the dataset 
#Written separately so that we can add any preprocessing steps here while the image is being loaded into memory
class DataLoader:
    def __init__(self,base_file_path, cache_size=500, ttl_seconds=20):
        self.base_file_path = base_file_path
        self.cache = TTLCache(maxsize=cache_size,ttl=ttl_seconds)
        
        
    ##will apply only brain windowing while loading the image for now. Need to change this to apply all windowing functions. 
    def load_image(self, image_id):
        if image_id in self.cache:
            return self.cache[image_id]
        
        else:
            current_file = pydicom.dcmread(image_folder_root + image_id + '.dcm')
            pixel_array = brain_window(current_file)
            self.cache[image_id] = pixel_array
            return pixel_array
    def trigger_expire(self):
        self.cache.expire()
        
        
        
        
        
        
        
from keras.preprocessing.image import ImageDataGenerator

class  ModelTrainer(object):
    
    def __init__(self, dataloader, training_batch_size=8, split_size = 400):
        
        self.dataloader = dataloader
        self.split_size = split_size
       
    
    """
    Takes a dataframe with X_y in it, uses Keras image generator
    """
    def fit_v1(self, X_y_dataframe, model, epochs=5, training_batch_size=8):
        
        datagen = ImageDataGenerator(validation_split=0.2)
        X_y_dataframe['ID'] = X_y_dataframe['ID'] + '.DCM'
        
        train_generator = datagen.flow_from_dataframe(dataframe=X_y_dataframe, directory=None,
                                             x_col='ID',
                                             y_col=['epidural', 'intraparenchymal','intraventricular','subarachnoid', 'subdural'],
                                             target_size=(512,512),
                                             color_mode = 'grayscale',
                                             class_mode='raw',
                                             batch_size=training_batch_size,
                                             subset='training',
                                             seed=7)
 
        validation_generator = datagen.flow_from_dataframe(dataframe=X_y_dataframe, directory=None,
                                             x_col='ID',
                                             y_col=['epidural', 'intraparenchymal','intraventricular','subarachnoid', 'subdural'],
                                             target_size=(512,512),
                                             color_mode = 'grayscale',
                                             class_mode='raw',
                                             batch_size=training_batch_size,
                                             subset='validation',
                                             seed=7)
        model.my_model.fit_generator(train_generator, steps_per_epoch=train_generator.n, epochs=epochs,
                            validation_data=validation_generator, validation_steps = validation_generator.n)
 
        
        
        
        
        
    
    """Takes X and y as the file name and labels and """    
    def fit(self, X,y,model, epochs=5, training_batch_size=8):
        splits = len(y) // self.split_size +1
        
        splitter = StratifiedKFold(n_splits=splits, random_state=None, shuffle=True)
        count = 1
        while epochs > 0:
            X,y = shuffle(X,y)
            print("Starting epoch ",count)
            ##TODO: add a better split and shuffle mechanism
            processed = 0
            while processed < len(y):
                batch_imgs = []
                batch_labels = []
            
                current_x = X[processed:min(processed+self.split_size,len(y))]
                current_y = y[processed:min(processed+self.split_size,len(y))]
                for img,label in zip(current_x,current_y):
                    image = self.dataloader.load_image(img)
                    
                    ##Figure out how many images are getting ignored because of this assumption
                    ##check if all reshape operations can happen in the dataloader
                    if image.shape != (512,512):
                        continue
                        
                    batch_imgs.append(image)
                    batch_labels.append(label)
                print("Length of images",len(batch_imgs))
                print("Using  batch with size", len(batch_imgs), len(batch_labels), "Processed ", processed, "Total ", len(y))
                model.fit(np.array(batch_imgs).reshape(len(batch_imgs), 512, 512, 1) ,np.array(batch_labels))
                #model.predict(np.array(batch_imgs),np.array(batch_labels))
                self.dataloader.trigger_expire()
                del batch_imgs
                del batch_labels
                
                gc.collect()
                processed +=self.split_size
            print("Ending epoch", count)
            #save_model(model.model,"epoch-model-four.hdf5")
            #model.save("basic-model-1-epochs.h5")
            #model.model.save_model(model,h5pyBasic-model-1-epoch)
            epochs-=1
            count+=1
        
                
        return model
    
    def predict(self, X_test,y_test,model,epochs=5, training_batch_size=8 ):
            
            splits = len(y) // self.split_size +1
            splitter = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
            processed = 0
            while processed < len(y_test):
                batch_imgs_test = []
                batch_labels_test = []
            
                current_x_test = X_test[processed:min(processed+self.split_size,len(y_test))]
                current_y_test = y_test[processed:min(processed+self.split_size,len(y_test))]
                for img,label in zip(current_x_test,current_y_test):
                    image = self.dataloader.load_image(img)
                  
                    
                    ##Figure out how many images are getting ignored because of this assumption
                    ##check if all reshape operations can happen in the dataloader
                    if image.shape != (512,512):
                        continue
                           
                    batch_imgs_test.append(image)
                    batch_labels_test.append(label)
                #print("Length of images",len(batch_imgs_test))
                #print("using  batch with size", len(batch_imgs_test), len(batch_labels_test), "Processed ", processed, "Total ", len(y_test))

                
                batch_imgs_test = np.array(batch_imgs_test)
               # print("Length of images",len(batch_imgs_test))
               # print("Shape before reshaping",batch_imgs_test.shape)
            
            #predict_input = batch_imgs_test.reshape(len(X_test),2)
            
            #batch_imgs_test = np.expand_dims(batch_imgs_test, axis=1)
            #batch_imgs_test = np.expand_dims(batch_imgs_test, axis=1)
            
                #print("shape of input to predict",batch_imgs_test.shape)
                preds = model.predict(batch_imgs_test)
                #print("the predicted y is of length : ", len(preds))
                #print("first sample prediciton is ", preds[0])
                ##classwise precision, recall 
                ##classwise precision, recall 
                acc_sum = 0
                true_pos = 0
                false_pos = 0
                true_neg = 0
                false_neg = 0
                
                class_true_pos = np.zeros([5,1])
                class_true_neg = np.zeros([5,1])
                class_false_pos = np.zeros([5,1])
                class_false_neg = np.zeros([5,1])
                
                class_recall = np.zeros([5,1])
                class_precision= np.zeros([5,1])
                
                
                for i in range(len(preds)):
                    for j in range(len(preds[0])):
                        if(preds[i][j])>=0.5:
                            preds[i][j] = 1
                        if(preds[i][j]) < 0.5 :
                            preds[i][j] = 0
                for a in range(len(preds)):
                    if(np.all(batch_labels_test[a] == preds[a])):
                        acc_sum = acc_sum +1;
                    for b in range(len(preds[a])):
                        if(batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 1):
                           # acc_sum = acc_sum+1
                            class_true_pos[b] = class_true_pos[b]+1
                            true_pos = true_pos+1
                            
                        if(batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 0):   
                            class_true_neg[b] = class_true_neg[b]+1
                            true_neg = true_neg+1
                            
                        if(batch_labels_test[a][b] < preds[a][b]):
                            class_false_pos[b] = class_false_pos[b]+1
                            false_pos = false_pos+1
                            
                        if(batch_labels_test[a][b] > preds[a][b]):
                            class_false_neg[b] = class_false_neg[b]+1
                            false_neg = false_neg+1
                            
                accuracy = (acc_sum/len(preds))*100
                recall =    (true_pos) / (true_pos + false_neg)
                precision =  (true_pos)/(true_pos + false_pos)
                
                for c in range(len(class_recall)):
                    class_recall[c] = (class_true_pos[c]) / (class_true_pos[c] + class_false_neg[c])
                    class_precision[c] = (class_true_pos[c]) / (class_true_pos[c] + class_false_pos[c])
                    
                print("Accuracy",accuracy)
                print("Recall",recall)
                print("Precision",precision)
                print("Class-wise precision\n", class_precision )
                print("Class-wise recall\n", class_recall)
                print("\n")
                
                self.dataloader.trigger_expire()
                del batch_imgs_test
                del batch_labels_test
                processed +=self.split_size
                        
            return(preds,accuracy,recall,precision, class_recall, class_precision)
                          
                          
                          
                          
                          
#####MULTILABEL RUN#######
dataloader = DataLoader(train_image_filepath)
model = Incept_ResNetV2((512, 512,3),(1,5)) 
trainer = ModelTrainer(dataloader,split_size=700)
model = trainer.fit(X, y, model, epochs = 10)


prediction , accuracy,recall,precision,class_recall, class_precision  = trainer.predict(X_test,y_test,model)


print(prediction[1])
print(y_test[1])

print("Recall ",recall)
print("Accuracy ",accuracy)

model.model.summary()

