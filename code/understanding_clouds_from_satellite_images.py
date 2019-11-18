#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages
import os, glob

import random
import numpy as np
import pandas as pd

import cv2
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion, CenterCrop

import multiprocessing
from copy import deepcopy
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

import matplotlib.pyplot as plt
from IPython.display import Image


# In[2]:


# Setting run specfic values

num_cores = multiprocessing.cpu_count()

debug = True if os.environ.get('debug') == 'True' else False

random_seed = int(os.environ.get('random_seed'))
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

get_ipython().run_line_magic('matplotlib', 'inline')

test_imgs_folder = "../data/test_images/"
train_imgs_folder = "../data/train_images/"


# # Data Generators

# In[3]:


if debug:
    train_df = pd.read_csv("../data/train.csv", nrows=160)
else:
    train_df = pd.read_csv("../data/train.csv")
    
print(train_df.info())
train_df.head()


# In[4]:


if debug:
    sample_df = pd.read_csv("../data/sample_submission.csv", nrows=40)
else:
    sample_df = pd.read_csv("../data/sample_submission.csv")
print(sample_df.info())
sample_df.head()


# In[5]:


train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['Class'].unique()
train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
train_df.head()


# In[6]:


# dictionary for fast access to ohe vectors
img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}


# In[7]:


# sorting present classes in lexicographical order, just to be sure
if debug:
    stratify = None
else:
    stratify = train_df['Class'].map(lambda x: str(sorted(list(x)))),

train_imgs, val_imgs = train_test_split(train_df['Image'].values, 
                                        test_size=0.2, 
                                        stratify=stratify,
                                        random_state=random_seed)


# In[8]:


class DataGenenerator(Sequence):
    def __init__(self, images_list=None, folder_imgs=train_imgs_folder, batch_size=32, shuffle=True, 
                 augmentation=None, resized_height=260, resized_width=260, num_channels=3, num_classes=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(folder_imgs)
        else:
            self.images_list = deepcopy(images_list)
        self.folder_imgs = folder_imgs
        self.len = len(self.images_list) // self.batch_size
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.is_test = not 'train' in folder_imgs
        if not shuffle and not self.is_test:
            self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len*self.batch_size]]

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.images_list)

    def __getitem__(self, idx):
        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
        y = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.folder_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img/255.0
            if not self.is_test:
                y[i, :] = img_2_ohe_vector[image_name]
        return X, y

    def get_labels(self):
        if self.shuffle:
            images_current = self.images_list[:self.len*self.batch_size]
            labels = [img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)


# In[9]:


albumentations_train = Compose([VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()], p=1)


# In[10]:


data_generator_train = DataGenenerator(images_list=train_imgs, augmentation=albumentations_train, batch_size=2)
data_generator_train_eval = DataGenenerator(images_list=train_imgs, shuffle=False, batch_size=2)
data_generator_val = DataGenenerator(images_list=val_imgs, shuffle=False, batch_size=2)


# ## PR-AUC-based Callback

# In[11]:


class PrAucCallback(Callback):
    def __init__(self, data_generator, num_workers=num_cores, 
                 early_stopping_patience=5, 
                 plateau_patience=3, reduction_rate=0.5,
                 stage='train', checkpoints_path='checkpoints/'):
        super(Callback, self).__init__()
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names) + 1)] # to store per each class and also mean PR AUC
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path
        
    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'#'*30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc/len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)        
        print(f"\n{'#'*20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'#'*20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean
              
    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]    
              
    def early_stopping_check(self, pr_auc_mean):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True    
              
    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            # remove previous checkpoints to save space
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_epoch_*')):
                os.remove(checkpoint)
        self.best_pr_auc = pr_auc_mean
        self.model.save(os.path.join(self.checkpoints_path, f'classifier_epoch_{epoch}_val_pr_auc_{pr_auc_mean}.h5'))              
        print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
              
    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.data_generator, workers=self.num_workers)
        y_true = self.data_generator.get_labels()
        # estimate AUC under precision recall curve for each class
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)
              
        if self.stage == 'val':
            # early stop after early_stopping_patience=4 epochs of no improvement in mean PR AUC
            self.early_stopping_check(pr_auc_mean)

            # save a model with the best PR AUC in validation
            self.model_checkpoint(pr_auc_mean, epoch)

            # reduce learning rate on PR AUC plateau
            self.reduce_lr_on_plateau()            
        
    def get_pr_auc_history(self):
        return self.history


# In[12]:


train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')


# # Classifier

# In[13]:


def get_model():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)

model = get_model()


# In[ ]:


if debug:
    num_epochs = 2
else:
    num_epochs = 20

for base_layer in model.layers[:-3]:
    base_layer.trainable = False
    
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy')
history_0 = model.fit_generator(generator=data_generator_train,
                                validation_data=data_generator_val,
                                epochs=num_epochs,
                                callbacks=[train_metric_callback, val_callback],
                                workers=num_cores,
                                verbose=1)


# ## Fine-tuning the whole model

# In[ ]:


if debug:
    num_epochs = 2
    initial_epoch = 1
else:
    num_epochs = 20
    initial_epoch = 3
    
for base_layer in model.layers[:-3]:
    base_layer.trainable = True
    
model.compile(optimizer=Adam(lr=5e-6), loss='binary_crossentropy')
history_1 = model.fit_generator(generator=data_generator_train,
                              validation_data=data_generator_val,
                              epochs=num_epochs,
                              callbacks=[train_metric_callback, val_callback],
                              workers=num_cores,
                              verbose=1,
                              initial_epoch=3
                             )


# In[ ]:


def plot_with_dots(ax, np_array):
    ax.scatter(list(range(1, len(np_array) + 1)), np_array, s=50)
    ax.plot(list(range(1, len(np_array) + 1)), np_array)


# In[ ]:


pr_auc_history_train = train_metric_callback.get_pr_auc_history()
pr_auc_history_val = val_callback.get_pr_auc_history()

plt.figure(figsize=(10, 7))
plot_with_dots(plt, pr_auc_history_train[-1])
plot_with_dots(plt, pr_auc_history_val[-1])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Mean PR AUC', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation PR AUC', fontsize=20)
plt.savefig('pr_auc_hist.png')


# In[ ]:


plt.figure(figsize=(10, 7))
plot_with_dots(plt, history_0.history['loss']+history_1.history['loss'])
plot_with_dots(plt, history_0.history['val_loss']+history_1.history['val_loss'])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Binary Crossentropy', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation Loss', fontsize=20)
plt.savefig('loss_hist.png')


# In[ ]:


model = load_model('../input/clouds-classifier-files/classifier_epoch_45_val_pr_auc_0.8344173287108075.h5')


# In[ ]:


Image("../input/clouds-classifier-files/loss_hist.png")


# In[ ]:


Image("../input/clouds-classifier-files/pr_auc_hist.png")

