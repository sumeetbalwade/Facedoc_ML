#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


celebs = ['Irrfan_Khan', 'Jacqueline_Fernandez', 'Juhi_Chawla', 
          'Kajal_Aggarwal', 'Paresh_Rawal']


# In[3]:


def prepare_image(file_name):
    img_path = 'D:\Studies\PICT\T&P\Flourisense\Facedoc\MobileNet\data\celeb_faces_dataset\custom_test\\'
    img = image.load_img(img_path + file_name, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# In[15]:


image_name = 'Juhi_Chawla.57.jpg'
processed_test_image = prepare_image(image_name)


# In[16]:


Image(filename = 'D:\Studies\PICT\T&P\Flourisense\Facedoc\MobileNet\data\celeb_faces_dataset\custom_test\\' + image_name,
      width = 340, height = 437)


# In[7]:


curr_model = tf.keras.models.load_model('models\model_82.67_MobileNet.h5')


# In[8]:


curr_model.summary()


# In[9]:


def recognize(image, model):
    pred_arr = model.predict(image)
    print(pred_arr)
    sns.barplot(x = celebs, y = pred_arr[0])
    plt.xticks(rotation=90, ha='right')
    name = celebs[np.where(pred_arr[0] == pred_arr[0].max())[0][0]]
    return name


# In[17]:


name = recognize(processed_test_image, curr_model)


# In[14]:


name


# In[ ]:




