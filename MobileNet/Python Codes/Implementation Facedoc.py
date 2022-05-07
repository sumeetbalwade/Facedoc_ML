#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
from IPython.display import Image
from IPython.display import clear_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


celebs = ['Irrfan_Khan', 'Jacqueline_Fernandez', 'Juhi_Chawla', 
          'Kajal_Aggarwal', 'Paresh_Rawal']


# In[3]:


os.chdir('data/celeb_faces_dataset')
if os.path.isdir('train/Irrfan_Khan/') is False: 
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for c in celebs:
        shutil.move(f'{c}', 'train')
        os.mkdir(f'valid/{c}')
        os.mkdir(f'test/{c}')

        valid_samples = random.sample(os.listdir(f'train/{c}'), 30)
        for i in valid_samples:
            shutil.move(f'train/{c}/{i}', f'valid/{c}')

        test_samples = random.sample(os.listdir(f'train/{c}'), 5)
        for j in test_samples:
            shutil.move(f'train/{c}/{j}', f'test/{c}')
os.chdir('../..')


# In[4]:


train_path = 'data/celeb_faces_dataset/train'
valid_path = 'data/celeb_faces_dataset/valid'
test_path = 'data/celeb_faces_dataset/test'


# In[5]:


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)                .flow_from_directory(directory=train_path, target_size=(224,224), batch_size=5)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)                .flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=5)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)               .flow_from_directory(directory=test_path, target_size=(224,224), batch_size=5, shuffle=False)


# In[6]:


mobile = tf.keras.applications.mobilenet.MobileNet(weights = 'imagenet', include_top = False)
for layer in mobile.layers[:-10]:
    layer.trainable = False
mobile.summary()


# In[7]:


top_model = mobile.output


# In[8]:


top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dense(512, activation='relu')(top_model)


# In[9]:


top_model = Dense(units=5, activation='softmax')(top_model)


# In[10]:


model = Model(inputs=mobile.input, outputs=top_model)


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


# In[14]:


callbacks_list = [PlotLearning()]


# In[17]:


model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=20,
            verbose=2,
            callbacks=callbacks_list
)


# In[18]:


test_labels = test_batches.classes


# In[19]:


predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)


# In[20]:


predictions


# In[21]:


cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))


# In[22]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[23]:


test_batches.class_indices


# In[24]:


cm_plot_labels = celebs[:]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# In[25]:


#model.save('models\model_3_82.67_MobileNet.h5')


# In[ ]:




