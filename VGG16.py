# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:13:19 2022

@author: User
"""

from keras.applications.vgg16 import VGG16
import keras as K
from keras import initializers
import pandas as pd

from keras.layers import Flatten, Dropout, Dense
from keras import models
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

import numpy as np 
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# Load data

data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\x_train_image.npz")
x_train_image = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\x_test_image.npz")
x_test_image = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\x_val_image.npz")
x_val_image = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_train_label.npz")
y_train_label = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_test_label.npz")
y_test_label = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_val_label.npz")
y_val_label = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_train_label_cat.npz")
y_train_label_cat = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_test_label_cat.npz")
y_test_label_cat = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\CNN model compare\data\y_val_label_cat.npz")
y_val_label_cat = data['a']
data.close()

# show loaded data
plt.figure(figsize=(20, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train_image[i])
    plt.title("Label = " + str(y_train_label[i]+1), size = 16)
plt.show()

# VGG16
vgg16_base_model = VGG16(weights = "imagenet", include_top=False, input_shape=(512, 128,3))

vgg16_model = models.Sequential()

vgg16_model.add(vgg16_base_model)
vgg16_model.add(Flatten())
vgg16_model.add(Dense(512,
                      kernel_initializer=initializers.random_normal(stddev=0.01),
                      bias_initializer=initializers.Zeros(),
                      activation='relu'))
vgg16_model.add(Dropout(0.5))
vgg16_model.add(Dense(512,
                      kernel_initializer=initializers.random_normal(stddev=0.01),
                      bias_initializer=initializers.Zeros(),
                      activation='relu'))
vgg16_model.add(Dropout(0.5))
vgg16_model.add(Dense(6,
                      kernel_initializer=initializers.random_normal(stddev=0.01),
                      bias_initializer=initializers.Zeros(),
                      activation='softmax'))
vgg16_model.summary()



# compile both models
optimizer = K.optimizers.Adam(learning_rate=0.00001)
vgg16_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

# Training model
batch_size = 16
history_vgg16 = vgg16_model.fit(x_train_image, 
                          y_train_label_cat, 
                          verbose=1, 
                          epochs=10,
                          batch_size=batch_size,
                          validation_data = (x_val_image, y_val_label_cat))



# Evaluating model performance
loss_vgg16, acc_vgg16 = vgg16_model.evaluate(x_test_image, 
                           y_test_label_cat, 
                           batch_size=batch_size)


print('\n\nVGG16\n\nLoss     : {} \nAccuracy : {}'.format(history_vgg16.history['loss'][-1],history_vgg16.history['accuracy'][-1]))
print('\n\nVGG16\n\nLoss     : {} \nAccuracy : {}'.format(loss_vgg16,acc_vgg16))


vgg16_model.save("CNN_vgg16.h5")

# Test data vgg16
Y_pred = vgg16_model.predict(x_test_image)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.array(y_test_label)
rows = 6
cols = 10
shift = 256
f = plt.figure(figsize=(2*cols,2*rows))
print("Validation Predictions")
for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1)
    img = x_test_image[i+shift]
    img = img.reshape((512, 128,3))
    plt.imshow(img[:,:,0],
               cmap='gray')
    plt.axis("off")
    if Y_pred_classes[i+shift] != Y_true[i+shift]:
        plt.title("Prediction: {}\nTrue Value: {}".format(Y_pred_classes[i+shift], Y_true[i+shift]),
                  y=-0.35,color="red")
    else:
        plt.title("Prediction: {}\nTrue Value: {}".format(Y_pred_classes[i+shift], Y_true[i+shift]),
                  y=-0.35,color="green")
    
f.tight_layout()
f.show()

# confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(confusion_mtx, annot=True,
            linewidths=3,cmap="gray",
            fmt= '.0f',ax=ax,
            cbar = False,
           annot_kws={"size": 16})
plt.yticks(rotation = 0)
plt.xlabel("Predicted Label", size = 30)
plt.ylabel("True Label", size = 30)
plt.title("Confusion Matrix", size = 30)
plt.show()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Plot linewidth.
lw = 2

n_classes = 6
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label_cat[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class (VGG16)')
plt.legend(loc="lower right")
plt.show()

# Zoom in view of the upper left corner.
plt.figure()
plt.xlim(-0.002, 0.202)
plt.ylim(0.8, 1.002)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoomed Receiver operating characteristic to multi-class (VGG16)')
plt.legend(loc="lower right")
plt.show()



res = []
for l in range(6):
    prec,recall,_,_ = precision_recall_fscore_support(y_test_label==l,
                                                      Y_pred_classes==l,
                                                      pos_label=True,average=None)
    res.append([l,recall[0],recall[1]])

df = pd.DataFrame(res,columns = ['class','specificity','sensitivity'])

print(metrics.classification_report(y_test_label, Y_pred_classes))
print("ACC: ",accuracy_score(y_test_label, Y_pred_classes))
print(f1_score(y_test_label, Y_pred_classes, average='macro'))
print(recall_score(y_test_label, Y_pred_classes, average='macro'))
print(df)