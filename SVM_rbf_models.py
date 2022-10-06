# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:07:45 2022

@author: User
"""

import keras as K
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
import seaborn as sns

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

import numpy as np 
import pandas as pd
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import pickle

# Load data
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\angle test\ML-based methods\data\x_train_angles.npz")
x_train_angles = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\angle test\ML-based methods\data\x_test_angles.npz")
x_test_angles = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\angle test\ML-based methods\data\y_train_classes.npz")
y_train_classes = data['a']
data.close()
data = np.load(r"D:\Programming\Python\Data\Msc_project_data\angle test\ML-based methods\data\y_test_classes.npz")
y_test_classes = data['a']
data.close()


acc_tst = []
acc_tr = []
i = 0
df = pd.DataFrame(columns = ['c','gamma','train_acc','test_acc'])

for c in [1]:
    for g in [0.1]:
        clf = svm.SVC(kernel='rbf', C=c, gamma=g)
        clf.fit(x_train_angles, y_train_classes)
    
        p_tr = clf.predict(x_train_angles)
        a_tr = accuracy_score(y_train_classes, p_tr)

        p_tst = clf.predict(x_test_angles)
        a_tst = accuracy_score(y_test_classes, p_tst)

        acc_tr.append(a_tr)
        acc_tst.append(a_tst)

        df.loc[i] = [c,g,a_tr,a_tst]
        i=i+1

print("RBF kernel")
print(df)


Y_pred_classes = clf.predict(x_test_angles)

Y_pred = np_utils.to_categorical(Y_pred_classes)
y_test_label_cat = np_utils.to_categorical(y_test_classes)

# confusion matrix
confusion_mtx = confusion_matrix(y_test_classes, Y_pred_classes) 

f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(confusion_mtx, annot=True,
            linewidths=3,cmap="gray",
            fmt= '.0f',ax=ax,
            cbar = False,
           annot_kws={"size": 16})
plt.yticks(rotation = 0)
plt.xlabel("Predicted Label", size = 30)
plt.ylabel("True Label", size = 30)
plt.title("Confusion Matrix (SVM RBF kernel)", size = 30)
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
plt.title('Receiver operating characteristic to multi-class (SVM RBF kernel)')
plt.legend(loc="lower right")
plt.show()


filename = 'SVM_RBF.sav'
pickle.dump(clf, open(filename, 'wb'))



'''
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
plt.title('Zoomed Receiver operating characteristic to multi-class (SVM RBF kernel)')
plt.legend(loc="lower right")
plt.show()
'''

'''
res = []
for l in range(6):
    prec,recall,_,_ = precision_recall_fscore_support(y_test_classes==l,
                                                      Y_pred_classes==l,
                                                      pos_label=True,average=None)
    res.append([l,recall[0],recall[1]])

df = pd.DataFrame(res,columns = ['class','specificity','sensitivity'])

print(metrics.classification_report(y_test_classes, Y_pred_classes))
print("ACC: ",accuracy_score(y_test_classes, Y_pred_classes))
print(f1_score(y_test_classes, Y_pred_classes, average='macro'))
print(recall_score(y_test_classes, Y_pred_classes, average='macro'))
print(df)
'''