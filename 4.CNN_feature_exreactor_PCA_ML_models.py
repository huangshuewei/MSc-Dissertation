# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:36:21 2022

@author: Shu-wei Huang
"""

# CNN feature extractor + PCA + Machine Learning Models
# load data
import numpy as np
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

# load libraries
# load DenseNet169 Feature Extractor
from keras.applications.densenet import DenseNet169

# include_top=False ==> without the fully connection layers
# the output will be a set of feature images/patterns
feature_extractor = DenseNet169(weights = "imagenet", include_top=False, input_shape=(512, 128,3))

x_train_patterns = feature_extractor.predict(x_train_image)
x_test_patterns = feature_extractor.predict(x_test_image)
x_val_patterns = feature_extractor.predict(x_val_image)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(x_train_patterns[10,:,:,0], cmap="gray")

# flatten images to arrays
x_train_patterns = x_train_patterns.reshape(x_train_patterns.shape[0], -1)
x_test_patterns = x_test_patterns.reshape(x_test_patterns.shape[0], -1)
x_val_patterns = x_val_patterns.reshape(x_val_patterns.shape[0], -1)

# Reduce dimension using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1000) #
pca.fit(x_train_patterns)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")

# transfer data to PCA data
x_train_pca = pca.transform(x_train_patterns)
x_test_pca = pca.transform(x_test_patterns)
x_val_pca = pca.transform(x_val_patterns)

# Try Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
y_pred_pca = bnb.fit(x_train_pca, y_train_label).predict(x_test_pca)

# compute accuracy
from sklearn import metrics
#print ("BernoulliNB Accuracy = ", metrics.accuracy_score(y_test_label, y_pred_pca)*100, "%")

# Try KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=15)
y_pred_pca_knn = KNN.fit(x_train_pca, y_train_label).predict(x_test_pca)
#print ("KNN Accuracy = ", metrics.accuracy_score(y_test_label, y_pred_pca_knn)*100, "%")

# Try neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# define the keras ANN model
model = Sequential()
model.add(Dense(1024, input_shape=(1000,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

adam_opt = Adam(learning_rate=0.00001)
model.compile(optimizer = adam_opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#print(model.summary()) 

# Model checkpoint
import tensorflow as tf
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_pca_ann.h5', verbose=1, save_best_only=True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            checkpointer]

results = model.fit(x_train_pca, y_train_label_cat, validation_data=(x_val_pca, y_val_label_cat), batch_size=16, epochs=100, callbacks=callbacks)

# compute accuracy
y_pred_pca_ann = model.predict(x_test_pca)
#print ("ANN Accuracy = ", metrics.accuracy_score(y_test_label, np.argmax(y_pred_pca_ann, axis=1))*100, "%")

###########################################################
# GRID SEARCH to find the best model and parameters
#########################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [0.001, 0.1, 1],  #Regularization parameter. Providing only two as SVM is slow
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2,4,6,8,10,12,14],
            'n_estimators': [10,20,30,50,70,100]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'penalty': ['l1','l2','elasticnet'],
            'C': [0.001, 0.1, 1]  #Regularization. . Providing only two as LR can be slow
        }
    }
}

# Grid Search
from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    grid =  GridSearchCV(estimator=mp['model'], 
                         param_grid=mp['params'], 
                         cv=5, n_jobs=4, 
                         return_train_score=False)
    
    grid.fit(x_train_pca, y_train_label)
    
    scores.append({
        'model': model_name,
        'best_score': grid.best_score_,
        'best_params': grid.best_params_
    })

import pandas as pd    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df)

# Set these three machine learning models (SVM, RF, LR)
svc_mdl = SVC(gamma='auto', C= 0.1, kernel='linear')
svc_mdl.fit(x_train_pca, y_train_label)
pred_svc = svc_mdl.predict(x_test_pca)
#print ("SVM Accuracy = ", metrics.accuracy_score(y_test_label, pred_svc)*100, "%")

rf_mdl = RandomForestClassifier(criterion = 'entropy', max_depth=12, n_estimators=100)
rf_mdl.fit(x_train_pca, y_train_label)
pred_rf = rf_mdl.predict(x_test_pca)
#print ("RF Accuracy = ", metrics.accuracy_score(y_test_label, pred_rf)*100, "%")

lr_mdl = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', multi_class='auto')
lr_mdl.fit(x_train_pca, y_train_label)
pred_lr = lr_mdl.predict(x_test_pca)
#print ("LR Accuracy = ", metrics.accuracy_score(y_test_label, pred_lr)*100, "%")

# ensemble model
pred_ensembles = np.array([y_pred_pca, 
                           y_pred_pca_knn, 
                           np.argmax(y_pred_pca_ann, axis=1),
                           pred_svc, 
                           pred_rf, 
                           pred_lr]).T
avg_ensemble_pred = np.array([round(np.average(pred_ensembles[i])) for i in range(len(pred_ensembles))])
med_ensemble_pred = np.array([round(np.median(pred_ensembles[i])) for i in range(len(pred_ensembles))])

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in list(List):
        curr_frequency = list(List).count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

vote_ensemble_pred = np.array([most_frequent(pred_ensembles[i]) for i in range(len(pred_ensembles))])

print ("BernoulliNB Accuracy = ", metrics.accuracy_score(y_test_label, y_pred_pca)*100, "%")
print ("KNN Accuracy         = ", metrics.accuracy_score(y_test_label, y_pred_pca_knn)*100, "%")
print ("ANN Accuracy         = ", metrics.accuracy_score(y_test_label, np.argmax(y_pred_pca_ann, axis=1))*100, "%")
print ("SVM Accuracy         = ", metrics.accuracy_score(y_test_label, pred_svc)*100, "%")
print ("RF Accuracy          = ", metrics.accuracy_score(y_test_label, pred_rf)*100, "%")
print ("LR Accuracy          = ", metrics.accuracy_score(y_test_label, pred_lr)*100, "%")
print ("AVG_ensemble Accuracy  = ", metrics.accuracy_score(y_test_label, avg_ensemble_pred)*100, "%")
print ("MED_ensemble Accuracy  = ", metrics.accuracy_score(y_test_label, med_ensemble_pred)*100, "%")
print ("VOTE_ensemble Accuracy = ", metrics.accuracy_score(y_test_label, vote_ensemble_pred)*100, "%")
