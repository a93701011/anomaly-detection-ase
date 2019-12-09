# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:38:19 2019

@author: chiku
"""

import pandas as pd
import numpy as np
import sys

#manual run
file = 'data_1123_30'
#input parameter
#file = str(sys.argv[1])

data = pd.read_csv('./{}.csv'.format(file))
datadrop = data.drop(columns = ['index','MCID','BONDHEAD','datetime','RECIPE'])

datadrop = datadrop.drop(columns = ['bhz_1_max','bhz_1_min','bhz_2_max','bhz_2_min','bhz_3_max','bhz_3_min'])
#data_0917_drop = data_0917_drop.drop(columns = ['bhz_1_max','bhz_1_min','bhz_2_max','bhz_2_min','bhz_3_max','bhz_3_min'])

datadrop = datadrop[datadrop['temp1']>194]
datadrop = datadrop[datadrop['force_3_max']<10]

from sklearn.model_selection import train_test_split
#data.drop(columns=['index'], inplace = True)
X_train, X_test = train_test_split( datadrop, test_size=0.05, random_state=42)

print(X_train.shape)
print(X_test.shape)
#import pickle
import joblib
from sklearn.svm import OneClassSVM
#from sklearn.model_selection import GridSearchCV, ParameterGrid

eplison =0.001
gamma = 0.0001
nu = 0.001
one_svm_rbf= OneClassSVM(nu=nu , kernel='rbf', gamma = gamma, tol = eplison)
one_svm_rbf.fit(X_train)
what_kernel = 'rbf'

print('testing data------------------------------------------------')
Y_result_rbf = one_svm_rbf.predict(X_test)
Y_scroe_rbf = one_svm_rbf.score_samples(X_test) 
print('test data size :{}'.format(X_test.shape[0]))
print('test data anomaly : {}'.format(np.sum(Y_result_rbf==-1)))
print('rbf:{}'.format(np.sum(Y_result_rbf==-1)/len(Y_result_rbf)*100))

print('traning data------------------------------------------------')

Y_result_rbf_t = one_svm_rbf.predict(X_train)
Y_scroe_rbf_t = one_svm_rbf.score_samples(X_train) 
print('train data size :{}'.format(X_train.shape[0]))
print('train data anomaly : {}'.format(np.sum(Y_result_rbf_t==-1)))
print('rbf:{}'.format(np.sum(Y_result_rbf_t==-1)/len(Y_result_rbf_t)*100))

print('all data------------------------------------------------')
eplison =0.001
gamma = 0.0001
nu = 0.001
one_svm_rbf= OneClassSVM(nu=nu , kernel='rbf', gamma = gamma, tol = eplison)
one_svm_rbf.fit(datadrop)
what_kernel = 'rbf'

Y_result_rbf_t = one_svm_rbf.predict(datadrop)
Y_scroe_rbf_t = one_svm_rbf.score_samples(datadrop) 
print('train data size :{}'.format(datadrop.shape[0]))
print('train data anomaly : {}'.format(np.sum(Y_result_rbf_t==-1)))
print('rbf:{}'.format(np.sum(Y_result_rbf_t==-1)/len(Y_result_rbf_t)*100))
print('good score:{}'.format(np.average(Y_scroe_rbf_t[Y_result_rbf_t==1])))
print('bad score:{}'.format(np.average(Y_scroe_rbf_t[Y_result_rbf_t==-1])))


print('data info------------------------------------------------')

good_data = datadrop[Y_result_rbf_t == 1]
bad_data = datadrop[Y_result_rbf_t == -1]
good_data.describe().reset_index().to_csv('./{}_good.csv'.format(file) , index = False)
bad_data.describe().reset_index().to_csv('./{}_bad.csv'.format(file) , index = False)

joblib.dump(one_svm_rbf, './OneClassSVM{}.pkl'.format(file))
X_test.columns