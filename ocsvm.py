import pandas as pd
import numpy as np

from utils.data_utils import *
from utils.perf_utils import *
from utils.plot_utils import *

from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM

class OCSVMIDS():
    def __init__(self):
        self.clf=OneClassSVM(gamma='auto')

    def fit(self,x_train):
        self.clf.fit(x_train)

    def pred(self,x):
        return self.clf.predict(x)


if __name__ == "__main__":
    kdd_path='../kdd_data'
    x_train,y_train=get_hdf5_data(kdd_path+'/processed/train_10.hdf5',labeled=True)
    x_train,x_val,y_train,y_val=split_data(x_train,y_train)
    x_train,y_train=filter_atk(x_train,y_train)

    x_test,y_test=get_hdf5_data(kdd_path+'/processed/test_10.hdf5',labeled=True)

    print("Train {} Val {} Test {}".format(x_train.shape[0],x_val.shape[0],x_test.shape[0]))

    trainer=OCSVMIDS()
    trainer.fit(x_train)

    #Evaluate Test Data
    y_pred= trainer.pred(x_test)
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    print(classification_report(y_test, y_pred))
