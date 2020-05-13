import cv2
import copy
import numpy as np 
import os
import random
from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics

# Đọc dữ liệu dùng để training trong thư mục chỉ định, ở đây là thư mục data
def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if (f.endswith(".png") or f.endswith(".PNG") or f.endswith(".jpg"))]
        for f in file_names:
            i=0
            img=cv2.imread(f)
            x=f.split('\\')

            cv2.imwrite("DT/"+x[1]+"/.jpg",img)
            
load_data("data")