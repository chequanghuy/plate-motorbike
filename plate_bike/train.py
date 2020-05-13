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
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if (f.endswith(".png") or f.endswith(".PNG"))]
        for f in file_names:
            gray = cv2.cvtColor(cv2.resize(cv2.imread(f),(50,50)), cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            images.append(np.array(gray.reshape(gray.shape[0]*gray.shape[1])))
            labels.append(int(d))
    return np.array(images), np.array(labels)

# Lưu trọng số đã được huấn luyện vào thư mục chỉ định, ở đây là thư mục weight
def save_model(model, name, db):
    if not os.path.exists(db):
        os.makedirs(db)
    os.chdir(db)
    file_name = name + ".joblib"
    print("[+] Saving model to file : " ,file_name)
    joblib.dump(model, file_name)

# Phân lớp bằng thuật toán SVM để huấn luyện cho mô hình nhận diện ký tự 
def linear_svm(X_train, y_train, X_test, y_test):
    print("[!] SVM data...")
    clf = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("[+] Finished")
    return clf

ROOT_PATH = "data"
PATH_WEIGHT = "weight"

train_data_dir = os.path.join(ROOT_PATH)
# Doc du lieu
X_train, y_train = load_data(train_data_dir)
# Huan luyen mo hinh
clf = linear_svm(X_train, y_train, X_train, y_train)
# Luu trong so da huan luyen
save_model(clf, "model", PATH_WEIGHT)
    