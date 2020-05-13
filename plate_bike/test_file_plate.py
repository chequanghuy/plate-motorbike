import numpy as np
import cv2
import sys
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics

plate_cascade = cv2.CascadeClassifier('weight/cascade.xml')

PATH = "images"
PATH_WEIGHT = "weight"
# Vị trí của mảng tương ứng với nhãn của ký tự, ví dụ vị trí 10 là nhãn A
LABELS = ['0','1','2','3','4','5','6','7','8','9', 'A','B','C','D','E',
        'F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z']

files = [os.path.join(PATH,f) for f in os.listdir(PATH) if f.endswith(".jpg")]
for f in files:
        i=0
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 3)
        if len(plates) != 0:   
                for (x,y,w,h) in plates:
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


                roi = img[y:y+h, x:x+w]

                # cv2.imwrite("result/plate.jpg", roi)

                # Chuyển ảnh BGR sang ảnh xám 
                roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

                # cv2.imwrite("result/roi_gray.jpg", roi_gray)

                # Lọc nhiễu bằng GaussianBlur
                roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)

                # cv2.imwrite("result/roi_blur.jpg", roi_blur)

                # Dùng THRESH_BINARY_INV đưa ảnh về trắng đen
                # ret,thre = cv2.threshold(roi_blur,170,255,cv2.THRESH_BINARY_INV)
                ret, thre = cv2.threshold(roi_blur,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                # cv2.imwrite("result/roi_thre.jpg", thre)

                kernel = np.ones((3,3), np.uint8)

                '''Thuật toán dilate
                Xem ví dụ tại: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html'''
                kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
                # thre_mor = cv2.dilate(thre_mor,kernel,iterations=1)

                # Tìm tất cả các contours trên ảnh
                mask = np.zeros(img.shape, np.uint8)
                cont, hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, cont, -1, 255, -1)

                # Xác định các contours là ký tự, biển xe ô tô VN có 8 ký tự
                areas_ind = {}
                areas = []
                for ind,cnt in enumerate(cont) :
                        area = cv2.contourArea(cnt)
                        # print(area, roi_gray.shape[0] * roi_gray.shape[1])
                        if((roi_gray.shape[0] * roi_gray.shape[1]) - area < 2000):
                                continue
                        else:
                                areas_ind[area] = ind
                                areas.append(area)
                areas = sorted(areas,reverse=True)[1:9]

                # Tạo ma trận rộng có kích thước bằng kích thước ảnh đầu vào
                mask_num = np.zeros(img.shape, np.uint8)

                # Nhận diện ký tự vừa xác định
                plate = []
                for c,i in enumerate(areas):
                        x1,y1,w1,h1 = cv2.boundingRect(cont[areas_ind[i]])
                        # Vẽ contour biển số đã lọc lên ảnh và hiển thị
                        cv2.drawContours(mask_num, [cont[areas_ind[i]]], -1, 255, -1)
                                
                        char = roi[y1:y1+h1, x1:x1+w1]
                        gray = cv2.cvtColor(cv2.resize(char,(50,50)), cv2.COLOR_BGR2GRAY)
                        _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        gray = gray.reshape(1, gray.shape[0]*gray.shape[1])
                        # Load trọng số của mạng đã training
                        clf = joblib.load(PATH_WEIGHT + '/model.joblib')
                        # print(clf.predict(gray)[0])

                        # Dự đoán ký tự và đưa vào mảng plate ký tự đã dự đoán
                        plate.append(clf.predict(gray)[0])

                        # cv2.imwrite("result/" + f.split("/")[-1].split(".")[0] + "_" + str(c) + ".jpg", char)
                        cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,255,0),2)
                        startX = x+x1
                        startY = y+y1 - 15 if y+y1 - 15 > 15 else y+y1 + 15
                        cv2.putText(img, str(LABELS[plate[-1]]), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
        cv2.imwrite("result/" + f.split("\\")[-1], img)
# cv2.imshow('mask_num',mask)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()