import cv2
import numpy as np
import os
import pandas as pd 


#获取内部轮廓面积
def getArea(img):
    conts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    conts = conts[0] if len(conts) == 2 else conts[1]
    c = max(conts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    return area, c

def prebinary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 128:
                img[i][j] = 0
            else:
                img[i][j] = 255
    
    return img

def main():
    filepath = input("Please entry the filepath:")
    img = cv2.imread(filepath)
    predGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred_uterus_binary = prebinary(predGray.copy())
    _, Temp_P_binary = cv2.threshold(predGray, 129, 255, cv2.THRESH_BINARY)
    pred_tumor_binary = pred_uterus_binary - Temp_P_binary

    #接收图像最大轮廓和最大轮廓的面积
    R1, c1 = getArea(pred_uterus_binary)
    R2, c2 = getArea(pred_tumor_binary)
    ratio = R2/R1

    cv2.drawContours(img, c2, -2, (255, 0, 0), 1)
    cv2.drawContours(img, c1, -2, (255, 0, 0), 1)
    cv2.putText(img, 'R1=%.3f' %R1, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, 'R2=%.3f' %R2, (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Ratio=%.5f' % ratio, (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
