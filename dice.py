import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def dice(x,y):
    
    s2 = x #cv2.imread(x, 0)# 模板
    row, col = s2.shape[0], s2.shape[1]
    s1 = y #cv2.imread(y, 0)
    s = []
    for r in range(row - 10):
        for c in range(col - 10):
            if s1[r][c] == s2[r][c]: # 计算图像像素交集
                s.append(s1[r][c])
#                 print(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    return 2*m1/m2

def prebinary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 128:
                img[i][j] = 0
            else:
                img[i][j] = 255
    
    return img

def main():
    pathlabel = r'Label/'
    pathpred = r'Predictions/'
    filesname = os.listdir(pathlabel)
    diceuterus = []
    dicetumor = []
    for filename in tqdm(filesname):

        label = cv2.imread(os.path.join(pathlabel, filename))
        pred = cv2.imread(os.path.join(pathpred, filename))
        labelGray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        predGray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        _, label_uterus_binary = cv2.threshold(labelGray, 1, 255, cv2.THRESH_BINARY)
        pred_uterus_binary = prebinary(predGray.copy())
        _, Temp_L_binary = cv2.threshold(labelGray, 40, 255, cv2.THRESH_BINARY)
        _, Temp_P_binary = cv2.threshold(predGray, 129, 255, cv2.THRESH_BINARY)
        label_tumor_binary = label_uterus_binary - Temp_L_binary
        pred_tumor_binary = pred_uterus_binary - Temp_P_binary
        Uterus_dice = dice(label_uterus_binary, pred_uterus_binary)
        Tumor_dice = dice(label_tumor_binary, pred_tumor_binary)

        diceuterus.append(Uterus_dice)
        dicetumor.append(Tumor_dice)

    # print('diceuterus:', diceuterus)
    # print('dicetumor:', dicetumor)
    df = pd.DataFrame({ 'diceuterus': diceuterus,
                        'dicetuomor': dicetumor}, index=filesname)
    
    df.to_csv('dice.csv')

if __name__ == '__main__':
    
    main()

