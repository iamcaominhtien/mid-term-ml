import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix,graycoprops
SIZE_IMAGE=128

def cal_entropy(glcm):
    s_glcm=glcm+glcm.T 
    n_glcm=s_glcm/np.sum(s_glcm)
    entropy_matrix=-np.log(n_glcm+1e-6)
    return np.sum(entropy_matrix*n_glcm)

img=cv2.imread('natural_images/test/airplane/airplane_0701.jpg',0)
img=cv2.resize(img,(SIZE_IMAGE,SIZE_IMAGE))
glcm=graycomatrix(img,[1],[0])
print('Kích thước ma trận hình ảnh: {}x{}'.format(img.shape[0],img.shape[1]))
df = pd.DataFrame()
df['entropy'] = [cal_entropy(glcm[:,:,0,0])]
df['contrast'] = graycoprops(glcm,'contrast')[0]
df['correlation'] = graycoprops(glcm,'correlation')[0]
df['energy'] = graycoprops(glcm,'energy')[0]
df['homogeneity'] = graycoprops(glcm,'homogeneity')[0]
print('Các giá trị đặc trưng kết cấu: ')
print(df)
