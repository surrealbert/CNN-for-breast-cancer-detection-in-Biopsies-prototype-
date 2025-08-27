
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

import math
import matplotlib.pyplot as plt

import numpy as np
import numpy.fft as fft

import numpy as np
import cv2 as cv
import os

import pywt
import pywt.data

from PIL import Image
from PIL import ImageFilter
import os

def concat_vh(list_2d):
    
      # return final image
    return cv.vconcat([cv.hconcat(list_h) 
                        for list_h in list_2d])

  
# path of the folder containing the raw images
#inPath ="C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/BreaKHis 400X/benign/benign/"
  
# path of the folder that will contain the modified image
#outPath ="C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/imagenes_preprocesadas/benign/malignant"
  
inPath ="C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/imagenes_IDC/test/malignant"
  
# path of the folder that will contain the modified images
outPath ="C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/imagenes_IDC_pre/test/malignant"

for imagePath in os.listdir(inPath):
    # Se juntan la ruta general y el nombre de la imagen
    inputPath = os.path.join(inPath, imagePath)
    # Se carga la imagen en la ruta obtenida
    img = Image.open(inputPath)
    #Se crea la ruta de salida para el resultado procesado
    fullOutPath = os.path.join(outPath, 'proc_'+imagePath)
    
    #Se realiza el procesamiento 
    img = cv.imread(inputPath)
    img = rgb2gray(img)
    fft_image = np.fft.fftshift(np.fft.fft2(img))
    fft_image=abs(fft_image)
    
    img = (img * 255).astype(np.uint8)
    fft_image = np.log(abs(fft_image))
    fft_image = fft_image/(fft_image.max()/255.0)
    
    fft_image = fft_image.astype(np.uint8)
      
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    wave_V = cv.resize(cV, dsize=(50, 50), interpolation=cv.INTER_CUBIC)
    wave_V = wave_V.astype(np.uint8)
    
    wave_H = cv.resize(cH, dsize=(50, 50), interpolation=cv.INTER_CUBIC)
    wave_H = wave_H.astype(np.uint8)
    imagen = concat_vh([[img],
                        [wave_H],
                        [wave_V],
                        [fft_image]])
    
    #blank_image.save(fullOutPath)
    
    #img = Combine_Images_Vertically(img, fft_image)
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(imagen, cmap='gray');

    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(wave_H, cmap='gray');
    
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(wave_V, cmap='gray');
    
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(fft_image, cmap='gray');
    cv.imwrite(fullOutPath,imagen)
    print(fullOutPath)

