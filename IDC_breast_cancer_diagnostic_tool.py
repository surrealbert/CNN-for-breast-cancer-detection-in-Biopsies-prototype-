#Tkinter libraries
from tkinter import *
from tkinter import filedialog

#Data and ploting support
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#Image manipulation library
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import transform

#PIL 
from PIL import Image, ImageTk
from PIL import ImageFilter

#Wavelet libraries
import pywt
import pywt.data

#CNN libraries from Pytorch
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open

import cv2 as cv

#Transformación de la imagen elegida para la CNN
#Image transformation to fit the first input layer
transformer=transforms.Compose([
    transforms.Resize((50,200)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])
#==============================================================================

#Declaración de las clases para clasificación
#Declare your classification clases here (binary):
classes = ['benign','malignant']

#Funciones del menú
#Menu functions
def about():
    win = Toplevel()
    win.wm_title("About")
    
    l = Label(win, text="University: Instituto Tecnologico de Tijuana\nProfessor: Arturo Sotelo Orozco\nCourse: Computer aided diagnosis\n\nMade by: Alberto Beltran Garcia\nContact: alberto.beltran16@tectijuana.edu.mx\n")
    l.grid(row=0, column=0)
    
    b = ttk.Button(win, text="Close", command=win.destroy)
    b.grid(row=1, column=0)

def instructions():
    win = Toplevel()
    win.wm_title("How to use")
    
    l = Label(win, text="Select File -> Load and choose the image to produce a diagnosis (malignant or benign).\nThe results will be shown on screen. ")
    l.grid(row=0, column=0)
    
    b = ttk.Button(win, text="Close", command=win.destroy)
    b.grid(row=1, column=0)

#Funciones de procesamiento de imagen
#Image processing functions
def concat_vh(list_2d):
    return cv.vconcat([cv.hconcat(list_h) 
                        for list_h in list_2d])

def Img_format(img_path, img_size):
    img = cv.imread(img_path)
    img = rgb2gray(img)
    img = (img * 255).astype(np.uint8)
    img = cv.resize(img, dsize=(img_size, img_size), interpolation=cv.INTER_CUBIC)
    
    return img

def HAAR_H(img, img_size):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    wave_H = cv.resize(cH, dsize=(img_size, img_size), interpolation=cv.INTER_CUBIC)
    wave_H = wave_H.astype(np.uint8)
    return wave_H

def HAAR_V(img, img_size):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    wave_V = cv.resize(cV, dsize=(img_size, img_size), interpolation=cv.INTER_CUBIC)
    wave_V = wave_V.astype(np.uint8)
    return wave_V

def FFT(img, img_size): 
    fft_image = np.fft.fftshift(np.fft.fft2(img))
    fft_image=abs(fft_image)
    fft_image = np.log(abs(fft_image))
    fft_image = fft_image/(fft_image.max()/255.0)
    fft_image = cv.resize(fft_image, dsize=(img_size, img_size), interpolation=cv.INTER_CUBIC)

    fft_image = fft_image.astype(np.uint8)
    return fft_image

def empty():
    return 0

def browsefunc():
    filename = filedialog.askopenfilename(filetypes=[('image files', '.png', '.jpg')])
    pathlabel.config(text='Filepath: \n'+filename)
    img_size = 200
    
    image = Img_format(filename,img_size)
    Haar_H = HAAR_H(image, img_size)
    Haar_V = HAAR_V(image, img_size)
    FFT_image =  FFT(image, img_size)
    
    CNN_input = concat_vh([[image],
                    [Haar_H],
                    [Haar_V],
                    [FFT_image]])

    img1 =  ImageTk.PhotoImage(Image.fromarray(image))
    label_photo = Label(root, image=img1).grid(pady=5, row=0, column=0) 
    Lower_left = Label(root, text="Original Image",font=('Arial', 12, 'bold')).grid(pady=5, row=1, column=0) 
    
    img2 =  ImageTk.PhotoImage(Image.fromarray(Haar_H))
    label_photo = Label(root, image=img2).grid(pady=5, row=0, column=1) 
    Lower_left2 = Label(root, text="Horizontal Haar Wvlt", font=('Arial', 12, 'bold')).grid(pady=5, row=1, column=1) 
   
    img3 =  ImageTk.PhotoImage(Image.fromarray(Haar_V))
    label_photo = Label(root, image=img3).grid(pady=5, row=2, column=0) 
    Lower_left = Label(root, text="Vertical Haar Wvlt", font=('Arial', 12, 'bold')).grid(pady=5, row=3, column=0) 
     
    img4 =  ImageTk.PhotoImage(Image.fromarray(FFT_image))
    label_photo = Label(root, image=img4).grid(pady=5, row=2, column=1) 
    Lower_left = Label(root, text="Fast Fourier transform", font=('Arial', 12, 'bold')).grid(pady=5, row=3, column=1) 

    pred = prediction(CNN_input,transformer)

    if (pred=='benign'):
        prediction_label.config(text='Diagnosis: '+pred, font=('Arial', 12, 'bold'),fg="green")
    else: 
        prediction_label.config(text='Diagnosis: '+pred, font=('Arial', 12, 'bold'),fg="red")

    return filepath

#Función de predicción
#Prediction function
def prediction(image,transformer):
    
    image = Image.fromarray(np.uint8(cm.gist_earth(image)*255))
    image = image.convert('RGB')
    image_tensor=transformer(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    input=Variable(image_tensor)
    
    model.eval()
    output=model(input)
    index=output.data.numpy().argmax()
    pred=classes[index]
    
    return pred

#==============================================================================
#Estructura de la red
#Network structure
class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=6)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)      
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=12)
        self.relu2=nn.ReLU() 
        self.pool2=nn.MaxPool2d(kernel_size=2) 
        
    
        self.conv3=nn.Conv2d(in_channels=12,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=24)
        self.relu3=nn.ReLU()        
        self.pool3=nn.MaxPool2d(kernel_size=2)
        
        self.hidden=nn.Linear(in_features=6 * 25 * 24,out_features=256)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(256, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.hidden = nn.Linear(500, 256)
        #self.out = nn.Linear(256, out_features=num_classes)
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool1(output)
            
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu2(output)
        output=self.pool2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=self.pool3(output)
            
        output=output.view(-1,6 * 25 * 24)
        output=self.hidden(output)
        output=self.sigmoid(output)
        output=self.out(output)
        output=self.softmax(output)
            
        return output
    
#==============================================================================
#Carga el modelo indicado
#Loads a particular trained model
modelPath = 'C:/Users/DELL/Desktop/ITT/Semestres pasados/11/Diagnostico asistido por Computadora/Proyecto final/Modelos/train_87_test_82_softmax.model' #Enter your filepath here
checkpoint=torch.load(modelPath)
model=ConvNet(num_classes=2)
model.load_state_dict(checkpoint)

#==============================================================================
#Interfaz de Usuario Tkinter
#Tkinter UI
root = Tk()
root.title('Breast Cancer Diagnostic Tool')
root.geometry("420x650")

# =============================================================================
# Menu
menu = Menu(root)
root.config(menu=menu)
#root.iconbitmap('itt.ico')

filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Load", command=browsefunc)
#filemenu.add_command(label="Exit", command=root.destroy)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=about)
helpmenu.add_command(label="Instrucions...", command=instructions)

#Espacio para gráficos y resultados
#Images and results
frame = Frame(root)
frame.grid(pady=5, row=4, column=0)
pathlabel = Label(frame,wraplength=150)
pathlabel.pack()

frame2 = Frame(root)
frame2.grid(pady=5, row=4, column=1)
prediction_label = Label(frame2)
prediction_label.pack()

root.mainloop()
