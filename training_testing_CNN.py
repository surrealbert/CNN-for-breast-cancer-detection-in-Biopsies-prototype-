#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt


#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Esta función realiza el acondicionamiento necesario de la imagen ingresada, 
#en este caso ajustar el tamaño de la imagen, introducir una inversión horizontal
#aleatoria para agregar variedad en el entrenamiento, normalizar y transformar 
#a un objeto de tipo tensor
transformacion=transforms.Compose([
    transforms.Resize((50,200)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

#Dataloader

#Path for training and testing directory + epochs
train_path='C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/imagenes_IDC_pre/train'
test_path='C:/Users/luisa/Desktop/11/Diagnostico asistido por Computadora/Proyecto final/imagenes_IDC_pre/test'
num_epochs=50

#Prepare the batch of images to feed the CNN
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformacion),
    batch_size=500, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformacion),
    batch_size=500, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

#CNN structure
class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()
        
        #Input shape= (50,200,3)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=6)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)      
        #(25,100,6)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=12)
        self.relu2=nn.ReLU() 
        self.pool2=nn.MaxPool2d(kernel_size=2) 
        #(12,50,12)
        
    
        self.conv3=nn.Conv2d(in_channels=12,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=24)
        self.relu3=nn.ReLU()        
        self.pool3=nn.MaxPool2d(kernel_size=2)
        #(6,25,24)
        
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
        

model=ConvNet(num_classes=2).to(device)
#Adam optimizer with lr of 1e-4 and Cross Entropy Loss function
optimizer=Adam(model.parameters(),lr=0.0001,weight_decay=0.001)
#optimizer=SGD(model.parameters(),lr=1e-4,weight_decay=0.0001)


loss_function=nn.CrossEntropyLoss()

#Data sets size calculation through folders
train_count=len(glob.glob(train_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))
print(train_count,test_count)

#Model training and saving best model
best_accuracy=0.0
plt.figure()

val_losses = []
train_losses = []
testing_accuracy = []
training_accuracy = []

#Epoch loop
for epoch in range(num_epochs):
    
    #CNN training
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    test_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))

    #Train accuracy calculation
    train_accuracy=train_accuracy/train_count
    training_accuracy.append(train_accuracy)
    
    train_loss=train_loss/train_count
    train_losses.append(train_loss)
    
    #Evaluation on testing dataset
    model.eval()
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
           
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        loss=loss_function(outputs,labels)
        
        test_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_count
    val_losses.append(test_loss)
    testing_accuracy.append(test_accuracy)
    
    #Performance information
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+'Test Loss:'+str(test_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
    #Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'new.model')
        best_accuracy=test_accuracy
    
torch.save(model.state_dict(),'new_final.model')
print(best_accuracy)
plt.figure(figsize=(10,5))
plt.title("Training and Testing accuracy")
plt.plot(testing_accuracy,label="Test accuracy")
plt.plot(training_accuracy,label="Training accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()