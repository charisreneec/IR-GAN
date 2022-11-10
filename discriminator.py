import torch
from torch import nn


class discriminator(nn.Module):
    
    def __init__(self,classes=12):
        super(discriminator,self).__init__()
        self.c1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c4=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c5=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c6=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.fc_source=nn.Linear(16*16*64,1)
        self.fc_class=nn.Linear(16*16*64,classes)
        self.fc_z=nn.Linear(16*16*64,32)
        self.relu = nn.LeakyReLU(0.2)
        self.sig=nn.Sigmoid()
        self.lsoft=nn.LogSoftmax()
        self.soft=nn.Softmax(dim=1)

    def forward(self,x):

        x=self.c1(x)
        #x=self.c2(x)
        x=self.c3(x)
        #x=self.c4(x)
        x=self.c5(x)
        #x=self.c6(x)
        x=torch.reshape(x,(-1,16*16*64))
        x = self.fc_class(x)
        #x = self.relu(x)
        #rf=self.sig(self.fc_source(x))#checks source of the data---i.e.--data generated(fake) or from training set(real)
        #z = self.sig(self.fc_z(x))
        c=self.lsoft(x)#checks class(label) of data--i.e. to which label the data belongs in the CIFAR10 dataset
        
        return c 