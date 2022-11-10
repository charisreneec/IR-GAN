import torch
from torch import nn

class generator(nn.Module):
    #generator model
    def __init__(self,in_channels):
        super(generator,self).__init__()
        self.fc1=nn.Linear(in_channels,1024)

        self.t000=nn.Sequential(
            nn.ConvTranspose2d(in_channels=1,out_channels=512,kernel_size=(2,2),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.t00=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(2,2),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


        self.t0=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.t1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.t2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(4,4),stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.t3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.t4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )

        
    
    def forward(self,x):
      #x=x.view(-1,44)
      x = torch.reshape(x,(-1,44))
      x=self.fc1(x)
      #x=x.view(-1,128,1,1)
      x = torch.reshape(x,(-1,1,32,32))
      x = self.t000(x)
      x = self.t00(x)
      x = self.t0(x)
      x=self.t1(x)
      x=self.t2(x)
      x=self.t3(x)
      x=self.t4(x)
      
      return x #output of generator