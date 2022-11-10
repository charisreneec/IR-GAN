#Author: Charis Cochran <crc356@drexel.edu>
#General Torch GAN Trainer (Currently setup for ACGAN Variants)

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
from IPython.display import clear_output 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import librosa
import soundfile as sf
from tqdm import tqdm, tqdm_notebook
import os
import datetime
from IPython.display import clear_output
#from utils import create_dir
from torchvision import models
from torchsummary import summary
from torchvision import models
from torchsummary import summary

class GANTrainer:
    '''
    
    '''
    def __init__(self, gen, disc, Validation_Label, 
                 loss, learning_rate, output_path, input_sizes, num_workers, batch_size, device):
        '''
        Initialize the GANTrainer
        -------------------------
        Parameters:


        '''
        #Models to Device --------------------------------------------------------
        self.Device = device
        self.disc = disc
        self.disc.to(self.Device)
        self.gen = gen
        self.gen.to(self.Device)

        #Other Globals
        self.Inp_Sizes = input_sizes
        self.Out_Path = output_path
        self.Num_Workers = num_workers
        self.Batch_Size = batch_size
        self.Learning_Rate = learning_rate

        #Label Encoder ------------------------------------------------------------
        self.le = preprocessing.LabelEncoder()
        self.le.fit(["cel","cla" ,"flu", "gac" ,"gel", "org", "pia", "sax" ,"tru" ,"vio", "voi"])

        #Set up fake lael ---------------------------------------------------------
        self.fake_label = np.ones((batch_size,))*11 # Creates a vector of labels all not instrument
        self.fake_label = ((torch.from_numpy(self.fake_label )).long()).to(self.Device)
        
        #DataLoaders --------------------------------------------------------------
        self.train_dataloader, self.val_dataloader = self.dataloaders(Validation_Label)

        #Optimizers ---------------------------------------------------------------
        self.optimD = optim.Adam(self.disc.parameters(),learning_rate)
        self.optimG = optim.Adam(self.gen.parameters(),learning_rate)

        #Loss ---------------------------------------------------------------------
        self.loss_function = loss 

        #Plot Values and Counts ---------------------------------------------------
        #Counts/Constants
        self.ep_count = 0
        self.instruments = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi','fake']

        # Loss
        self.x = [-1]
        self.train_loss_r_values = [0]
        self.train_loss_f_values = [0]
        self.train_loss_g_values = [0]
        self.val_loss_values = [0]

        self.fig_loss = plt.figure(figsize=(16,5))
        self.loss_title = "Loss After {ep:.2f} Epochs"
        self.loss_ax = self.fig_loss.add_subplot()
        self.dlr_p, = self.loss_ax.plot(self.x,self.train_loss_r_values, label = 'Disc Real Loss', color='blue')
        self.dlf_p, = self.loss_ax.plot(self.x,self.train_loss_f_values, label = 'Disc Fake Loss', color='green')
        self.gl_p, = self.loss_ax.plot(self.x,self.train_loss_g_values, label = 'Generator Loss', color='purple')
        self.vl_p, = self.loss_ax.plot(self.x,self.val_loss_values, label = 'Validation Loss', color='pink')

        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.legend(loc = 'best')
        # Accuracy
        self.train_acc_values = [0]
        self.val_acc_values = [0]

        self.fig_acc = plt.figure(figsize=(16,5))
        self.acc_title = "Accuracy After {ep:.2f} Epochs"
        self.acc_ax = self.fig_acc.add_subplot()
        self.ta_p, = self.acc_ax.plot(self.x,self.train_acc_values, color='red', label = 'Train Accuracy')
        self.va_p, = self.acc_ax.plot(self.x,self.val_acc_values, color='blue', label = 'Validation Accuracy')

        self.acc_ax.set_ylabel("Accuracy")
        self.acc_ax.legend(loc = 'best')
        # Spectrograms
        self.spec_title = "Results from Epoch {ep:.2f}"
        self.set_up_spec = True
        #Zero Loss and Accuracy ---------------------------------------------------
        self.train_loss_r = 0.0
        self.train_loss_f = 0.0
        self.train_loss_g = 0.0
        self.train_acc = 0.0

        self.val_loss = 0.0
        self.val_acc = 0.0

    #def load_checkpoint(): Secondary function

    def gen_eval_noise(self,):
        #Set up Eval Noise
        num_classes = 12
        eval_noise_size = 3*num_classes
        self.eval_noise = torch.FloatTensor(eval_noise_size, 44, 1, 1).normal_(0, 1)
        eval_noise_ = np.random.normal(0, 1, (eval_noise_size, 44))
        x=np.arange(0,12,1)
        eval_label = np.concatenate((x,x,x))
        eval_onehot = self.one_hot(eval_label,12)
        eval_noise_[np.arange(32), :12] = eval_onehot[np.arange(32)]
        eval_noise_ = (torch.from_numpy(eval_noise_))
        self.eval_noise.data.copy_(eval_noise_.view(eval_noise_size, 44, 1, 1))
        self.eval_noise=self.eval_noise.to(self.Device)


    def label_encoder(self, labels):
        labels_encoded = self.le.transform(labels)
        return labels_encoded

    def one_hot(self,a,shape):
        b = np.zeros((a.size, shape))
        b[np.arange(a.size), a] = 1
        return b

    def compute_accuracy(self, preds, labels): #Single Label Only - Based on Max Post-SoftMax
        correct = 0
        preds_ = preds.data.max(1)[1]
        correct = preds_.eq(labels.data).cpu().sum()
        acc = float(correct) / float(len(labels.data)) * 100.0
        return acc

    def weights_init(self,m): #Currently Only Normalizes Conv and BatchN
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def generate_noise_batch(self, batch_size, latent_size, num_classes, encoding_size = 12):
        noise_ = np.random.normal(0, 1, (batch_size, latent_size))
        label_ = np.random.randint(0, num_classes, batch_size)
        label_onehot = self.one_hot(label_,encoding_size)
        noise_[np.arange(batch_size), :encoding_size] = label_onehot[np.arange(batch_size),:]
        noise=((torch.from_numpy(noise_)).float()).to(self.Device)
        label=((torch.from_numpy(label_)).long()).to(self.Device)
        return noise, label 

    def experiment_setup(self, ):
        now = datetime.datetime.utcnow()+datetime.timedelta(hours = -4)
        date_time_str = now.strftime("%Y-%m-%d %I:%M%p")
        experiment_model_name = input("Enter model name: ")
        experiment_description = input("Enter a breif experiment description: ")
        experiment_name = f"{date_time_str} {experiment_model_name} : {experiment_description}"
        experiment_path = os.path.join(self.Out_Path, experiment_name)

        os.mkdir(experiment_path)

        #Save Summary of Models and Globals
        txt_path = experiment_path + "/summary.txt"
        with open(txt_path,'a') as f:
            print("Globals\n-----------------------\n", file=f)
            print(("Batch Size: ", self.Batch_Size, "\n"), file=f)
            print(("Learning Rate: ", self.Learning_Rate, "\n"), file=f)
            print(("Num Workers: ", self.Num_Workers, "\n"), file=f)
            print(("Loss: ", self.loss_function, "\n"), file=f)
            print(("OptimD: ", self.optimD, "\n"), file=f)
            print(("OptimG: ", self.optimG, "\n"), file=f)
            print("Models\n------------------------\n", file=f)
            print(("Discriminator\n", self.disc, "\n"), file=f)
            print(("Generator\n", self.gen, "\n"), file=f)
            print()
        # with open("summary.txt",'a') as f:
        #     print("Models\n--------------\n", file=f)
        #     print(("Discriminator\n", self.disc, "\n"), file=f)
        #     print(("Generator\n", self.gen, "\n"), file=f)
        #     print()
        
        return experiment_path
    
    def dataloaders(self, validation_label):
        #Load Validation Split
        X_Train = np.load('/content/drive/MyDrive/IRMAS/128x128/Split_' + validation_label + '/Train_X_overlap_0.npy')
        Y_Train = np.load('/content/drive/MyDrive/IRMAS/128x128/Split_' + validation_label + '/Train_Y_overlap0.npy')
        X_Val = np.load('/content/drive/MyDrive/IRMAS/128x128/Split_' + validation_label + '/Val_X_overlap_0.npy')
        Y_Val = np.load('/content/drive/MyDrive/IRMAS/128x128/Split_' + validation_label + '/Val_Y_overlap_0.npy')

        #Normalize Data
        for i in np.arange(0,len(X_Train),1):
            x = 2.*(X_Train[i]-np.min(X_Train[i]))/np.ptp(X_Train[i]) - 1
            X_Train[i] = x
        for i in np.arange(0,len(X_Val),1):
            x = 2.*(X_Val[i]-np.min(X_Val[i]))/np.ptp(X_Val[i]) - 1
            X_Val[i] = x

        #Encode Labels
        Y_Train_Trans = self.label_encoder(Y_Train)
        Y_Val_Trans = self.label_encoder(Y_Val)

        #Convert to Tensor
        tensor_x = ((torch.from_numpy(X_Train)).float())
        tensor_x = torch.unsqueeze(tensor_x,1)
        tensor_y = ((torch.from_numpy(Y_Train_Trans)).int())

        tensor_x_v = ((torch.from_numpy(X_Val)).float())
        tensor_x_v = torch.unsqueeze(tensor_x_v,1)
        tensor_y_v = ((torch.from_numpy(Y_Val_Trans)).int())

        irmas_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        train_dataloader = DataLoader(irmas_dataset, batch_size=self.Batch_Size,drop_last = True,shuffle = True, num_workers = self.Num_Workers) # create your dataloader

        irmas_dataset_val = TensorDataset(tensor_x_v,tensor_y_v) # create your datset
        val_dataloader = DataLoader(irmas_dataset_val, batch_size=self.Batch_Size,drop_last = True, num_workers = self.Num_Workers) # create your dataloader

        return train_dataloader, val_dataloader

    def train_epoch(self, ):
        #Set Models to Train
        self.gen.train()
        self.disc.train()

        #Zero Loss and Accuracy
        self.train_loss_r = 0.0
        self.train_loss_f = 0.0
        self.train_loss_g = 0.0
        self.train_acc = 0.0


        #Pass through full dataset once
        for i,data in tqdm_notebook(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            #Split data into labels and images
            image,label=data
            image,label=image.to(self.Device),label.to(self.Device)
            label=label.to(torch.int64)
            
            #Train Discriminator ---------------------------------------------------
            #with real data---
            self.optimD.zero_grad()
            class_= self.disc(image)
            loss_r = self.loss_function(class_,label)
            loss_r.backward()
            epoch_accuracy = self.compute_accuracy(class_[:,:-1],label)
            self.train_acc += epoch_accuracy
            self.train_loss_r += loss_r#.item()
            #with fake data---
            noise , noise_label = self.generate_noise_batch(batch_size=self.Batch_Size, latent_size=44, num_classes=12)
            #noise.to(self.Device)
            #noise_label.to(self.Device)
            noise_image = self.gen(noise)
            class_ = self.disc(noise_image.detach())
            loss_f = self.loss_function(class_,self.fake_label)
            loss_f.backward()
            self.optimD.step()
            self.train_loss_f += loss_f#.item()
            #Train Generator -------------------------------------------------------
            self.gen.zero_grad()
            class_ = self.disc(noise_image)
            loss_g = self.loss_function(class_,noise_label)
            loss_g.backward()
            self.optimG.step()
            self.train_loss_g += loss_g#.item()

        self.train_loss_r = self.train_loss_r / len(self.train_dataloader)
        self.train_loss_f = self.train_loss_f / len(self.train_dataloader)
        self.train_loss_g = self.train_loss_g / len(self.train_dataloader)
        self.train_acc = self.train_acc / len(self.train_dataloader)

        self.train_loss_r_values = np.concatenate((self.train_loss_r_values,np.asarray([float(self.train_loss_r)])))
        self.train_loss_f_values = np.concatenate((self.train_loss_f_values,np.asarray([float(self.train_loss_f)])))
        self.train_loss_g_values = np.concatenate((self.train_loss_g_values,np.asarray([float(self.train_loss_g)])))
        self.train_acc_values = np.concatenate((self.train_acc_values,np.asarray([float(self.train_acc)])))
        # self.train_loss_r_values.append(self.train_loss_r)
        # self.train_loss_f_values.append(self.train_loss_f)
        # self.train_loss_g_values.append(self.train_loss_g)
        # self.train_acc_values.append(self.train_acc)
        

    def valid_epoch(self,):
        #Set Models to Train
        self.gen.eval()
        self.disc.eval()

        #Zero the Loss and Accuracy
        self.val_loss = 0.0
        self.val_acc = 0.0

        #Test on Validation
        with torch.no_grad():
            for i, data_v in tqdm_notebook(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                image_v,label_v = data_v
                image_v,label_v=image_v.to(self.Device),label_v.to(self.Device)
                label_v=label_v.to(torch.int64)
                    
                # Forward Pass
                target_v = self.disc(image_v)
                # Find the Loss
                self.val_loss += self.loss_function(target_v,label_v)
                accuracy_val= self.compute_accuracy(target_v[:,:-1],label_v)
                self.val_acc += accuracy_val

        self.val_loss = self.val_loss / len(self.val_dataloader)
        self.val_acc = self.val_acc / len(self.val_dataloader)

        self.val_loss_values = np.concatenate((self.val_loss_values,np.asarray([float(self.val_loss)])))
        self.val_acc_values = np.concatenate((self.val_acc_values,np.asarray([float(self.val_acc)])))
        #self.val_loss_values.append(self.val_loss)
        #self.val_acc_values.append(self.val_acc)

    def plot_loss_acc_spec(self,eval_noise_size = 3*12): #Every 5 Epochs (Though Array is updated every epoch)
        #Plot Loss -------------------------------------------------------------
        self.fig_loss.suptitle(self.loss_title.format(ep = self.ep_count), fontsize=25)
        self.dlr_p.set_data(self.x,self.train_loss_r_values)
        self.dlf_p.set_data(self.x,self.train_loss_f_values)
        self.gl_p.set_data(self.x,self.train_loss_g_values)
        self.vl_p.set_data(self.x,self.val_loss_values)
        #Set new limits
        y_max_l = np.max([np.max(self.train_loss_r_values),np.max(self.train_loss_f_values),np.max(self.train_loss_g_values),np.max(self.val_loss_values)])
        y_min_l = np.min([np.min(self.train_loss_r_values),np.min(self.train_loss_f_values),np.min(self.train_loss_g_values),np.min(self.val_loss_values)])
        self.loss_ax.set_xlim(0,self.ep_count)
        self.loss_ax.set_ylim(y_min_l,y_max_l)
        self.fig_loss.savefig('%s/Loss_Plot.png' % (self.experiment_path))
        np.save((self.experiment_path + '/train_loss_r_values.npy'),self.train_loss_r_values)
        np.save((self.experiment_path + '/train_loss_f_values.npy'),self.train_loss_f_values)
        np.save((self.experiment_path + '/train_loss_g_values.npy'),self.train_loss_g_values)
        np.save((self.experiment_path + '/val_loss_values.npy'),self.val_loss_values)
        #Plot Accuracy ---------------------------------------------------------
        self.fig_acc.suptitle(self.acc_title.format(ep = self.ep_count), fontsize=25)
        self.ta_p.set_data(self.x,self.train_acc_values)
        self.va_p.set_data(self.x,self.val_acc_values)
        #Set new limits
        y_max_a = np.max([np.max(self.train_acc_values),np.max(self.val_acc_values)])
        y_min_a = np.min([np.min(self.train_acc_values),np.min(self.val_acc_values)])
        self.acc_ax.set_xlim(0,self.ep_count)
        self.acc_ax.set_ylim(y_min_a,y_max_a)
        self.fig_acc.savefig('%s/Accuracy_Plot.png' % (self.experiment_path))
        np.save((self.experiment_path + '/train_acc_values.npy'),self.train_acc_values)
        np.save((self.experiment_path + '/val_acc_values.npy'),self.val_acc_values)
        #Plot Spectrograms -----------------------------------------------------
        #Gen Images
        constructed = self.gen(self.eval_noise)
        self.constructed = constructed
        prediction_c = self.disc(constructed)
        im_to_npy = constructed.cpu()
        im_to_npy = im_to_npy.detach().numpy()
        pred_to_npy_c = prediction_c.cpu()
        pred_to_npy_c = pred_to_npy_c.detach().numpy()

        #Plot
        if self.set_up_spec == True:
            self.spec_fig, self.spec_ax = plt.subplots(3, 12)
            self.spec_fig.suptitle(self.spec_title.format(ep = self.ep_count), fontsize=25)
            self.spec_fig.set_figheight(15)
            self.spec_fig.set_figwidth(30)
            count = 0
            self.imshows = [ [] for _ in range(eval_noise_size) ]
            for p in np.arange(0,3,1):
                for k in np.arange(0,12,1):
                    self.imshows[count] = self.spec_ax[p,k].imshow(im_to_npy[count][0], interpolation='nearest',aspect='auto', origin='lower')
                    sub_title = "GT: F, {tc:.2f} ; Pred: {pd:.2f}, {pc:.2f}"
                    max = np.argmax(pred_to_npy_c[count])
                    self.spec_ax[p,k].get_xaxis().set_visible(False)
                    self.spec_ax[p,k].get_yaxis().set_visible(False)
                    count += 1
            self.spec_fig.savefig(self.check_dir + '/specs.png')
            np.save((self.check_dir + '/specs.npy'),im_to_npy)
            self.set_up_spec = False
        elif self.set_up_spec == False:
            count = 0
            self.spec_fig.suptitle(self.spec_title.format(ep = self.ep_count), fontsize=25)
            for p in np.arange(0,3,1):
                for k in np.arange(0,12,1):
                    self.imshows[count].set_data(im_to_npy[count][0])
                    sub_title = "GT: F, {tc:.2f} ; Pred: {pd:.2f}, {pc:.2f}"
                    max = np.argmax(pred_to_npy_c[count])
                    self.spec_ax[p,k].get_xaxis().set_visible(False)
                    self.spec_ax[p,k].get_yaxis().set_visible(False)
                    count += 1 
            self.spec_fig.savefig(self.check_dir + '/specs.png')
            np.save((self.check_dir + '/specs.npy'),im_to_npy)
        return
    
    def print_loss_acc(self,): #Every Epoch
        print("Epoch--[{} / {}]:".format(self.ep_count,self.epochs))
        print("Loss_Discriminator[r]--[{}], Loss_Discriminator[f]--[{}], Loss_Generator--[{}], Loss_Validation--[{}] ".format(self.train_loss_r,self.train_loss_f,self.train_loss_g,self.val_loss,))
        print("Accuracy[t]--[{}], Accuracy[v]--[{}]".format(self.train_acc,self.val_acc))

    def gen_audio(self, S = [0]): #Every 5 Epochs
        constructed_npy = self.constructed.cpu()
        os.mkdir(self.check_dir + '/audio')
        for s in S:
            for i in np.arange(0,12,1):
                inverse_mel = librosa.feature.inverse.mel_to_audio(np.asarray(constructed_npy[i][0].detach().numpy()),  sr=44100, n_fft=int(2048), hop_length=320, win_length=1024)
                path = self.check_dir + '/audio/' + self.instruments[i] +'_'+ str(s) + '.wav'
                sf.write( path, inverse_mel, samplerate = 44100)

    def save_checkpoint(self,): #Every 5 Epochs
        gen_path = self.check_dir + '/gen.pt'
        disc_path = self.check_dir + '/disc.pt'
        torch.save({"epochs_trained" : self.ep_count,
            "model_state_dict" : self.gen.state_dict(),
            "optimizer_state_dict" : self.optimG.state_dict(),
            }, gen_path)
        torch.save({"epochs_trained" : self.ep_count,
            "model_state_dict" :self. disc.state_dict(),
            "optimizer_state_dict" : self.optimD.state_dict(),
            }, disc_path)

    def train_model(self, epochs, weights_init = True, epochs_per_checkpoint = 5):
        self.epochs = epochs
        if weights_init == True:
            self.disc = self.disc.apply(self.weights_init)
            self.gen = self.gen.apply(self.weights_init)
        # create experiment name and save path
        self.experiment_path = self.experiment_setup()
        self.gen_eval_noise()

        for epoch in range(epochs):
            print(f"\nepoch: {self.ep_count}")
            self.train_epoch()
            self.valid_epoch()
            clear_output(wait=True) # clears the output cell (so that it doesn't become super long)
            self.print_loss_acc()
            self.x.append(self.ep_count)
            self.ep_count += 1
            # save checkpoint after the first epoch, then all multiples of the epochs_per_checkpoint 
            if self.ep_count == 1 or self.ep_count % epochs_per_checkpoint == 0:
                self.check_dir = self.experiment_path + '/' + str(self.ep_count)
                os.mkdir(self.check_dir)
                self.save_checkpoint()
                self.plot_loss_acc_spec()
                self.gen_audio()
