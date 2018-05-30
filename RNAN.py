import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime
import sys
import time



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 2, (1, 256), stride = 1, padding = 0, bias = True),
            nn.Conv2d(2, 2, (24), stride = 1, padding = 0, bias = True),
            nn.Conv2d(2, 2, (1, 128), stride = 1, padding = 0, bias = True),
            nn.Conv2d(2, 2, (7), stride = 1, padding = 0, bias = True),
        )
        
        self.linear_first = nn.Sequential( 
            nn.Linear(204+20, 20),
            nn.ReLU(),
            nn.Linear(20, 204),
            nn.ReLU()
        )
        
        self.rnn = nn.Sequential(
            nn.Linear(204+20, 60),
            nn.ReLU(),
            nn.Linear(60, 20),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 2, (7), stride = 1, padding = 0, bias = True),
            nn.ConvTranspose2d(2, 2, (1, 128), stride = 1, padding = 0, bias = True),
            nn.ConvTranspose2d(2, 2, (24), stride = 1, padding = 0, bias = True),
            nn.ConvTranspose2d(2, 2, (1, 256), stride = 1, padding = 0, bias = True),
        )
        
        
    def forward(self, x):

        encode = self.encoder(x)
        print (encode.shape)
        out = encode.view(4, -1)
        
        for i in range(x.shape[0]):
            
            if i < 1:
                first_layer = torch.cat((out[i,:], torch.zeros([20])),0)
                
            else:
                first_layer = torch.cat((out[i,:], time_context),0)
            
            linear_out  = self.linear_first(first_layer)
            
            time_context = self.rnn(first_layer)   # 20 units
            
            linear_out = linear_out.view(-1, 2, 1, 102)
            
            try:
                print (i)
                decoded = torch.cat((decoded,self.decoder(linear_out)),0)
            except NameError:
                decoded = self.decoder(linear_out)
                
        return decoded
              



def GenerateRandomData(seed = 2451, dimension = [30, 513]):
    ''' 
        Creates random data with a standard size of 128 * 128
        Generates a float Tensor object with dimensions 1 * 1 * height * width
    '''
    return np.random.rand(4, 2, 30, 513)
    
def trainNetwork():

    denoiser = Encoder()
    '''
    autoencoder_audio = AutoEncoder().cuda()
    autoencoder_audio.load_state_dict(torch.load(config.log_dir+dataset+'_'+str(epoch)+'.pt'))
    '''
    
    optimizer   =  torch.optim.Adagrad(denoiser.parameters(), 0.0001 )

    #loss_func   =  nn.MSELoss( size_average=False )
    loss_func   =  nn.L1Loss( size_average=False )

    inputs = Variable(torch.FloatTensor(GenerateRandomData()))

    targets = Variable(torch.FloatTensor(GenerateRandomData()))

    optimizer.zero_grad()

    out = denoiser(inputs)
    print (out.shape)
    step_loss = loss_func(out, targets)
    
    step_loss.backward()

    optimizer.step()

if __name__ == "__main__":
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        if len(sys.argv)<3:
            print("Please give a dataset to load")
        else:
            dataset = sys.argv[2]
            
            print("Training")
            trainNetwork()


