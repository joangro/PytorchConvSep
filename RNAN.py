import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime
import sys
import time
from PytorchConvSep import AutoEncoder#, loss_calc
from data_pipeline import data_gen
import config

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 2, (1, 386), stride = 1, padding = 0, bias = True),
        )
        
        self.rnn = nn.LSTM(256, 256,2, batch_first= True)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 2, (1, 386), stride = 1, padding = 0, bias = True),

        )
        
    def forward(self, x):

        encode = self.encoder(x)
        encode = torch.squeeze(encode.view(4,30, -1))

        lstm,_ = self.rnn(encode)
        
        to_decode = lstm.contiguous().view(4, 2, 30, 128)

        decoded = self.decoder(to_decode)
        return decoded
              



def GenerateRandomData(seed = 2451, dimension = [30, 513]):
    ''' 
        Creates random data with a standard size of 128 * 128
        Generates a float Tensor object with dimensions 1 * 1 * height * width
    '''
    return np.random.rand(4, 2, 30, 513)
    
def trainNetwork(dataset='model1'):
    save_name = 'dn_model'
    # Encoder
    denoiser_vocals = Encoder().cuda()
    autoencoder =  AutoEncoder()
    
    autoencoder.load_state_dict(torch.load(config.log_dir + dataset + '.pt'))
    
    optimizer   =  torch.optim.Adagrad(denoiser_vocals.parameters(), 0.0001 )

    loss_func   =  nn.MSELoss( size_average=False )

    optimizer.zero_grad()
    
    train_evol = []

    val_evol = []
    
    count = 0
    
    for epoch in range(config.dn_num_epochs):
    
        start_time = time.time()
        
        train_gen = data_gen()

        val_gen = data_gen(mode= "Val")
        
        optimizer.zero_grad()
        
        for inputs, targets in train_gen:
            
            out_sources = autoencoder(inputs)
            
            vocals = output[:,:2,:,:]

            drums = output[:,2:4,:,:]

            bass = output[:,4:6,:,:]

            others = output[:,6:,:,:]
            
            total_sources = vocals + bass + drums + others

            mask_vocals = vocals/total_sources

            mask_drums = drums/total_sources

            mask_bass = bass/total_sources

            mask_others = 1 - (mask_vocals+mask_drums+mask_bass)

            out_vocals = in_batches * mask_vocals

            out_drums = in_batches * mask_drums

            out_bass = in_batches * mask_bass

            out_others = in_batches * mask_others
            
            denoised_vocals = denoiser_vocals(Variable(torch.FloatTensor(out_vocals)).cuda())
            
            step_loss = loss_func(denoised_vocals, targets[:,:2,:,:])
            
            train_loss += step_loss.item()
            
            step_loss.backward()
            
            optimizer.step()
            
            utils.progress(count,config.batches_per_epoch_train, suffix = 'training done')

            count+=1
            
        count = 0

        for inputs, targets in val_gen:
        
            out_sources = autoencoder(inputs)
            
            vocals = output[:,:2,:,:]

            drums = output[:,2:4,:,:]

            bass = output[:,4:6,:,:]

            others = output[:,6:,:,:]
            
            total_sources = vocals + bass + drums + others

            mask_vocals = vocals/total_sources

            mask_drums = drums/total_sources

            mask_bass = bass/total_sources

            mask_others = 1 - (mask_vocals+mask_drums+mask_bass)

            out_vocals = in_batches * mask_vocals

            out_drums = in_batches * mask_drums

            out_bass = in_batches * mask_bass

            out_others = in_batches * mask_others
            
            denoised_vocals = denoiser_vocals(Variable(torch.FloatTensor(out_vocals)).cuda())
        
            step_loss = loss_func(denoised_vocals, targets[:,:2,:,:])
            
            val_evol += step_loss.item()
            
        duration = time.time()-start_time

        if (epoch+1)%config.print_every == 0:
            print('epoch %d/%d, took %.2f seconds, epoch total loss: %.7f' % (epoch+1, config.num_epochs, duration, train_loss))
            print('                                  validation total loss: %.7f' % ( val_loss))
            
        if (epoch+1)%config.save_every  == 0:
            torch.save(autoencoder.state_dict(), config.log_dir+save_name+'_'+str(epoch)+'.pt')
            np.save(config.log_dir+'dn_train_loss',np.array(train_evol))
            np.save(config.log_dir+'dn_val_loss',np.array(val_evol))
            
            
if __name__ == "__main__":
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        if len(sys.argv)<3:
            trainNetwork()
        else:
            dataset = sys.argv[2]
            
            print("Training")
            trainNetwork(dataset)


