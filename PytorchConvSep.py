import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
from data_pipeline import data_gen
import matplotlib.pyplot as plt
import config


class AutoEncoder(nn.Module):
    def __init__(self, conv_hor_in = (1, 513), conv_ver_in = (12, 1)):
        '''
        I tried to make this as customizable as possible.
        INPUT:
                -   conv_hor_in: size of the kernel filter for the horizontal convolution.
                                 Must be a tuple.
                                                        (Height, Width)
                    
                -   conv_hor_in: same as the conv_hor_in but for the vertical convolution.
                                 Must be a tuple.
                                                        (Height, Width)
                    
                -   out_channels_in: number of channels of features we want the network to create 
                                   on each convolution
        '''
    
        super(AutoEncoder, self).__init__() # reference current class in each instance
        
        # init conv/deconv filter shapes
        self.conv_hor = conv_hor_in
        self.conv_ver = conv_ver_in

        ### ENCODER
        # init autoencoder architecture shape
        # we need to use sequential, as it's a way to add modules one after the 
        # another in an ordered way
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.Conv2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True)
        )
        
        ### DECODERS
        self.decode_drums = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_voice = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_bass = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_other = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        
        ### FULLY CONNECTED LAYERS
        self.layer_first = nn.Linear(38, 128)

        self.layer_drums = nn.Sequential(
            nn.Linear(128, 38),
            nn.ReLU()
        )
        self.layer_voice = nn.Sequential(
            nn.Linear(128, 38),
            nn.ReLU()
        )
        self.layer_bass = nn.Sequential(
            nn.Linear(128, 38),
            nn.ReLU()
        )
        self.layer_other = nn.Sequential(
            nn.Linear(128, 38),
            nn.ReLU()
        )
        
        # put the layers and deconv in libraries to make it easier to work with
        self.layers = OrderedDict([
            ("voice", self.layer_voice),
            ("drums", self.layer_drums),
            ("bass",  self.layer_bass),
            ("other", self.layer_other)
            ])
            
        self.deconvs = OrderedDict([
            ("voice", self.decode_voice),
            ("drums", self.decode_drums),
            ("bass",  self.decode_bass),
            ("other", self.decode_other)
            ])
                
        # OUTPUT MATRIX
        # Create a tensor variable with shape (15, 1, 30, 513)
        # care, as the channels dimension is initialized with 1, and we are appending to it
        self.final_output = Variable()

        
    def forward(self, x):  
          
        encode = self.encoder(x)

        encode = encode.view(15, -1)

        layer_output = self.layer_first(encode)
        
        output_flag = 0
        
        for key in self.layers:
            
            source_output = self.layers[key](layer_output)

            source_deconv = self.deconvs[key](source_output.view(-1,2,19,1))

            if  output_flag == 0:
                self.final_output = source_deconv
                output_flag = 1
            else:
                self.final_output = torch.cat((source_deconv, self.final_output), dim = 1)
                
        return self.final_output


def trainNetwork(track = 0, sources = 0):

    assert torch.cuda.is_available(), "Code only usable with cuda"

    autoencoder =  AutoEncoder().cuda()

    optimizer   =  torch.optim.Adagrad( autoencoder.parameters(), config.init_lr )

    loss_func   =  nn.MSELoss( size_average=False )
    #loss_func   =  nn.L1Loss( size_average=False )

    for epoch in range(10):

        generator = data_gen()

        train_loss = 0

        train_evol = 0

        optimizer.zero_grad()

        count = 0


        for inputs, targets in generator:
        
            targets = torch.from_numpy(targets).cuda()
            data = torch.from_numpy(inputs).cuda()
            
            data = Variable(data)
            
            output = autoencoder(data)

            step_loss = loss_func(output, targets)

            train_loss += step_loss

            train_evol.append(train_loss)

            loss.backward()

            optimizer.step()

            utils.progress(count,config.batches_per_epoch_train, suffix = 'training done')

            count+=1

        if (epoch+1)%config.print_every == 0:
            print (train_loss)
        if (epoch+1)%config.save_every  == 0:
            torch.save(autoencoder.state_dict(), './joan-test')
            print (autoencoder.state_dict())



if __name__ == "__main__":
    trainNetwork()

