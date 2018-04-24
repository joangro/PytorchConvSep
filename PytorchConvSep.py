import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict


class AutoEncoder(nn.Module):
    def __init__(self, conv_hor_in = (1, 513), conv_ver_in = (12, 1), out_channels_in = 2):
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
        
        # init architecture parameters
        self.out_channels = out_channels_in
        
        ### ENCODER
        # init autoencoder architecture shape
        # we need to use sequential, as it's a way to add modules one after the 
        # another in an ordered way
        self.encoder = nn.Sequential(
            nn.Conv2d(2, self.out_channels, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU(),   # can be any activation function
            nn.Conv2d(self.out_channels, 2, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU()
        )
        
        ### DECODERS
        self.decode_drums = nn.Sequential(
            nn.ConvTranspose2d(2, self.out_channels, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_voice = nn.Sequential(
            nn.ConvTranspose2d(2, self.out_channels, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_bass = nn.Sequential(
            nn.ConvTranspose2d(2, self.out_channels, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        self.decode_other = nn.Sequential(
            nn.ConvTranspose2d(2, self.out_channels, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        
        ### FULLY CONNECTED LAYERS
        self.layer_first = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU()
        )
        self.layer_drums = nn.Sequential(
            nn.Linear(128, 3),
            nn.ReLU()
        )
        self.layer_voice = nn.Sequential(
            nn.Linear(128, 3),
            nn.ReLU()
        )
        self.layer_bass = nn.Sequential(
            nn.Linear(128, 3),
            nn.ReLU()
        )
        self.layer_other = nn.Sequential(
            nn.Linear(128, 3),
            nn.ReLU()
        )
        
        # put the layers and deconv in libraries to make it easier to work with
        self.layers = OrderedDict({
            "voice": self.layer_voice,
            "other": self.layer_other,
            "drums": self.layer_drums,
            "bass":  self.layer_bass,

            })
            
        self.deconvs = OrderedDict({
            "voice": self.decode_voice,
            "other": self.decode_other,
            "drums": self.decode_drums,
            "bass":  self.decode_bass,

            })
        
        # OUTPUT MATRIX
        # Create a tensor variable with shape (15, 1, 30, 513)
        # care, as the channels dimension is initialized with 1, and we are appending to it
        self.final_output = Variable()

        
    def forward(self, x):        
        encode = self.encoder(x)
        print " Encoded input shape ", encode.shape
        layer_output = self.layer_first(encode)
        print " First NN layer shape ", layer_output.shape
        
        output_flag = 0
        for key in self.layers:
            print "\nDecoding "  + key +  " source..."
            
            source_output = self.layers[key](layer_output)
            print "Source NN shape:     ", source_output.shape
            
            source_deconv = self.deconvs[key](source_output)
            print "Source deconv shape: ", source_output.shape
            # The first time we reemplace the output with the current source 
            # output in the first two channels, otherwise we append
            if  output_flag == 0:
                self.final_output = source_deconv
                output_flag = 1
            else:
                self.final_output = torch.cat((source_deconv, self.final_output), dim = 1)
                
            print "New number of channels: ", self.final_output.shape[1]
            
        return self.final_output


def GenerateRandomData(seed = 2451, dimension = [128, 128]):
    ''' 
        Creates random data with a standard size of 128 * 128
        Generates a float Tensor object with dimensions 1 * 1 * height * width
    '''
    torch.manual_seed(seed)             # seed for replication purposes
    dtype = torch.FloatTensor           # afterwards it can be modified to work with CUDA
    
    rNum = torch.randn(dimension).type(dtype)
    # needs 4 dimensions to work with conv2d (BatchSize, Channels, Height, Width)
    # we add the two extra dimensions at the beginning
    while len(rNum.shape) is not 4:
        rNum = rNum.unsqueeze(0)
  
    print rNum.size()   	 	        
    return rNum
    
    
def trainNetwork(track = 0, sources = 0):
    # Init autoencoder object
    autoencoder_audio = AutoEncoder()


    # Init learning parameters
    learning_rate = 0.2
    optimization_audio  = torch.optim.Adagrad(autoencoder_audio.parameters(), learning_rate) # gradient descent - audio ae
    loss_function = nn.MSELoss()
    
    # Call function to generate random data
    rNum = GenerateRandomData(dimension = [15, 2, 30, 513])
    
    # create a variable object of the data
    
    test_data = Variable(rNum)	

    # Train Autoencoder
    print "Training Audio Autoencoder"
    output = autoencoder_audio(test_data)
    # output
    # doesn't work because of the new output size ( 16, 8, 30, 513 )
    '''
    loss = loss_function(output, test_data)

    optimization_audio.zero_grad() # reset to zero 
    loss.backward()
    optimization_audio.step()
    '''

if __name__ == "__main__":
    trainNetwork()

    





