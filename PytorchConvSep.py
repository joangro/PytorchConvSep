import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn



class AutoEncoder(nn.Module):
    def __init__(self, conv_hor_in = (128, 4), conv_ver_in = (1, 64), out_channels_in = 2):
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
            nn.Conv2d(1, self.out_channels, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU(),   # can be any activation function
            nn.Conv2d(self.out_channels, 1, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU()
        )
        ### DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, self.out_channels, self.conv_ver, stride = 1, padding = 1, bias = True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, 1, self.conv_hor, stride = 1, padding = 0, bias = True),
            nn.ReLU()
        )
        
    def forward(self, x):        
        encode = self.encoder(x)
        print "Encoded image dimensions: "  + str(encode.size())
        decode = self.decoder(encode)
        print "Decoded image dimensions: "  + str(decode.size())
        print "Horizontal filter shape: "   + str(self.conv_hor)
        print "Vertical filter shape: "     + str(self.conv_ver)
        assert str(decode.size()[2]) == str(x.size()[2]), "Wrong output size (height, different than 128)"
        assert str(decode.size()[3]) == str(x.size()[3]), "Wrong output size (width, different than 128)"
        return decode


def GenerateRandomData(seed = 2451, height = 128, width = 128):
    ''' 
        Creates random data with a standard size of 128 * 128
        Generates a float Tensor object with dimensions 1 * 1 * height * width
    '''
    torch.manual_seed(seed)             # seed for replication purposes
    dtype = torch.FloatTensor           # afterwards it can be modified to work with CUDA
    
    rNum = torch.randn(height, width).type(dtype)
    # needs 4 dimensions to work with conv2d (BatchSize, Channels, Height, Width)
    # we add the two extra dimensions at the beginning
    rNum = rNum.unsqueeze(0)
    rNum = rNum.unsqueeze(0) 
    print rNum.size()   	 	        # output = (1, 1, 128, 128)
    return rNum


if __name__ == "__main__":
    
    # Init autoencoder object
    autoencoder_audio = AutoEncoder()


    # Init learning parameters
    learning_rate = 0.2
    optimization_audio  = torch.optim.Adagrad(autoencoder_audio.parameters(), learning_rate) # gradient descent - audio ae
    loss_function = nn.MSELoss()
    
    # Call function to generate random data
    rNum = GenerateRandomData()
    
    # create a variable object of the data
    
    test_data = Variable(rNum)	

    # Train Autoencoder
    print "Training Audio Autoencoder"
    output = autoencoder_audio(test_data)
    print output
    loss = loss_function(output, test_data)

    optimization_audio.zero_grad() # reset to zero 
    loss.backward()
    optimization_audio.step()




