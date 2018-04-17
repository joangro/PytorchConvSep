import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class AutoEncoder(nn.Module):
    def __init__(self, conv_hor_in = (64, 16), conv_ver_in = (2, 8), out_channels_in = 2):
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
            nn.Conv2d(1, self.out_channels, self.conv_hor, stride = 2, padding = 0, bias = True),
            nn.ReLU(),   # can be any activation function
            nn.Conv2d(self.out_channels, 1, self.conv_ver, stride = 4, padding = 1, bias = True),
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
        decode = self.decoder(x)
        print "Decoded image dimensions: "  + str(decode.size())
        print "Horizontal filter shape: "   + str(self.conv_hor)
        print "Vertical filter shape: "     + str(self.conv_ver)
        return decode

autoencoder = AutoEncoder()


learning_rate = 0.2
optimization  = torch.optim.Adagrad(autoencoder.parameters(), learning_rate) # gradient descent
loss_function = nn.MSELoss()


# test with random data
torch.manual_seed(2451) 
dtype = torch.FloatTensor
rNum = torch.randn(128, 128).type(dtype)

rNum = rNum.unsqueeze(0)
rNum = rNum.unsqueeze(0) # needs 4 dimensions to work with conv2d
print rNum.size()   # (1, 1, 128, 128)
test_data = Variable(rNum)


output = autoencoder(test_data)
print output
loss = loss_function(output, test_data)

optimization.zero_grad() # reset to zero 
loss.backward()
optimization.step()



