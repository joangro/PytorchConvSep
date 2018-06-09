from __future__ import print_function
from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
from data_pipeline import data_gen
import matplotlib.pyplot as plt
import config
import utils
import datetime
import sys, os
import time
import h5py
import stempeg

def loss_calc(inputs, targets, loss_func, autoencoder):

    eps=1e-18

    targets = targets *np.linspace(1.0,0.7,513)

    targets_cuda = Variable(torch.FloatTensor(targets)).cuda()
    inputs = Variable(torch.FloatTensor(inputs)).cuda()


    output = autoencoder(inputs) + eps

    # import pdb;pdb.set_trace()

    vocals = output[:,:2,:,:]

    drums = output[:,2:4,:,:]

    bass = output[:,4:6,:,:]

    others = output[:,6:,:,:]

    total_sources = vocals + bass + drums + others

    mask_vocals = vocals/total_sources

    mask_drums = drums/total_sources

    mask_bass = bass/total_sources

    mask_others = others/total_sources

    out_vocals = inputs * mask_vocals

    out_drums = inputs * mask_drums

    out_bass = inputs * mask_bass

    out_others = inputs * mask_others

    targets_vocals = targets_cuda[:,:2,:,:]

    targets_drums = targets_cuda[:,2:4,:,:]

    targets_bass = targets_cuda[:,4:6,:,:]

    targets_others = targets_cuda[:,6:,:,:]

    step_loss_vocals = loss_func(out_vocals, targets_vocals)
    alpha_diff =  config.alpha * loss_func(out_vocals, targets_bass)
    alpha_diff += config.alpha * loss_func(out_vocals, targets_drums)
    beta_other_voc   =  config.beta_voc * loss_func(out_vocals, targets_others)

    step_loss_drums = loss_func(out_drums, targets_drums)
    alpha_diff +=  config.alpha *  loss_func(out_drums, targets_vocals)
    alpha_diff +=  config.alpha *  loss_func(out_drums, targets_bass)
    beta_other  =  config.beta  *  loss_func(out_drums, targets_others)

    step_loss_bass = loss_func(out_bass, targets_bass)
    alpha_diff +=  config.alpha *  loss_func(out_bass, targets_vocals)
    alpha_diff +=  config.alpha *  loss_func(out_bass, targets_drums)
    beta_other  =  config.beta  *  loss_func(out_bass, targets_others)

    return step_loss_vocals, step_loss_drums, step_loss_bass, alpha_diff, beta_other, beta_other_voc

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
            # nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            # nn.ReLU()
        )
        self.decode_voice = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            # nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            # nn.ReLU()
        )
        self.decode_bass = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            # nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            # nn.ReLU()
        )
        self.decode_other = nn.Sequential(
            nn.ConvTranspose2d(2, 2, self.conv_ver, stride = 1, padding = 0, bias = True),
            # nn.ReLU(),
            nn.ConvTranspose2d(2, 2, self.conv_hor, stride = 1, padding = 0, bias = True),
            # nn.ReLU()
        )
        
        ### FULLY CONNECTED LAYERS
        self.layer_first = nn.Linear(38, 128)

        self.layer_drums = nn.Sequential(
            nn.Linear(128, 38),
            # nn.ReLU()
        )
        self.layer_voice = nn.Sequential(
            nn.Linear(128, 38),
            # nn.ReLU()
        )
        self.layer_bass = nn.Sequential(
            nn.Linear(128, 38),
            # nn.ReLU()
        )
        self.layer_other = nn.Sequential(
            nn.Linear(128, 38),
            # nn.ReLU()
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

        encode = encode.view(config.batch_size, -1)

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



    
def trainNetwork(save_name = 'model_e' + str(config.num_epochs) + '_b' + str(config.batches_per_epoch_train) + '_bs' + str(config.batch_size) ):
    assert torch.cuda.is_available(), "Code only usable with cuda"

    #autoencoder =  AutoEncoder().cuda()

    autoencoder =  AutoEncoder().cuda()

    optimizer   =  torch.optim.Adam(autoencoder.parameters(), lr = 0.000001)

    #loss_func   =  nn.MSELoss( size_average=False )
    loss_func   =  nn.L1Loss( size_average=False )

    train_evol = []

    val_evol = []

    count = 0

    for epoch in range(config.num_epochs):

        start_time = time.time()

        generator = data_gen()

        val_gen = data_gen(mode= "Val")

        train_loss = 0
        train_loss_vocals = 0
        train_loss_drums = 0
        train_loss_bass = 0
        train_alpha_diff = 0 
        train_beta_other = 0
        train_beta_other_voc = 0

        val_loss = 0
        val_loss_vocals = 0
        val_loss_drums = 0
        val_loss_bass = 0
        val_alpha_diff = 0 
        val_beta_other = 0
        val_beta_other_voc = 0



        optimizer.zero_grad()

        count = 0

        for inputs, targets in generator:

            step_loss_vocals, step_loss_drums, step_loss_bass, alpha_diff, beta_other, beta_other_voc = loss_calc(inputs, targets, loss_func, autoencoder)
            # start_time = time.time()

            # add regularization terms from paper
            step_loss = abs(step_loss_vocals + step_loss_drums + step_loss_bass - beta_other - alpha_diff - beta_other_voc)

            # print time.time()-start_time
            # import pdb;pdb.set_trace()
            # start_time = time.time()

            train_loss += step_loss.item()
            if np.isnan(train_loss):
               # import pdb;pdb.set_trace()
               print ("error output contains NaN")
            train_loss_vocals +=step_loss_vocals.item()
            train_loss_drums +=step_loss_drums.item()
            train_loss_bass +=step_loss_bass.item()
            train_alpha_diff += alpha_diff.item()
            train_beta_other += beta_other.item()  
            train_beta_other_voc+=beta_other_voc.item()          

            step_loss.backward()

            optimizer.step()
            # print time.time()-start_time

            utils.progress(count,config.batches_per_epoch_train, suffix = 'training done')

            count+=1

        train_loss = train_loss/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_loss_vocals = train_loss_vocals/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_loss_drums = train_loss_drums/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_loss_bass = train_loss_bass/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_alpha_diff = train_alpha_diff/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_beta_other = train_beta_other/(config.batches_per_epoch_train*count*config.max_phr_len*513)
        train_beta_other_voc= train_beta_other_voc/(config.batches_per_epoch_train*count*config.max_phr_len*513)

        train_evol.append([train_loss,train_loss_vocals,train_loss_drums,train_loss_bass,train_alpha_diff,train_beta_other,train_beta_other_voc])

        count = 0

        for inputs, targets in val_gen:

            step_loss_vocals, step_loss_drums, step_loss_bass, alpha_diff, beta_other, beta_other_voc  = loss_calc(inputs, targets, loss_func, autoencoder)

            # add regularization terms from paper
            step_loss = abs(step_loss_vocals + step_loss_drums + step_loss_bass - beta_other - alpha_diff - beta_other_voc)

            val_loss += step_loss.item()
            val_loss_vocals +=step_loss_vocals.item()
            val_loss_drums +=step_loss_drums.item()
            val_loss_bass +=step_loss_bass.item()
            val_alpha_diff += alpha_diff.item()
            val_beta_other += beta_other.item()  
            val_beta_other_voc+=beta_other_voc.item()           

            utils.progress(count,config.batches_per_epoch_val, suffix = 'validation done')

            count+=1
        val_loss = val_loss/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_loss_vocals = val_loss_vocals/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_loss_drums = val_loss_drums/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_loss_bass = val_loss_bass/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_alpha_diff = val_alpha_diff/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_beta_other = val_beta_other/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_beta_other_voc= val_beta_other_voc/(config.batches_per_epoch_val*count*config.max_phr_len*513)
        val_evol.append([val_loss,val_loss_vocals,val_loss_drums,val_loss_bass,val_alpha_diff,val_beta_other,val_beta_other_voc])

        # import pdb;pdb.set_trace()

        duration = time.time()-start_time

        if (epoch+1)%config.print_every == 0:
            print('epoch %d/%d, took %.2f seconds, epoch total loss: %.7f' % (epoch+1, config.num_epochs, duration, train_loss))
            print('                                  epoch vocal loss: %.7f' % (train_loss_vocals))
            print('                                  epoch drums loss: %.7f' % (train_loss_drums))
            print('                                  epoch bass  loss: %.7f' % (train_loss_bass))
            print('                                  epoch alpha diff: %.7f' % (train_alpha_diff))
            print('                                  epoch beta  diff: %.7f' % (train_beta_other))
            print('                                  epoch beta2 diff: %.7f' % (train_beta_other_voc))

            print('                                  validation total loss: %.7f' % ( val_loss))
            print('                                  validation vocal loss: %.7f' % (val_loss_vocals))
            print('                                  validation drums loss: %.7f' % (val_loss_drums))
            print('                                  validation bass  loss: %.7f' % (val_loss_bass))
            print('                                  validation alpha diff: %.7f' % (val_alpha_diff))
            print('                                  validation beta  diff: %.7f' % (val_beta_other))
            print('                                  validation beta2 diff: %.7f' % (val_beta_other_voc))

        # import pdb;pdb.set_trace()
        if (epoch+1)%config.save_every  == 0:
            torch.save(autoencoder.state_dict(), config.log_dir+save_name+'_'+str(epoch)+'.pt')
            np.save(config.log_dir+'train_loss',np.array(train_evol))
            np.save(config.log_dir+'val_loss',np.array(val_evol))
        # import pdb;pdb.set_trace()


    torch.save(autoencoder.state_dict(), config.log_dir+save_name+'_'+str(epoch)+'.pt')


def evalNetwork(file_name, load_name='model_e4000_b500_bs1_1699', plot = False, synth = False):
    autoencoder_audio = AutoEncoder().cuda()
    epoch = 50
    # autoencoder_audio.load_state_dict(torch.load(config.log_dir+load_name+'_'+str(epoch)+'.pt'))
    autoencoder_audio.load_state_dict(torch.load(config.log_dir+'model1/'+load_name+'.pt'))

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    # import pdb;pdb.set_trace()
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    max_feat_tars = max_feat[:8,:].reshape(8,1,513)
    min_feat_tars = min_feat[:8,:].reshape(8,1,513)

    max_feat_ins = max_feat[-2:,:].reshape(2,1,513)
    min_feat_ins = min_feat[-2:,:].reshape(2,1,513)


    audio,fs = stempeg.read_stems(os.path.join(config.wav_dir_test,file_name), stem_id=[0,1,2,3,4])

    mixture = audio[0]

    drums = audio[1]

    bass = audio[2]

    acc = audio[3]

    vocals = audio[4]

    mix_stft, mix_phase = utils.stft_stereo(mixture,phase=True)

    mix_stft = (mix_stft-min_feat_ins)/(max_feat_ins-min_feat_ins)

    drums_stft = utils.stft_stereo(drums)

    bass_stft = utils.stft_stereo(bass)

    acc_stft = utils.stft_stereo(acc)

    voc_stft = utils.stft_stereo(vocals)

    in_batches, nchunks_in = utils.generate_overlapadd(mix_stft)

    out_batches = []

    for in_batch in in_batches:
        # import pdb;pdb.set_trace()
        in_batch = Variable(torch.FloatTensor(in_batch)).cuda()
        out_batch = autoencoder_audio(in_batch)
        out_batches.append(np.array(out_batch.data.cpu().numpy()))
        

    out_batches = np.array(out_batches)
    


    vocals = out_batches[:,:,:2,:,:]

    drums = out_batches[:,:,2:4,:,:]

    bass = out_batches[:,:,4:6,:,:]

    others = out_batches[:,:,6:,:,:]

    total_sources = vocals + bass + drums + others

    mask_vocals = vocals/total_sources

    mask_drums = drums/total_sources

    mask_bass = bass/total_sources

    mask_others = 1 - (mask_vocals+mask_drums+mask_bass)

    out_vocals = in_batches * mask_vocals

    out_drums = in_batches * mask_drums

    out_bass = in_batches * mask_bass

    out_others = in_batches * mask_others
    

    out_vocals = out_vocals*(max_feat_tars[:2,:,:]-min_feat_tars[:2,:,:])+min_feat_tars[:2,:,:]

    out_drums = out_drums*(max_feat_tars[2:4,:,:]-min_feat_tars[2:4,:,:])+min_feat_tars[2:4,:,:]

    out_bass = out_bass*(max_feat_tars[4:6,:,:]-min_feat_tars[4:6,:,:])+min_feat_tars[4:6,:,:]

    out_others = out_others*(max_feat_tars[6:,:,:]-min_feat_tars[6:,:,:])+min_feat_tars[6:,:,:]


    
    out_drums = utils.overlapadd(out_drums, nchunks_in) 

    out_bass = utils.overlapadd(out_bass, nchunks_in) 

    out_others = utils.overlapadd(out_others, nchunks_in) 

    out_vocals = utils.overlapadd(out_vocals, nchunks_in) 

    if plot:
        plt.figure(1)
        plt.suptitle(file_name[:-9])
        ax1 = plt.subplot(411)
        plt.imshow(np.log(drums_stft[0].T),aspect = 'auto', origin = 'lower')
        ax1.set_title("Drums Left Channel Ground Truth", fontsize = 10)
        ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_drums[0].T),aspect = 'auto', origin = 'lower')
        ax2.set_title("Drums Left Channel Network Output", fontsize = 10)
        ax3 = plt.subplot(413, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(drums_stft[1].T),aspect = 'auto', origin = 'lower')
        ax3.set_title("Drums Right Channel Ground Truth", fontsize = 10)
        ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_drums[1].T),aspect = 'auto', origin = 'lower')
        ax4.set_title("Drums Right Channel Network Output", fontsize = 10)

        plt.figure(2)
        plt.suptitle(file_name[:-9])
        ax1 = plt.subplot(411)
        plt.imshow(np.log(voc_stft[0].T),aspect = 'auto', origin = 'lower')
        ax1.set_title("Vocals Left Channel Ground Truth", fontsize = 10)
        ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_vocals[0].T),aspect = 'auto', origin = 'lower')
        ax2.set_title("Vocals Left Channel Network Output", fontsize = 10)
        ax3 = plt.subplot(413, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(voc_stft[1].T),aspect = 'auto', origin = 'lower')
        ax3.set_title("Vocals Right Channel Ground Truth", fontsize = 10)
        ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_vocals[1].T),aspect = 'auto', origin = 'lower')
        ax4.set_title("Vocals Right Channel Network Output", fontsize = 10)


        plt.figure(3)
        plt.suptitle(file_name[:-9])
        ax1 = plt.subplot(411)
        plt.imshow(np.log(bass_stft[0].T),aspect = 'auto', origin = 'lower')
        ax1.set_title("Bass Left Channel Ground Truth", fontsize = 10)
        ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_bass[0].T),aspect = 'auto', origin = 'lower')
        ax2.set_title("Bass Left Channel Network Output", fontsize = 10)
        ax3 = plt.subplot(413, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(bass_stft[1].T),aspect = 'auto', origin = 'lower')
        ax3.set_title("Bass Right Channel Ground Truth", fontsize = 10)
        ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_bass[1].T),aspect = 'auto', origin = 'lower')
        ax4.set_title("Bass Right Channel Network Output", fontsize = 10)

        plt.figure(4)
        plt.suptitle(file_name[:-9])
        ax1 = plt.subplot(411)
        plt.imshow(np.log(acc_stft[0].T),aspect = 'auto', origin = 'lower')
        ax1.set_title("Others Left Channel Ground Truth", fontsize = 10)
        ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_others[0].T),aspect = 'auto', origin = 'lower')
        ax2.set_title("Others Left Channel Network Output", fontsize = 10)
        ax3 = plt.subplot(413, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(acc_stft[1].T),aspect = 'auto', origin = 'lower')
        ax3.set_title("Others Right Channel Ground Truth", fontsize = 10)
        ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_others[1].T),aspect = 'auto', origin = 'lower')
        ax4.set_title("Others Right Channel Network Output", fontsize = 10)


        plt.show()

    if synth:
        # import pdb;pdb.set_trace()
        utils.inverse_stft_write(out_drums[:,:mix_phase.shape[1],:],mix_phase,config.out_dir+file_name+"_drums.wav")
        utils.inverse_stft_write(out_bass[:,:mix_phase.shape[1],:],mix_phase,config.out_dir+file_name+"_bass.wav")
        utils.inverse_stft_write(out_vocals[:,:mix_phase.shape[1],:],mix_phase,config.out_dir+file_name+"_vocals.wav")
        utils.inverse_stft_write(out_others[:,:mix_phase.shape[1],:],mix_phase,config.out_dir+file_name+"_others.wav")


def plot_loss():
    train_loss = np.load(config.log_dir+'train_loss.npy')
    val_loss = np.load(config.log_dir+'val_loss.npy')

    plt.plot(train_loss)
    plt.show()
        
if __name__ == '__main__':
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        print("Training")
        trainNetwork()
    elif sys.argv[1] == '-synth' or sys.argv[1] == '--synth' or sys.argv[1] == '--s' or sys.argv[1] == '-s':
        if len(sys.argv)<3:
            print("Please give a file to synthesize")
        else:
            file_name = sys.argv[2]
            if not file_name.endswith('.stem.mp4'):
                file_name = file_name+'.stem.mp4'

            print("Synthesizing File %s"% file_name)
            if '-p' in sys.argv or '--p' in sys.argv or '-plot' in sys.argv or '--plot' in sys.argv:                
                if '-ns' in sys.argv or '--ns' in sys.argv: 
                    print("Just showing plots for File %s"% sys.argv[2])
                    evalNetwork(file_name,plot=True, synth =False)
                else:
                    print("Showing Plots And Synthesizing File %s"% sys.argv[2])
                    evalNetwork(file_name,plot=True, synth =True)
            else:
                evalNetwork(file_name,plot=False, synth =True)


    elif sys.argv[1] == '-plot' or sys.argv[1] == '--pl' or sys.argv[1] == '--plot_loss':
        plot_loss()
            # else:
            #     print("Synthesizing File %s, Not Showing Plots"% sys.argv[2])
            #     synth_file(file_name,show_plots=False, save_file=True)

    elif sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --train to train the model"%sys.argv[0])
        print("%s --synth <filename> to synthesize file"%sys.argv[0])
        print("%s --synth <filename> -- plot to synthesize file and show plots"%sys.argv[0])
        print("%s --synth <filename> -- plot --ns to just show plots"%sys.argv[0])
    else:
        print("Unable to decipher inputs please use %s --help for help on how to use this function"%sys.argv[0])
