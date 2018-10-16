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
import utils
import h5py
import stempeg
import os, sys
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 2, (1, 8), stride = 1, padding = 0, bias = True),
            #nn.Conv2d(2, 2, (4, 1), stride = 1, padding = 0, bias = True)
        )
        
        self.rnn = nn.LSTM(506, 506,1, batch_first= True)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 2, (1, 8), stride = 1, padding = 0, bias = True),
            #nn.ConvTranspose2d(2, 2, (4, 1), stride = 1, padding = 0, bias = True)
        )
        self.linear = nn.Sequential(
                      nn.Linear(64, 64),
                      nn.ReLU() 
        )
    def forward(self, x):

        encode = self.encoder(x)
        #print (encode.shape)
        encode = torch.squeeze(encode.view(2,30, -1))
        #print (encode.shape)
        lstm,_ = self.rnn(encode)
        #print (lstm.shape)
        to_decode = lstm.contiguous().view(1, 2, 30, 506)
        #to_decode = self.linear(to_decode)
        decoded = self.decoder(to_decode)
        #print (np.amax(decoded.data.cpu().numpy()))
        return decoded
              
    
def trainNetwork(dataset='model6'):
    save_name = 'dn_model'
    # Encoder
    denoiser_vocals = Encoder().cuda()
    autoencoder =  AutoEncoder()
    
    autoencoder.load_state_dict(torch.load(config.log_dir + dataset + '.pt'))
    
    optimizer   =  torch.optim.SGD(denoiser_vocals.parameters(), 1e-6 )

    loss_func   =  nn.L1Loss( size_average=False )

    optimizer.zero_grad()
    
    train_evol = []

    eval_evol = []
    
    
    for epoch in range(config.dn_num_epochs):
    
        start_time = time.time()
        
        train_gen = data_gen()

        val_gen = data_gen(mode= "Val")
        
        optimizer.zero_grad()
        train_loss = 0
        eval_loss = 0
        count = 0
        for inputs, targets in train_gen:
            
            output = autoencoder(Variable(torch.FloatTensor(inputs))).cuda()
            
            target_vocals = targets[:,:2,:,:]
           
            target_drums = targets[:,2:4,:,:] 

            target_bass = targets[:,4:6,:,:] 

            target_others = targets[:,6:,:,:] 
 
            vocals = output[:,:2,:,:]

            drums = output[:,2:4,:,:]

            bass = output[:,4:6,:,:]

            others = output[:,6:,:,:]
            
            total_sources = vocals + bass + drums + others

            mask_vocals = vocals/total_sources

            mask_drums = drums/total_sources

            mask_bass = bass/total_sources

            mask_others = others/total_sources

            out_vocals = vocals * mask_vocals

            out_drums = drums * mask_drums

            out_bass = bass * mask_bass

            out_others = others * mask_others

            input_vocals = Variable(out_vocals)
           
            denoised_vocals = denoiser_vocals(input_vocals).cuda()
            
            step_loss = loss_func(denoised_vocals, Variable(torch.cuda.FloatTensor(target_vocals), requires_grad = False))
            
            train_loss += step_loss.item()
            step_loss.backward()
            
            optimizer.step()
            
            utils.progress(count,config.batches_per_epoch_train, suffix = 'training done')

            count+=1
        train_evol.append(train_loss)   
        count = 0

        for inputs, targets in val_gen:
        
            out_sources = autoencoder(Variable(torch.FloatTensor(inputs))).cuda()
            
            vocals = output[:,:2,:,:]

            drums = output[:,2:4,:,:]

            bass = output[:,4:6,:,:]

            others = output[:,6:,:,:]
            
            target_vocals = targets[:,:2,:,:]

            target_drums = targets[:,2:4,:,:]

            target_bass = targets[:,4:6,:,:]

            target_otherss = targets[:,6:,:,:]
            
            total_sources = vocals + bass + drums + others

            mask_vocals = vocals/total_sources

            mask_drums = drums/total_sources

            mask_bass = bass/total_sources

            mask_others = others/total_sources

            out_vocals = vocals * mask_vocals

            out_drums = drums * mask_drums

            out_bass = bass * mask_bass

            out_others = others * mask_others
            input_vocals = Variable(out_vocals)
            denoised_vocals = denoiser_vocals(input_vocals).cuda()
        
            step_loss = loss_func(denoised_vocals, Variable(torch.cuda.FloatTensor(target_vocals), requires_grad = False))
            
            eval_loss += step_loss.item()
            utils.progress(count,config.batches_per_epoch_val, suffix = 'validation done')
            count += 1
        eval_evol.append(eval_loss)    
        duration = time.time()-start_time

        if (epoch+1)%config.print_every == 0:
            print('epoch %d/%d, took %.2f seconds, epoch total loss: %.7f' % (epoch+1, config.num_epochs, duration, train_loss/(config.batches_per_epoch_train*count*config.max_phr_len*513)))
            print('                                  validation total loss: %.7f' % ( eval_loss / (config.batches_per_epoch_train*count*config.max_phr_len*513)))
            
        if (epoch+1)%config.save_every  == 0:
            torch.save(denoiser_vocals.state_dict(), config.dn_log_dir+save_name+'_'+str(epoch)+'.pt')
            np.save(config.dn_log_dir+'dn_train_loss',np.array(train_evol))
            np.save(config.dn_log_dir+'dn_val_loss',np.array(eval_evol))


def evalNetwork(file_name='Al James - Schoolboy Facination.stem.mp4', load_name_sep = 'model6',load_name_dn = 'dn_model_719',  plot = True, synth = False):

    autoencoder_audio = AutoEncoder().cuda()
    denoiser = Encoder().cuda()
    epoch = 50
    autoencoder_audio.load_state_dict(torch.load(config.log_dir+load_name_sep+'.pt'))
    denoiser.load_state_dict(torch.load(config.dn_log_dir+load_name_dn+'.pt'))

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

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

    out_vocals_2 = out_vocals*(max_feat_tars[:2,:,:]-min_feat_tars[:2,:,:])+min_feat_tars[:2,:,:]
    out_drums = out_drums*(max_feat_tars[2:4,:,:]-min_feat_tars[2:4,:,:])+min_feat_tars[2:4,:,:]

    out_bass = out_bass*(max_feat_tars[4:6,:,:]-min_feat_tars[4:6,:,:])+min_feat_tars[4:6,:,:]

    out_others = out_others*(max_feat_tars[6:,:,:]-min_feat_tars[6:,:,:])+min_feat_tars[6:,:,:]

    out_batches_vocals = []
    #print (np.array(out_vocals_2).shape)
    for vocal_batch in range(vocals.shape[0]):
        vocal_batch =  Variable(torch.FloatTensor(out_vocals_2[vocal_batch,:,:])).cuda()
        out_batch = denoiser(vocal_batch)
        out_batches_vocals.append(np.array(out_batch.data.cpu().numpy()))
    out_vocals_2 = utils.overlapadd(out_vocals_2, nchunks_in)
    out_vocals = utils.overlapadd(np.array(out_batches_vocals), nchunks_in) 
    #out_vocals = out_vocals*(max_feat_tars[:2,:,:]-min_feat_tars[:2,:,:])+min_feat_tars[:2,:,:]
    print (out_vocals.shape)
    if plot:
        plt.figure(1)
        ax1 = plt.subplot(411)
        plt.imshow(np.log(out_vocals_2[0].T),aspect = 'auto', origin = 'lower')
        ax1.set_title("Vocals Left Channel Input", fontsize = 10)
        ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_vocals[0].T),aspect = 'auto', origin = 'lower')
        ax2.set_title("Vocals Left Channel Network Output", fontsize = 10)
        ax3 = plt.subplot(413, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_vocals_2[1].T),aspect = 'auto', origin = 'lower')
        ax3.set_title("Vocals Right Channel Input", fontsize = 10)
        ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)
        plt.imshow(np.log(out_vocals[1].T),aspect = 'auto', origin = 'lower')
        ax4.set_title("Vocals Right Channel Network Output", fontsize = 10)
   

        plt.show()            
            
if __name__ == "__main__":
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        if len(sys.argv)<3:
            trainNetwork()
        else:
            dataset = sys.argv[2]
            
            print("Training")
            trainNetwork(dataset)
    elif sys.argv[1] == '-e':
        evalNetwork()

