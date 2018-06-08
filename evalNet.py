import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
from data_pipeline import data_gen
import matplotlib.pyplot as plt
import config
import utils
import sys, os
import time
import h5py
import stempeg
import mir_eval
from PytorchConvSep import AutoEncoder
from separate_dsd import main_dsd

def createWav():
    pass

def evalNets(pcs_model = 'model1', file_to_eval = "None", path = '/media/joan/Data/Users/Joan/Desktop/DATA/STEMS'):
    
    autoencoder_audio = AutoEncoder().cuda()
    autoencoder_audio.load_state_dict(torch.load(config.log_dir+'/models/'+pcs_model+'.pt'))
    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    max_feat_tars = max_feat[:8,:].reshape(8,1,513)
    min_feat_tars = min_feat[:8,:].reshape(8,1,513)

    max_feat_ins = max_feat[-2:,:].reshape(2,1,513)
    min_feat_ins = min_feat[-2:,:].reshape(2,1,513)

    wav_files=[x for x in os.listdir(config.wav_dir_test) if x.endswith('.stem.mp4') and not x.startswith(".")]
    
    err_total = np.array()
    
    err_drums = np.array()
    
    err_voice = np.array()
    
    err_bass = np.array()
    
    err_other = np.array()
        
    for file_name in wav_files:
    
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
        import pdb;pdb.trace()
        mir_eval.separation.bss_eval_sources(bass, out_bass)



if __name__ == "__main__":
    if len(sys.argv) is 1:
        subfolders = os.listdir('/media/joan/Data/Users/Joan/Desktop/DATA/STEMS')
        evalNets()
        print (subfolders)
    elif sys.argv[1] == '-h' or sys.argv[1] == '-help' or (len(sys.argv) < 2):
        print ('Please input a sequence such as:')
        print("%s -m   <PytorchConvSep model>"%sys.argv[0])
        print("%s -m   <PytorchConvSep model>   -f   <file_to_evaluate_path>"%sys.argv[0])
           
        
    else:
        if (len(sys.argv) < 3):
            print ('Please input a model')
        elif (len(sys.argv) is 3):
            evalNets(pcs_model = sys.argv[2])
        elif (len(sys.argv) is 4):
            print ('Please input a file to evaluate or remove the -f argument')
        elif (len(sys.argv) is 5):
            evalNets(pcs_model = sys.argv[2], file_to_eval = sys.argv[4])
            
                

