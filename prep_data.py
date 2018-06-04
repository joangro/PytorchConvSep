# from __future__ import division
import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import h5py

import matplotlib.pyplot as plt

import sys
import stempeg

import config
import utils


def main():

    # maximus=np.zeros(66)
    # minimus=np.ones(66)*1000


    wav_files=[x for x in os.listdir(config.wav_dir_test) if x.endswith('.stem.mp4') and not x.startswith(".")]
    count=0


    for lf in wav_files:
        
        # print(lf)
        audio,fs = stempeg.read_stems(os.path.join(config.wav_dir_test,lf), stem_id=[0,1,2,3,4])

        mixture = audio[0]

        drums = audio[1]

        bass = audio[2]

        acc = audio[3]

        vocals = audio[4]

        mix_stft = utils.stft_stereo(mixture)

        drums_stft = utils.stft_stereo(drums)

        bass_stft = utils.stft_stereo(bass)

        acc_stft = utils.stft_stereo(acc)

        voc_stft = utils.stft_stereo(vocals)

        hdf5_file = h5py.File(config.dir_hdf5_test+lf[:-9]+'.hdf5', mode='w')

        shape_train = [config.channels, voc_stft.shape[1], config.features]

        # import pdb;pdb.set_trace()

        # hdf5_file.create_dataset("voc_stft", shape_train, np.float32)

        hdf5_file.create_dataset("mix_stft", shape_train, np.float32)

        # hdf5_file.create_dataset("drums_stft", shape_train, np.float32)

        # hdf5_file.create_dataset("bass_stft", shape_train, np.float32)

        # hdf5_file.create_dataset("acc_stft", shape_train, np.float32)

        hdf5_file.create_dataset("tar_stft", [config.channels*4, voc_stft.shape[1], config.features], np.float32)

        # hdf5_file["voc_stft"][:,:,:] = voc_stft

        hdf5_file["mix_stft"][:,:,:] = mix_stft

        # hdf5_file["drums_stft"][:,:,:] = drums_stft

        # hdf5_file["bass_stft"][:,:,:] = bass_stft

        # hdf5_file["acc_stft"][:,:,:] = acc_stft

        hdf5_file["tar_stft"][:,:,:] = np.concatenate((voc_stft,drums_stft,bass_stft,acc_stft),axis = 0)

        hdf5_file.close()


        
        

        # np.save(config.dir_npy+lf[:-9]+'_voc_stft',voc_stft)

        # np.save(config.dir_npy+lf[:-9]+'_mix_stft',mix_stft)

        # np.save(config.dir_npy+lf[:-9]+'_drums_stft',drums_stft)

        # np.save(config.dir_npy+lf[:-9]+'_bass_stft',bass_stft)

        # np.save(config.dir_npy+lf[:-9]+'_acc_stft',acc_stft)


        count+=1
        utils.progress(count,len(wav_files))
    import pdb;pdb.set_trace()




if __name__ == '__main__':
    main()
