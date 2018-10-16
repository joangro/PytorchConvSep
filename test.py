import numpy as np
import os
import time
import h5py

import config, utils


in_dir=config.dir_hdf5
num_batches = config.batches_per_epoch_train

maximus = np.zeros((10,1,513))

minimus = np.ones((10,1,513))*100

count =0

file_list = [x for x in os.listdir(in_dir) if x.endswith('.hdf5') and not x.startswith('._')]

for file_to_open in file_list:
    hdf5_file = h5py.File(in_dir+file_to_open, "r")

    tar_stft = np.array(hdf5_file["tar_stft"])

    tar_stft_max = tar_stft.max(axis = 1).reshape(8,1,513)

    tar_stft_min = tar_stft.min(axis = 1).reshape(8,1,513)

    mix_stft = np.array(hdf5_file["mix_stft"])
    mix_stft_max = mix_stft.max(axis = 1).reshape(2,1,513)
    mix_stft_min = mix_stft.min(axis = 1).reshape(2,1,513)

    if np.isnan(tar_stft).any():
        print "tar nan"
        print file_to_open
    if np.isnan(mix_stft).any():
        print "mix nan"
        print file_to_open

    loc_max = np.concatenate((tar_stft_max,mix_stft_max),axis=0)

    loc_min = np.concatenate((tar_stft_min,mix_stft_min),axis=0)

    maximus = np.concatenate((maximus,loc_max),axis=1).max(axis=1).reshape(10,1,513)

    minimus = np.concatenate((minimus,loc_min),axis=1).min(axis=1).reshape(10,1,513)
    utils.progress(count,100)
    count+=1

# import pdb;pdb.set_trace()

hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

hdf5_file.create_dataset("feats_maximus", [10,513], np.float32) 
hdf5_file.create_dataset("feats_minimus", [10,513], np.float32)   
hdf5_file["feats_maximus"][:] = maximus.reshape(10,513)
hdf5_file["feats_minimus"][:] = minimus.reshape(10,513)
hdf5_file.close()

# import pdb;pdb.set_trace()