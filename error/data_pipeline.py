import numpy as np
import os
import time
import h5py

import config

def data_gen(mode = 'Train'):
    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    #import pdb;pdb.set_trace()
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    max_feat_tars = max_feat[:8,:].reshape(1,8,1,513)
    min_feat_tars = min_feat[:8,:].reshape(1,8,1,513)

    max_feat_ins = max_feat[-2:,:].reshape(1,2,1,513)
    min_feat_ins = min_feat[-2:,:].reshape(1,2,1,513)
    if mode == "Train":
        in_dir=config.dir_hdf5
        num_batches = config.batches_per_epoch_train
    elif mode =="Val":
        in_dir = config.dir_hdf5_test
        num_batches = config.batches_per_epoch_val

    sources = ['voc_stft', 'drums_stft', 'bass_stft', 'acc_stft']
    
    file_list = [x for x in os.listdir(in_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    max_files_to_process = int(config.batch_size/config.samples_per_file)


    num_files = len(file_list)

    for k in range(num_batches):

        inputs = []
        targets = []

        #start_time = time.time()

        for i in range(max_files_to_process):

            file_index = np.random.randint(0,num_files)
            
            file_to_open = file_list[file_index]

            hdf5_file = h5py.File(in_dir+file_to_open, "r")

            tar_stft = hdf5_file["tar_stft"]

            mix_stft = hdf5_file['mix_stft']



            file_len = mix_stft.shape[1]
            # start_time = time.time()
            for j in range(config.samples_per_file):
                index=np.random.randint(0,file_len-config.max_phr_len)
                # targets.append(np.concatenate((voc_stft[:,index:index+config.max_phr_len,:],drums_stft[:,index:index+config.max_phr_len,:],bass_stft[:,index:index+config.max_phr_len,:],acc_stft[:,index:index+config.max_phr_len,:]),axis=0))

                targets.append(tar_stft[:,index:index+config.max_phr_len,:])
                inputs.append(mix_stft[:,index:index+config.max_phr_len,:])
            hdf5_file.close()

            # print("One file took %0.00f" % (time.time()-start_time))
        #import pdb;pdb.set_trace()
        targets = (np.array(targets)-min_feat_tars)/(max_feat_tars-min_feat_tars)
        inputs = (np.array(inputs)-min_feat_ins)/(max_feat_ins-min_feat_ins)
        yield inputs, targets
            
            # p = np.random.random_sample()
            
            # # Randomize which batches are augmentated
            # if config.data_aug is True and p < -1:
            #     #print ('random')
            #     # each sample is a different file
            #     for j in range(config.samples_per_file):
                    
            #         # Random file for each source
            #         file_index = [np.random.randint(0,num_files) for x in range(4)]
                    
            #         source_i = 0
                    
            #         mix_stft = []
            #         mix_stft = np.ndarray(mix_stft)
                    
            #         sources_stft = []
                    
            #         for source in sources:
            #             file_to_open = file_list[file_index[source_i]]
                        
            #             hdf5_file = h5py.File(in_dir+file_to_open, "r")
                        
            #             source_stft = hdf5_file[source]
            #             file_len = source_stft.shape[1]
                        
            #             # random stft time index
            #             index=np.random.randint(0,file_len-config.max_phr_len)
                        
            #             source_stft = source_stft[:,index:index+config.max_phr_len,:]
                        
            #             if source_i == 0:
            #                 sources_stft = source_stft
                            
            #                 # this might be wrong, but I think it still makes sense:
            #                 mix_stft = source_stft/4
                            
            #             else:
            #                 sources_stft = np.concatenate((sources_stft, source_stft),axis=0)
                            
            #                 mix_stft += source_stft/4
                            
            #             source_i += 1
                        
            #         targets.append(sources_stft)
            #         inputs.append(mix_stft)

            # else:

                
                # Normalize data 
        #         for j in range(config.samples_per_file):
        #             index=np.random.randint(0,file_len-config.max_phr_len)
        #             stfts = [ voc_stft[:,index:index+config.max_phr_len,:],\
        #                       drums_stft[:,index:index+config.max_phr_len,:],\
        #                       bass_stft[:,index:index+config.max_phr_len,:],\
        #                       acc_stft[:,index:index+config.max_phr_len,:],\
        #                       mix_stft[:,index:index+config.max_phr_len,:]]   
        #             source_index = 0
                    
        #             for source_stft in stfts:
        #                 stft_max = np.amax(source_stft[:,:,:], axis = 1)
                        
        #                 if np.amax(stft_max) is not 0:
        #                     stft_max[stft_max == 0] = 0.0001 # for the indexes where the maximum is still zero
        #                     stft_norm = [source_stft[:,x,:] / stft_max for x in range(config.max_phr_len)]
        #                     stft_norm = np.array(stft_norm).view().reshape((2, 30, 513))
        #                 else:
        #                     # in the case where all the stft is made of zeros
        #                     stft_norm = source_stft
                        
        #                 #print(np.array(stft_norm).shape)
        #                 #print (np.amax(np.array(stft_norm)))
        #                 if source_index < 4:
        #                     if source_index == 0:
        #                         stft_stream = np.array(source_stft)
        #                     else:
        #                         stft_stream = np.concatenate((stft_stream, stft_norm), axis = 0)
        #                 else:
        #                     inputs.append(stft_norm)
        #                 source_index += 1
        #             targets.append(stft_stream)
                
        # #print(time.time()-start_time)
        # # normalization
        # '''
        # max_inp = np.amax(inputs,axis = 0)
        # print (max_inp.shape)
        # '''
        # # print (np.array(targets).shape)
        # #print (np.array(inputs).shape)
        # yield np.array(inputs), np.array(targets)


    
    #import pdb;pdb.set_trace()

def get_stats():
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

    #import pdb;pdb.set_trace()

    hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [10,513], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [10,513], np.float32)   
    hdf5_file["feats_maximus"][:] = maximus.reshape(10,513)
    hdf5_file["feats_minimus"][:] = minimus.reshape(10,513)
    
def main():
    # get_stats(feat='feats')
    gen = data_gen()
    start_time = time.time()
    for inp, tar in gen:
        print(time.time()-start_time)


        import pdb;pdb.set_trace()
        start_time = time.time()
    # vg = val_generator()
    # gen = get_batches()


    #import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
