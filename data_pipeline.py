import numpy as np
import os
import time
import h5py

import config

def data_gen(mode = 'Train'):
    if mode == "Train":
        in_dir=config.dir_hdf5
    elif mode =="Val":
        in_dir = config.dir_hdf5_test

    sources = ['voc_stft', 'drums_stft', 'bass_stft', 'acc_stft']
    
    file_list = [x for x in os.listdir(in_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    max_files_to_process = int(config.batch_size/config.samples_per_file)


    num_files = len(file_list)

    for k in range(config.batches_per_epoch_train):

        inputs = []
        targets = []

        #start_time = time.time()

        for i in range(max_files_to_process):
            
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

            # print("One file took %0.00f" % (time.time()-start_time))
        targets = np.array(targets)
        inputs = np.array(inputs)
        yield inputs, targets
                
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
