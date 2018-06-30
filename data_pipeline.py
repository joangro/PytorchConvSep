import numpy as np
import os
import time
import h5py

import config

def data_gen(mode = 'Train', data_aug = False):
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

    sources = range(4)
    
    file_list = [x for x in os.listdir(in_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    max_files_to_process = int(config.batch_size/config.samples_per_file)


    num_files = len(file_list)

    for k in range(num_batches):

        inputs = []
        targets = []

        #start_time = time.time()

        for i in range(max_files_to_process):
        
            if data_aug is True:
                #p = np.random.random_sample()
                p = 0.1
                print (p)
                if p < 0.4:
                    mix_stft = []
                    print ("IEE")
                    for source in sources:
                        print (source)
                        file_index = np.random.randint(0,num_files)
                        file_to_open = file_list[file_index]

                        hdf5_file = h5py.File(in_dir+file_to_open, "r")

                        source_stft = hdf5_file["tar_stft"]
                        
                        file_len = source_stft.shape[1]
                                                
                        source_stft_c = source_stft[source*2:source*2+2,:,:]
                        for j in range(config.samples_per_file):
                            index=np.random.randint(0,file_len-config.max_phr_len)
                            source_stft_f = source_stft_c[:,index:index+config.max_phr_len,:]
                            
                        if source is 0:
                            mix_stft = source_stft_f
                            targets_all = source_stft_f
                        else:
                            mix_stft += source_stft_f
                            targets_all =np.concatenate((targets_all,source_stft_f), axis = 0)
                        
                    targets.append(targets_all)
                    inputs.append(mix_stft)
                #yield inputs, targets

                    
            else:
                file_index = np.random.randint(0,num_files)
                
                file_to_open = file_list[file_index]

                hdf5_file = h5py.File(in_dir+file_to_open, "r")

                tar_stft = hdf5_file["tar_stft"]

                mix_stft = hdf5_file['mix_stft']



                file_len = mix_stft.shape[1]
                # start_time = time.time()
                for j in range(config.samples_per_file):
                    flag = False
                    while flag is False:
                        index=np.random.randint(0,file_len-config.max_phr_len)#;print ('small')
                    # targets.append(np.concatenate((voc_stft[:,index:index+config.max_phr_len,:],drums_stft[:,index:index+config.max_phr_len,:],bass_stft[:,index:index+config.max_phr_len,:],acc_stft[:,index:index+config.max_phr_len,:]),axis=0))
                        #import pdb;pdb.set_trace()
                        if mix_stft[:,index:index+config.max_phr_len,:425].mean() > 0.02:
                            targets.append(tar_stft[:,index:index+config.max_phr_len,:])
                            inputs.append(mix_stft[:,index:index+config.max_phr_len,:])
                            flag = True
                hdf5_file.close()

                # print("One file took %0.00f" % (time.time()-start_time))
            #import pdb;pdb.set_trace()
            targets_norm = (np.array(targets)-min_feat_tars)/(max_feat_tars-min_feat_tars)
            inputs_norm = (np.array(inputs)-min_feat_ins)/(max_feat_ins-min_feat_ins)
            #yield inputs, targets
        yield inputs_norm, targets_norm
            
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
    gen = data_gen(data_aug = True)
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
