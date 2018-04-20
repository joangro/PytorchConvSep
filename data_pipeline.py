import numpy as np
import os
import time
import h5py

import config

def data_gen(in_dir=config.dir_hdf5):

    file_list = [x for x in os.listdir(in_dir) if x.endswith('.hdf5') and not x.startswith('._')]

    max_files_to_process = int(config.batch_size/config.samples_per_file)

    num_files = len(file_list)

    for k in range(config.batches_per_epoch_train):

        inputs = []
        targets = []

        # start_time = time.time()

        for i in range(max_files_to_process):
            file_index = np.random.randint(0,num_files)
            file_to_open = file_list[file_index]

            hdf5_file = h5py.File(in_dir+file_to_open, "r")

            voc_stft = hdf5_file['voc_stft']

            mix_stft = hdf5_file['mix_stft']

            drums_stft = hdf5_file['drums_stft']

            bass_stft = hdf5_file['bass_stft']

            acc_stft = hdf5_file['acc_stft']

            file_len = voc_stft.shape[1]

            for j in range(config.samples_per_file):
                    index=np.random.randint(0,file_len-config.max_phr_len)
                    targets.append(np.concatenate((voc_stft[:,index:index+config.max_phr_len,:],drums_stft[:,index:index+config.max_phr_len,:],bass_stft[:,index:index+config.max_phr_len,:],acc_stft[:,index:index+config.max_phr_len,:]),axis=0))
                    inputs.append(mix_stft[:,index:index+config.max_phr_len,:])
        targets = np.array(targets)
        inputs = np.array(inputs)
        yield inputs, targets
    # print(time.time()-start_time)
    # import pdb;pdb.set_trace()



def main():
    # get_stats(feat='feats')
    gen = data_gen()
    # vg = val_generator()
    # gen = get_batches()


    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()