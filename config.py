import numpy as np
# import tensorflow as tf

wav_dir_train = '../../../../datasets/musdb18/train/'

wav_dir_test = '../../../../datasets/musdb18/test/'

out_dir = './outputs/'

err_dir = './error/'
data_dir = '/data/'
log_dir = './log/'
dn_log_dir = './dn_log/'
data_log = './log/data_log.log'
data_aug = False

dir_hdf5 = '../../data_h5py/'
dir_hdf5_test = '../../data_h5py_test/'
stat_dir = './stats/'
h5py_file_train = './data_h5py/train.hdf5'
h5py_file_val = './data_h5py/val.hdf5'
val_dir = './val_dir/'

in_mode = 'mix'
norm_mode_out = "max_min"
norm_mode_in = "max_min"

alpha = 0.001
beta  = 0.01
beta_voc = 0.03

max_len = 3939892
channels = 2
features = 513

split = 0.9

# Hyperparameters
num_epochs = 8000
batches_per_epoch_train = 50
batches_per_epoch_val = 50
batch_size = 5
samples_per_file = 1
max_phr_len = 30
input_features = 513
lstm_size = 128
output_features = 66
highway_layers = 4
highway_units = 128
init_lr = 1
num_conv_layers = 8
conv_filters = 128
num_ch_out_hor = 50
num_ch_out_ver = 30
# conv_activation = tf.nn.relu
dropout_rate = 0.0
projection_size = 3
fs = 44100
comp_mode = 'mfsc'
dn_num_epochs = 4000
print_every = 1
save_every = 10

