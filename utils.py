import sys
import os,re
import collections
import csv
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt

import stempeg

import config

def stft(data, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow

    # import pdb;pdb.set_trace()
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data

    # import pdb;pdb.set_trace()
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    
    # zero-padding data such that it holds an exact number of frames

    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = n*hopsize
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[int(beginFrame):int(endFrame)]
        STFT[int(n),:] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")
        
    # frequency and time stamps:
    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT

def istft(mag, phase, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0,
          analysisWindow=None):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    """
    X = mag * np.exp(1j*phase)
    X = X.T
    if analysisWindow is None:
        analysisWindow = window

    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)

    normalisationSeq = np.zeros(lengthData)

    data = np.zeros(lengthData)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft), norm = 'ortho')
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)

    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.

    data = data / normalisationSeq

    return data

def stft_stereo(data, phase=False):
    assert data.shape[1] == 2
    if phase:
        stft_left = stft(data[:,0])
        stft_right = stft(data[:,1])
        return np.array([abs(stft_left),abs(stft_right)]),np.array([np.angle(stft_left),np.angle(stft_right)])
    else:
        stft_left = abs(stft(data[:,0]))
        stft_right = abs(stft(data[:,1]))
        return np.array([stft_left,stft_right])


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def file_to_stft(input_file):
    audio,fs=sf.read(input_file)
    mixture=np.clip(audio[:,0]+audio[:,1],0.0,1.0)
    mix_stft=abs(stft(mixture))
    return mix_stft




def generate_overlapadd(allmix,time_context=config.max_phr_len, overlap=config.max_phr_len/2,batch_size=config.batch_size):
    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    input_size = allmix.shape[-1]

    i=0
    start=0  
    while (start + time_context) < allmix.shape[1]:
        i = i + 1
        start = start - overlap + time_context 
    fbatch = np.zeros([int(np.ceil(float(i)/batch_size)),batch_size,2,time_context,input_size])+1e-10
    
    
    i=0
    start=0  

    while (start + time_context) < allmix.shape[1]:
        fbatch[int(i/batch_size),int(i%batch_size),:,:,:]=allmix[:,int(start):int(start+time_context),:]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    
    return fbatch,i

def overlapadd(fbatch,nchunks,overlap=int(config.max_phr_len/2)):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[1]


    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    

    sep = np.zeros((2,int(nchunks*(time_context-overlap)+time_context),input_size))

    # import pdb;pdb.set_trace()

    
    i=0
    start=0 
    while i < nchunks:
        # import pdb;pdb.set_trace()
        sa = fbatch[int(i/batch_size),int(i%batch_size),:,:,:]
        # import pdb;pdb.set_trace()

        #print s1.shape
        if start==0:
            sep[:,0:time_context,:] = sa

        else:
            #print start+overlap
            #print start+time_context
            sep[:,int(start+overlap):int(start+time_context),:] = sa[:,overlap:time_context]
            # import pdb;pdb.set_trace()
            sep[:,start:int(start+overlap),:] = window[overlap:]*sep[:,start:int(start+overlap),:] + window[:overlap]*sa[:,:overlap]
        i = i + 1 #index for each block
        start = int(start - overlap + time_context) #starting point for each block
    return sep  


def normalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        # import pdb;pdb.set_trace()
        outputs = (inputs-minimus)/(maximus-minimus)

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs-means)/stds
    elif mode == "clip":
        outputs = np.clip(inputs, 0.0,1.0)

    return outputs

def inverse_stft_write(mix_stft,mix_phase,file_name):
    audio_out_l = istft(mix_stft[0],mix_phase[0])

    audio_out_r = istft(mix_stft[1],mix_phase[1])

    audio_out = np.array([audio_out_l,audio_out_r]).T  

    sf.write(file_name,audio_out,config.fs)

def denormalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        # import pdb;pdb.set_trace()
        outputs = (inputs*(maximus-minimus))+minimus

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs*stds)+means
    return outputs

def main():
    lf = "Al James - Schoolboy Facination.stem.mp4"
    audio,fs = stempeg.read_stems(os.path.join(config.wav_dir_test,lf), stem_id=[0,1,2,3,4])

    mixture = audio[0]

    mix_stft, mix_phase = stft_stereo(mixture,phase=True)

    inverse_stft_write(mix_stft,mix_phase,'./test.wav')

    # audio_out_l = istft(mix_stft[0],mix_phase[0])

    # audio_out_r = istft(mix_stft[1],mix_phase[1])

    # audio_out = np.array([audio_out_l,audio_out_r]).T



    # sf.write('./test.wav',audio_out,fs)
    # test(harmy, 10*np.log10(harm))

    # test_sample = np.random.rand(5170,66)

    # fbatch,i = generate_overlapadd(test_sample)

    # sampled = overlapadd(fbatch,i)

    import pdb;pdb.set_trace()



if __name__ == '__main__':
    main()