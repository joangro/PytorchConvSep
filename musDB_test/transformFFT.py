"""
    Adaptation from https://github.com/MTG/DeepConvSep/blob/master/transform.py
    Original header:
    
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Affero GPL License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """
import numpy as np



def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)

    Computes a \"sinebell\" window function of length L=lengthWindow

    The formula is:

    .. math::

        window(t) = sin(\pi \\frac{t}{L}), t=0..L-1

    """
    window = np.sin((np.pi*(np.arange(lengthWindow)))/(1.0*lengthWindow))
    return window
    
    

        
def calculateFFT( data, window=sinebell(2048),hopsize=256.0, nfft=2048.0, fs=44100.0):
    if data.shape[1] == 2:
        data = (data[:,0] + data[:,1])/2
        
    lengthWindow = window.size
    lengthData = data.size
    numberFrames = int(np.ceil(lengthData / np.double(hopsize)) + 2)
    newLengthData = int((numberFrames-1) * hopsize + lengthWindow)
    data = np.concatenate((np.zeros(int(lengthWindow/2.0)), data))
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))
    numberFrequencies = int(nfft / 2 + 1)

    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)

    for n in np.arange(numberFrames):
        beginFrame = int(n*hopsize)
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft))
        frameToProcess = None

    
    # care
    return np.real(STFT.T)
