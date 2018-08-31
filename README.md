# PytorchConvSep

A Deep Learning approach for signal separation. Presented as the final degree project with the title "Understanding and Improving Current Deep Learning Source Separation Algorithms", this repository contains all of the technical aspects of the project. The project's objective was to improve upon an already established state-of-the-art algorithm, by adding various improvements later explained. This repository only contains the python code and some examples, does not include the original dataset.


## Baseline model (_DeepConvSep_)

Read more about it in [DeepConvSep repository](https://github.com/MTG/DeepConvSep/blob/master/README.md) and in the following article: [ [2](_Monoaural Audio Source Separation Using Deep Convolutional Neural Networks_, P.Chandna)](http://mtg.upf.edu/node/3680). 

The original _DeepConvSep_ framework proposes and implements a CNN (Convolutional Neural Network) approach in order to obtain a mask-based approach in order to separate various instruments from an audio mixture. Aswell, it includes an autoencoder architecture in order to obtain compressed representations of data in order to achieve a better result and avoid overfitting, plus making the algorithm convenient for low-latency applications, due to its good performance. 

The overall architecture of the net can be seen in the following figure:

![Oldframework](https://i.imgur.com/2GnEfAv.png)

Where the forward step of the algorithm goes left to right, taking STFT's as the features of the network. (Standarized at 30 time bins -5ms- and with a size of 513 frequency bins -due to the STFT transform being made with a size of 1024 bins).

The algorithm combines CNN with an autoencoder by doing two convolutions sequentially, first vertically, in order to obtain representations of the _frequency_ information of the STFT, and then taking this output and sending it to the horizontal convolutional layer, which uses this features in order to model the time-frequency characteristics of the input.

The generated convolved output is then sent into a fully connected layer, with 128 units, greatly reducing the number of inputs in order to create a bottleneck layer, common in autoencoders. This step is done in order to avoid overfitting and to reduce the number of features of the data into a smaller number while keeping the most robust inputs to represent the data.

The next step is the decoding step, which sends the previously encoded output into four different branches, one for each different source, which  independently deconvolute and reconstruct the STFT for each different source independently, which is then compared to the original to calculate the error and backpropagate.

## Dataset (_MusDB_ [3])

The dataset chosen to be used in this project is the _MusDB_ dataset which can be consulted and freely downloaded in the following [(link)](https://sigsep.github.io/datasets/musdb.html). This dataset contains a total of 150 different profesionally-mixed songs, 100 of which are distributed in training and 50 in evaluation. These files are encoded in a STEM format which is a multi-track format encoded in _.mp4_ (a lossy format in contrast to the original DSD100 dataset which used loseless _.wav_ format).

The files have a sampling frequency of 44.1 kHz and have an average duration of 236 ± 95 seconds. 

The dataset was treated by using STFT's of a size of 1024 bins and a Hanning window with an overlap of 75% (5ms).

## PytorchConvSep Improvements

- The first improvement made was to update the framework from the old discontinued _lasagne_ into a current supported and updated one, from which Pytorch was chosen, which is a deep-learning and tensor-based framework.

![pytorch-logo](https://pytorch.org/static/img/logos/pytorch-logo-dark.png)

- The architecture was further upgraded in order to support and work with _stereophonic_ signals, for which the new architecture can be depicted in the following figure:

![Newframework](https://i.imgur.com/VuB3T5q.png)

- Data transformations were applied in order to do data augmentation, random STFT creation and mixing, channel muting and random song mixing in the same batch are implemented.

## Instructions

🚨 ATTENTION: Only works in versions of Python 3.6 or higher! 🚨

Install packages:
```
pip3 install torch torchvision
pip3 install h5py stempeg
pip3 install numpy matplotlib scipy
pip3 install mir_eval
```

- Important files:

- _PytorchConvSep.py_: main file of the algorithm, implements all of the main functions such as the network architecture, the training and the evaluation methods of the algorithm.

- _data_pipeline.py_: file controling and processing the data feeding into the algorithm during the training step. Not recommended to change.

- _evalNet.py_: MIR evaluation tools used to measure the quality of the audio separation. Not recommended to be used.

- **_config.py_**: configuration file with the paths for the training and evaluation step of the network. Change accord to the path where the STEM files are located locally.

- Running the algorithm:

(The current release still doesn't support training and evaluating into fully different releases, but they can be run separately by using the next commands)

Train the model by using the following command, the first argument <optional_model> allows the user to load the current network with an already trained model and keep training it.

```
python3 PytorchConvSep.py --train <optional_model>
```

Evaluate and generate files from the network by using the following commands:

Synthesize file:
```
python3 PytorchConvSep.py <filename>
```

Plot and synthesize file:
```
python3 PytorchConvSep.py <filename> -- plot
```

Plot file results:
```
python3 PytorchConvSep.py <filename> -- plot --ns
```

Further help by using:
```
python3 PytorchConvSep.py --help
```

IMPORTANT NOTE: The files to separate must be in STEM format (but only with the standard two stereophonic channels, please see the [original STEM website](https://www.stems-music.com/stem-creator-tool/) for information on how to convert files to this format.

## Contributors

- Pritish Chandna (pritish.chandna@upf.edu), PhD Student in Music Information Research Lab (Universitat Pompeu Fabra) 

- Joan Grau (joan.grau01@estudiant.upf.edu), graduated Audiovisual Systems Engineering student (Universitat Pompeu Fabra) 

## Presented Thesis

Incoming (still not uploaded at university repository as of 08/2018)

## References

[2] P. Chandna, M. Miron, J. Janer, and E. Gomez, “Monoaural audio source separation using deep convolutional neural networks” International Conference on Latent Variable Analysis and Signal Separation, 2017.

[3] Zafar  Rafii,  Antoine  Liutkus,  Fabian-Robert  Stöter,  Stylianos  Ioannis  Mimilakis,  and Rachel Bittner. The MUSDB18 corpus for music separation, December 2017.
