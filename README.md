# PytorchConvSep

A Deep Learning approach for signal separation. Presented as the final degree project with the title "Understanding and Improving Current Deep Learning Source Separation Algorithms", this repository contains all of the final implementation of the project. The project's objective was to improve an existing and established state-of-the-art algorithm, by adding various improvements later explained in this document. 

This repository includes all the resources used and created from the project, except for the original dataset, which is later linked to its external provider.

you can read the final thesis here: [Understanding and Improving Deep Learning Source Separation Algorithms, Joan Grau](https://github.com/joangro/PytorchConvSep/blob/master/TFG_Grau_Noel_%20Joan.pdf)

## Baseline model (_DeepConvSep_)

The original _DeepConvSep_ framework proposes and implements a CNN (Convolutional Neural Network) approach to obtain a mask-based separation of various instruments from an audio mixture. Aswell, it includes an autoencoder on its architecture as a middle step, used to obtain compressed representations of data whixh helps to achieve a better result and avoid overfitting. This in turn improves its performance, which makes the algorithm convenient for low-latency applications. 

The overall architecture of the network is represented in the following figure:

![Oldframework](https://i.imgur.com/2GnEfAv.png)

The forward step of the algorithm goes from left to right, taking [STFT's](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) (Short-time Fourier transform) as the input data of the network. 

The STFT size is set at 30 time units -exactly 5ms- and has a frequency bandwidth of 513 units, as the STFT transform was made with a size of 1024 units.

The algorithm combines CNN with an autoencoder by doing two convolutions sequentially, first vertically, used to obtain representations of the _frequency_ information of the STFT, and then taking this output and sending it to the horizontal convolutional layer, which uses this features to model the time-frequency characteristics of the input.

The generated convolved output is then sent into a fully connected layer, with 128 units, greatly reducing the number of inputs, otherwise known as a bottleneck layer, common in autoencoders. This step is done to avoid overfitting and to reduce the number of features of the data into a smaller number, while keeping the most robust inputs to represent the data.

The next step is the decoding step, which sends the previously encoded output into four different branches, one for each  source, which independently deconvolute and reconstruct the STFT for each different source independently, which is then compared to the original to calculate the error and start the backpropagation step.

For more details, please see the [DeepConvSep repository](https://github.com/MTG/DeepConvSep/blob/master/README.md) and the following article: [ [2] (_Monoaural Audio Source Separation Using Deep Convolutional Neural Networks_, P.Chandna)](http://mtg.upf.edu/node/3680). 

## Dataset (_MusDB_ [3])

The dataset chosen in this project is the _MusDB_ dataset which can be freely downloaded on its  [(official site)](https://sigsep.github.io/datasets/musdb.html). This dataset contains a total of 150 different profesionally-mixed songs.

100 of those songs are used in training and 50 used in evaluation. These files are encoded in a STEM format which is a multi-track format encoded in _.mp4_ (a lossy format in contrast to the original DSD100 dataset which used loseless _.wav_ format).

The files have a sampling frequency of 44.1 kHz and have an average duration of 236 ± 95 seconds. 

The dataset was treated by using STFT's of a size of 1024 bins and a Hanning window with an overlap of 75% (5ms).

## PytorchConvSep Improvements

- The first improvement made was to update the framework from the old discontinued _lasagne_ into a currently supported and updated one. Pytorch was chosen, which is a deep-learning and tensor-based framework.

![pytorch-logo](https://cdn-images-1.medium.com/max/1200/1*KKADWARPMxHb-WMxCgW_xA.png)

- The architecture was further upgraded in order to support and work with _stereophonic_ signals. This also meant taht the architecture had to be slightly modified, as seen in the following figure:

![Newframework](https://i.imgur.com/VuB3T5q.png)

- Data transformations were applied to perform data augmentation: random STFT creation and mixing, channel muting and random song mixing in the same batch.

## Instructions

🚨 Note: Python 3.6.X or higher versions are needed to run the code and its dependencies! 🚨

Install packages:
```
pip install torch torchvision
pip install h5py stempeg
pip install numpy matplotlib scipy
pip install mir_eval
```

- Important files:

  - _PytorchConvSep.py_: main file of the algorithm, implements all of the main functions such as the network architecture, the training and the evaluation methods of the algorithm.

  - _data_pipeline.py_: file controling and processing the data feeding into the algorithm during the training step. Change with caution.

  - _evalNet.py_: MIR evaluation tools used to measure the quality of the audio separation.

  - **_config.py_**: configuration file with the paths for the training and evaluation step of the network. Change according to the absolute path where the STEM files are located.

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
python3 PytorchConvSep.py <filename> --plot
```

Plot file results:
```
python3 PytorchConvSep.py <filename> --plot --ns
```

Further help by using:
```
python3 PytorchConvSep.py --help
```

IMPORTANT NOTE: The files to separate must be in STEM format, but only with the standard two stereophonic channels, please see the [original STEM website](https://www.stems-music.com/stem-creator-tool/) for information on how to convert files to this format.

## Contributors

- Pritish Chandna (pritish.chandna@upf.edu), PhD Student in Music Information Research Lab (Universitat Pompeu Fabra) 

- Joan Grau (joan.grau01@estudiant.upf.edu), university graduate in Audiovisual Systems Engineering (Universitat Pompeu Fabra) 

## Presented Thesis

Read the thesis here: [Understanding and Improving Deep Learning Source Separation Algorithms, Joan Grau](https://github.com/joangro/PytorchConvSep/blob/master/TFG_Grau_Noel_%20Joan.pdf)

## References

[2] P. Chandna, M. Miron, J. Janer, and E. Gomez, “Monoaural audio source separation using deep convolutional neural networks” International Conference on Latent Variable Analysis and Signal Separation, 2017.

[3] Zafar  Rafii,  Antoine  Liutkus,  Fabian-Robert  Stöter,  Stylianos  Ioannis  Mimilakis,  and Rachel Bittner. The MUSDB18 corpus for music separation, December 2017.
