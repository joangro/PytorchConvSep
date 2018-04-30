import musdb
import numpy as np
from transformFFT import calculateFFT
from PytorchConvSep import Network

class loadDatabase():
    '''
        Main object for reading data, training, testing and sending result
        
        IN: 
            -   directory   (string):       path where database is localized
            -   sources_in  (list):         list of string of the sources we want to train 
                                            (there are five by default, accompainment source is     
                                            discarded by default) 
    '''
    def __init__(self, directory = '../MUS-STEMS-SAMPLE',
                 sources_in = ['vocals', 'drums', 'bass', 'other']):
        self.directory = directory
        self.sources   = sources_in
        self.autoencoders = {}
        
    def loadTestTrack(self, track):
        '''
            Main function for testing (still to finish)
        '''
        track.audio
        
        track.path
        
        track.rate
        
        for source in track.targets:
            if source not in self.sources:
                continue

            estimates[source] = track.targets[str(source)].audio
            # call fft function
            fft_source = calculateFFT( estimates[source] )
            self.autoencoders[ source ].testNetwork( fft_source )
            
            
        return estimates
        
    def loadTrainTrack(self, track):
        '''
            Main function for training from a given list of tracks
        '''
        track.audio
        
        track.path
        
        track.rate
        
        estimates = {}

        
        # INIT AUTOENCODER OBJECTS FOR EACH SOURCE
        # we create a Network object, which contains the autoencoder, for each source
        #
        for source in self.sources:
            aux_source = Network(source)
            self.autoencoders[source] = aux_source
            
            
        # FEED TRAINING
        # Read each target (optimal separated source for each track)
        #   1. Calculate FFT for each target we are given (FULL TRACK FFT)
        #   2. Send FFT to train in each sources' autoencoder in Network
        #
        #   Note: I don't think this is very efficient
        #
        for source in track.targets:
            if source not in self.sources:
                continue

            estimates[source] = track.targets[str(source)].audio
            # call fft function
            fft_source = calculateFFT( estimates[source] )
            self.autoencoders[ source ].trainNetwork( fft_source )
          
        '''
        estimates = {
            'vocals':   track.targets['vocals'].audio,
            'drums':    track.targets['drums'].audio,
            'bass':     track.targets['bass'].audio,
            'other':    track.targets['other'].audio
        }
        '''
        return estimates
        
    
    def chooseSubset(self, subset):
        if subset is None:
            return mus.load_mus_tracks()
        if subset is 'train':
            return mus.load_mus_tracks(subsets = 'train')
        return mus.load_mus_tracks(subsets = 'test')
        

def loadSet(subset = 'train'):
    '''
        I did this so later we can directly call this function instead of running another piece of code
    '''
    # init objects
    mus = musdb.DB(root_dir = '../MUS-STEMS-SAMPLE')
    db = loadDatabase()
    
    ### choose subset 'train' or 'test' or both (don't send any argument) 
    track_list = db.chooseSubset(subset)
    
    #print 'track-list ', track_list
    # check it's correctly working
    #assert mus.test(db.loadTestTrack), "doesn't work" 
    if subset is 'train':
        tracks = mus.run(
            db.loadTrainTrack,
            estimates_dir='./Estimates',
            tracks = track_list,
            subsets = subset
        )
        for i in tracks:
            print i # why does this return None?
    else:
         tracks = mus.run(
            db.loadTestTrack,
            estimates_dir='./Estimates',
            tracks = track_list,
            subsets = subset
        )   


if __name__ == "__main__":
    '''
        Attempt at modulating a little the given STEM code.
    '''
    # init objects
    mus = musdb.DB(root_dir = '../MUS-STEMS-SAMPLE')
    db = loadDatabase()
    loadSet('train')
    #loadSet('test')
    


