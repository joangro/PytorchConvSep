import musdb
import numpy as np
import transform

class loadDatabase():
    def __init__(self, directory = '../MUS-STEMS-SAMPLE'):
        self.directory = directory
        
    def loadTrack(self, track):

        track.audio
        
        track.path
        
        track.rate
        
        estimates = {
            'track': track.audio,
        }
            
        return estimates
        
    def loadTrainTrack(self, track):
        track.audio
        
        track.path
        
        track.rate
        estimates = {}
        for source in track.targets:
            estimates[source] = track.targets[str(source)].audio
            # call fft function
        
        
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
            print i


if __name__ == "__main__":
    '''
        Attempt at modulating a little the given STEM code.
    '''
    # init objects
    mus = musdb.DB(root_dir = '../MUS-STEMS-SAMPLE')
    db = loadDatabase()
    loadSet('train')
    



        

