# filter warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# import libraries
import os
import h5py
import argparse
import numpy as np
import scipy.io as io
import cPickle as pickle
import matplotlib.pyplot as plt
from collections import namedtuple

# import utilities
import img_utils

np.random.seed(1)

def loadmat(path,fname,ttype):
    if ttype == 'expert':
        mat_out = io.loadmat('%s/%s' % (path, fname))
    else:
        mat_out = {}
        f = h5py.File('%s/%s' % (path, fname), 'r')
        for k, v in f.items():
            mat_out[k] = np.array(v)
    return mat_out

def main():

    # named tuple to record demonstrations
    Step = namedtuple('Step','cur_state action next_state reward done')

    # argument parser for command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--data', default='data', type=str,
                        help='path argument for dataset')
    args = parser.parse_args()

    # create dataset dictionary
    dataset = {}

    # create list of trajectory types and data types
    traj_types = ['random', 'expert']
    data_types = ['state', 'action']

    # loop over folders to load the dataset:
    for traj_type in traj_types:
        print(traj_type)

        # assign keys in dictionary
        dataset[traj_type] = {}
        
        # parse all files in folder
        stateFolder = traj_type+'_state'
        statePath = '%s/%s' % (args.data, stateFolder)
        files = os.listdir(statePath)

        actionFolder = traj_type+'_action'
        actionPath = '%s/%s' % (args.data, actionFolder)
        
        for data_type in data_types:
            dataset[traj_type][data_type] = []

        # loop over the matfiles
        for fname in files:
            # load the datasets to a list
            for data_type in data_types:
                if data_type == 'state':
                    mat_out = loadmat(statePath,fname,traj_type)
                    dat = mat_out['procposPool'].astype(np.int16)
                    dataset[traj_type][data_type].append(dat)
                else:
                    mat_out = loadmat(actionPath,fname[:-4]+'_act.mat',traj_type)
                    dat = mat_out['proc_actPool'].astype(np.int16)
                    dataset[traj_type][data_type].append(dat)

    # save all the dataset as a pickle file
    pickle.dump(dataset, open('%s/data.p' % (args.data),'wb'))
        
if __name__=="__main__":
    main()