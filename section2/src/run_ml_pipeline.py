"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
import numpy as np


class Config:
    """
    Holds configuration parameters
    """

    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"YOUR DIRECTORY HERE"
        self.n_epochs = 5
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "RESULTS GO HERE"


if __name__ == "__main__":
    # Get configuration
    c = Config()
    c.root_dir = "../data"
    c.test_results_dir = "../out/results/"

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(
        c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)

    # Create test-train-val split
    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like
    # a k-fold training and combining the results.

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation
    # and testing respectively.
    def splitdata(keys, train_ratio=0.6, val_ratio=0.2):
        keys = np.random.permutation(keys)
        train_ind = int(train_ratio*len(keys))
        val_ind = int((train_ratio+val_ratio)*len(keys))
        test_ind = int((1-train_ratio-val_ratio)*len(keys))
        return range(0, train_ind), range(train_ind, val_ind), range(val_ind, test_ind+val_ind)

    split['train'], split['val'], split['test'] = splitdata(keys)

    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
