import glob
import random
import numpy as np 
import torch
from scipy.io.wavfile import read
from sklearn import preprocessing
import sys
import pdb

class GetSequence:

    def __init__(self):
        self.all_files = list(glob.iglob('data/*.wav'))
        self.split_at = 5000
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    def get_sequnce(self):
        while True:
            temp = 0
            temp_file = random.choice(self.all_files)
            value_array = read(temp_file)
            value_array = value_array[1]
            self.min_max_scaler.fit(value_array.astype(float))
            value_array = self.min_max_scaler.transform(value_array)
            # value_array = value_array/ np.linalg.norm(value_array)
            value_array = np.delete(value_array, 1)
            seq = value_array.shape
            while temp + self.split_at + 1 < seq[0]:
                in_val = torch.from_numpy(value_array[temp: temp+ self.split_at])
                # lab_val = torch.from_numpy(value_array[(temp * self.split_at) + self.diff : ((temp+1) * self.split_at) + self.diff])
                lab_val = torch.tensor([value_array[(temp + self.split_at + 1)]])
                temp += 1
                yield in_val, lab_val
                
    def get_train_data(self):
        in_lab , lab_val = next(self.get_sequnce())
        return in_lab, lab_val
