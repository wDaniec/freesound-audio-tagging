import neptune
import torch
import numpy as np
import librosa
import os
import wave
from torch.utils.data import Dataset, DataLoader

import pandas as pd


neptune.init(project_qualified_name='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/freesound-audio-tagging')
neptune.create_experiment()

train_curated = pd.read_csv("./input/train_curated.csv")
train_noisy = pd.read_csv("./input/train_noisy.csv")


class Config():
    def __init__(self, num_of_epochs=5, learning_rate=0.001, batch_size=16, audio_duration=4, sampling_rate=16000):
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.audio_duration = audio_duration
        self.sampling_rate = sampling_rate
        self.audio_length = self.sampling_rate * self.audio_duration


class read_sound_file():

    def __init__(self, config):
        self.config = config


    def __call__(self, sample):
        file_path = sample[0]
        data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')
        input_length = self.config.audio_length
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        

        labels = sample[1].split(',')
        return torch.from_numpy(data), labels

class SoundDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        data = pd.read_csv(csv_file)
        self.data = data.values
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
     

    
config = Config()
dataset = SoundDataset("./input/train_curated.csv", './input/train_curated/', read_sound_file(config))

train_loader = DataLoader(dataset, batch_size=config.batch_size)
for i, A in enumerate(train_loader):
    print(A)
    x, y = A
    if(i <= 1):
        print(x[0])
        print(y)
        print(y[0])
neptune.stop()