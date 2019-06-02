import neptune
import torch
import numpy as np
import os
import wave
from torch.utils.data import Dataset, DataLoader

import pandas as pd


neptune.init(project_qualified_name='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/freesound-audio-tagging')
neptune.create_experiment()

train_curated = pd.read_csv("./input/train_curated.csv")
train_noisy = pd.read_csv("./input/train_noisy.csv")


class Config():
    def __init__(self, num_of_epochs=5, learning_rate=0.001, batch_size=16):
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


class read_sound_file():
    def __call__(self, sample):
        file_name = sample
        # jest legit
        return file_name[0]

class SoundDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=read_sound_file()):
        data = pd.read_csv(csv_file)
        self.data = data.values
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if(self.transform):
            sample = self.transform(sample)
        
        return sample

    
config = Config()
dataset = SoundDataset("./input/train_curated.csv", '.')

train_loader = DataLoader(dataset, batch_size=config.batch_size)
for i, x in enumerate(train_loader):
    if(i <= 1):
        print(x)
neptune.stop()