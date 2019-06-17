import neptune
import torch
import numpy as np
import librosa
import os
import wave
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


import utils
import pandas as pd


neptune.init(project_qualified_name='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/freesound-audio-tagging')
neptune.create_experiment()

train_curated = pd.read_csv("./input/train_curated.csv")
train_noisy = pd.read_csv("./input/train_noisy.csv")


class Config():
    def __init__(self, num_of_epochs=5, learning_rate=0.001, batch_size=16, audio_duration=4, sampling_rate=16000, val_freq=0.2):
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.audio_duration = audio_duration
        self.sampling_rate = sampling_rate
        self.audio_length = self.sampling_rate * self.audio_duration
        self.val_freq = val_freq


class ReadSoundFile():

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
        labels = utils.multi_hot_embedding(labels)
        data = torch.from_numpy(data)
        data = data.view(1, -1)
        return data, labels

class SoundDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        data = pd.read_csv(csv_file)
        self.data = data.values
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample[0] = self.root_dir + sample[0]
        if(self.transform):
            sample = self.transform(sample)
        return sample


config = Config()
train_data = SoundDataset("./input/train_curated_small.csv", './input/curated_data/', ReadSoundFile(config))
val_data = SoundDataset("./input/val_curated.csv", "./input/curated_data/", ReadSoundFile(config))
test_data = SoundDataset("./input/test_curated.csv", "./input/curated_data/", ReadSoundFile(config))

print(len(test_data), len(val_data), len(train_data))

train_loader = DataLoader(train_data, batch_size=config.batch_size)
val_loader = DataLoader(val_data, batch_size=config.batch_size)
test_loader = DataLoader(test_data, batch_size = config.batch_size)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.MaxPool1d(pool_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()


        self.conv = nn.Sequential(
            ConvBlock(1, 16, 9, 16),
            ConvBlock(16, 32, 3, 4),
            ConvBlock(32, 32, 3, 4),
            ConvBlock(32, 256, 3, 4)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1028),
            nn.ReLU(),
            nn.Linear(1028, num_classes)
        )


    def forward(self, x):
        x = self.conv(x)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x


model = Classifier(80)

for i, (x, y) in enumerate(val_loader):
    if(i <= 1):
        print(model(x).shape)
        for instance in y:
            print(utils.convert_to_labels(instance))
        
neptune.stop()