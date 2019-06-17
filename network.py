import neptune
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import pandas as pd


neptune.init(project_qualified_name='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/freesound-audio-tagging')
neptune.create_experiment()


class Config():
    def __init__(self, num_of_epochs=5, learning_rate=0.001, batch_size=128, audio_duration=4, sampling_rate=16000, val_freq=0.2):
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
        print(data.shape)
        return data, labels

class MockTransform():

    def __init__(self, config):
        self.config = config
    
    def __call__(self, sample):
        return torch.randn(1, config.audio_length), torch.randn(80)

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


def train_model(config):
    train_data = SoundDataset("./input/train_curated_small.csv", './input/curated_data/', MockTransform(config))
    val_data = SoundDataset("./input/val_curated.csv", "./input/curated_data/", MockTransform(config))
    test_data = SoundDataset("./input/test_curated.csv", "./input/curated_data/", ReadSoundFile(config))

    print(len(test_data), len(val_data), len(train_data))

    train_loader = DataLoader(train_data, batch_size=config.batch_size)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size = config.batch_size)

    model = Classifier(80).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate, amsgrad=False)
    best_epoch = -1

    for epoch in range(config.num_of_epochs):
        model.train()
        avg_train_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            print(i, ' | ', len(train_loader))
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda().float()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_train_loss += loss.item() / len(train_loader)

        model.eval()

        avg_val_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(val_loader):
            print('validation set: ', i, ' | ', len(val_loader))
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda().float()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            avg_val_loss += loss.item() / len(val_loader)
        
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f} | avg_val_loss: {avg_val_loss:.4f}')


config = Config()

train_model(config)
        
neptune.stop()