import neptune
import torch
import numpy as np
import os


import pandas as pd


neptune.init(project_qualified_name='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/freesound-audio-tagging')
neptune.create_experiment()

train_curated = pd.read_csv("./input/train_curated.csv")
train_noisy = pd.read_csv("./input/train_noisy.csv")
print('udalo sie?')

neptune.stop()