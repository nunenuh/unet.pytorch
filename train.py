import os
import random
import pathlib
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm

from unet.data import loader


root = '/data/idcard/results/combined/segmentv3/20kv1/'
trainloader, trainset = loader.pair_data_loader(root_path=root, mode='train', batch_size=16)
validloader, validset = loader.pair_data_loader(root_path=root, mode='valid', batch_size=16)

from unet.models.unet import UNet
from unet.trainer.task import TaskUNet

model = UNet(in_chan=1, n_classes=1, start_feat=32)
model_state_dict = torch.load('weights/unet_sfeat32_v3.pth')
model.load_state_dict(model_state_dict)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
task = TaskUNet(model, optimizer, criterion)


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# DEFAULTS used by the Trainer
SAVED_CHECKPOINT_PATH = 'checkpoints/'
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=SAVED_CHECKPOINT_PATH,
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='unetv3'
)

SAVED_LOGS_PATH = 'logs/'
tb_logger = pl_loggers.TensorBoardLogger(SAVED_LOGS_PATH)

trainer = pl.Trainer(
    weights_summary="top",
    max_epochs=10,
    val_check_interval=100,
    gpus=-1,
    logger=tb_logger, 
    checkpoint_callback=checkpoint_callback, 
)

trainer.fit(task, trainloader, validloader)