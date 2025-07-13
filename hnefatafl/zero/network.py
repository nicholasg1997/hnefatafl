import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class DualNetwork(pl.LightningModule):
    def __init__(self, encoder, learning_rate=0.001):
        super(DualNetwork, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate

        input_shape = self.encoder.shape()
        num_input_channels = input_shape[0]

        self.conv_body = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * input_shape[1] * input_shape[2], self.encoder.num_moves()),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * input_shape[1] * input_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.save_hyperparameters('learning_rate')

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        shared_out = self.conv_body(x)
        policy_out = self.policy_head(shared_out)
        value_out = self.value_head(shared_out)
        return policy_out, value_out

    def training_step(self, batch, batch_idx):
        states, policy_targets, value_targets = batch
        policy_out, value_out = self(states)
        value_preds = value_out.squeeze(dim=1)
        policy_loss = nn.CrossEntropyLoss()(policy_out, policy_targets)
        value_loss = nn.MSELoss()(value_preds, value_targets)
        loss = policy_loss + value_loss

        self.log('policy_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('value_loss', value_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self, optimizer='adam'):
        if optimizer.lower() == 'adam':
            return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif optimizer.lower() == 'sgd':
            return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Use 'adam' or 'sgd'.")