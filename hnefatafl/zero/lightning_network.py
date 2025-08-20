import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class DualNetwork(pl.LightningModule):
    def __init__(self, encoder, learning_rate=0.001, batch_size=128):
        super(DualNetwork, self).__init__()
        self.save_hyperparameters('learning_rate', 'batch_size')
        self.encoder = encoder
        self.current_generation = 0

        input_shape = self.encoder.get_shape()
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
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * input_shape[1] * input_shape[2], self.encoder.num_moves()),
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

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        shared_out = self.conv_body(x)
        policy_out = self.policy_head(shared_out)
        value_out = self.value_head(shared_out)
        return policy_out, value_out

    def training_step(self, batch, batch_idx):
        # This is where the model learns from a single batch of data.
        states, policy_targets, value_targets = batch
        policy_out, value_out = self(states)

        value_preds = value_out.squeeze(dim=1)
        policy_loss = torch.nn.functional.cross_entropy(policy_out, policy_targets)
        value_loss = torch.nn.functional.mse_loss(value_preds, value_targets)
        loss = policy_loss + value_loss

        self.log_dict({
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'total_loss': loss
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self, optimizer='adam'):
        if optimizer.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif optimizer.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Use 'adam' or 'sgd'.")

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_generation'] = self.current_generation
        if hasattr(self, 'wandb_logger') and self.wandb_logger.experiment.id:
            checkpoint['wandb_run_id'] = self.wandb_logger.experiment.id

    def on_load_checkpoint(self, checkpoint):
        self.current_generation = checkpoint.get('current_generation', 0)
        print(f"[Checkpoint] Loaded current_generation = {self.current_generation}")
