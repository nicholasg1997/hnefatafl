import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ACModel(nn.Module):
    def __init__(self, encoder):
        super(ACModel, self).__init__()
        self.encoder = encoder
        num_input_channels = encoder.num_planes
        num_actions = encoder.num_moves()

        self.conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        conv_output_size = 64 * encoder.board_height * encoder.board_width

        self.dense_common = nn.Linear(conv_output_size, 512)

        self.policy_hidden = nn.Linear(512, 512)
        self.policy_output = nn.Linear(512, num_actions)

        self.value_hidden = nn.Linear(512, 512)
        self.value_output = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) # Flatten the tensor

        x = F.relu(self.dense_common(x))

        policy_x = F.relu(self.policy_hidden(x))
        policy_logits = self.policy_output(policy_x)

        value_x = F.relu(self.value_hidden(x))
        value_est = torch.tanh(self.value_output(value_x))

        return policy_logits, value_est




