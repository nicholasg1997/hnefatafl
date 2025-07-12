import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_visit_counts = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self.current_episode_states.append(state)
        self.current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states.extend(self.current_episode_states)
        self.visit_counts.extend(self.current_episode_visit_counts)
        self.rewards.extend([reward] * num_states)
        self.begin_episode()

    def get_dataloader(self, batch_size:int = 64):
        visit_sums = np.sum(self.visit_counts, axis=1, keepdims=True)
        policy_targets = np.array(self.visit_counts) / (visit_sums + 1e-8)  # add very small number to avoid 0 div
        

class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset(
            'states', data=self.states)
        h5file['experience'].create_dataset(
            'visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset(
            'rewards', data=self.rewards)

    def __len__(self):
        return len(self.states)

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ZeroExperienceBuffer(
        states=combined_states,
        visit_counts=combined_visit_counts,
        rewards=combined_rewards
    )

def load_experience(h5file):
    return ZeroExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        visit_counts=np.array(h5file['experience']['visit_counts']),
        rewards=np.array(h5file['experience']['rewards'])
    )