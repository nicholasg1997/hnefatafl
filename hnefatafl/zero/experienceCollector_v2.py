import numpy as np
from collections import deque
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self.is_result = []
        self.current_episode_states = []
        self.current_episode_visit_counts = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self.current_episode_states.append(state)
        self.current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward, is_result):
        num_states = len(self.current_episode_states)
        self.states.extend(self.current_episode_states)
        self.visit_counts.extend(self.current_episode_visit_counts)
        self.rewards.extend([reward] * num_states)
        self.is_result.extend([is_result] * num_states)
        self.begin_episode()

    def get_dataloader(self, batch_size:int = 64):
        visit_sums = np.sum(self.visit_counts, axis=1, keepdims=True)
        policy_targets = np.array(self.visit_counts) / (visit_sums + 1e-8)  # add very small number to avoid 0 div

        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        policy_tensor = torch.tensor(policy_targets, dtype=torch.float32)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)

        dataset = TensorDataset(states_tensor, policy_tensor, rewards_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards, is_result):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards
        self.is_result = is_result

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('is_result', data=self.is_result)

    def get_dataloader(self, batch_size:int = 64, priority_alpha= 0.6):
        visit_sums = np.sum(self.visit_counts, axis=1, keepdims=True)
        policy_targets = np.array(self.visit_counts) / (visit_sums + 1e-8)  # add very small number to avoid 0 division error

        priorities = np.where(self.is_result, 1.0, 0.1)  # Higher priority for terminal states
        priorities = priorities ** priority_alpha
        priorities /= np.sum(priorities)

        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        policy_tensor = torch.tensor(policy_targets, dtype=torch.float32)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)

        dataset = TensorDataset(states_tensor, policy_tensor, rewards_tensor)
        sampler = WeightedRandomSampler(priorities, len(priorities), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    def __len__(self):
        return len(self.states)

class PersistentExperienceBuffer:
    def __init__(self, max_games=1000):
        self.max_games = max_games
        self.games = deque(maxlen=max_games)

    def add_game(self, game):
        self.games.append(game)

    def get_all_experience(self):
        all_states = np.concatenate([game[0] for game in self.games])
        all_visit_counts = np.concatenate([game[1] for game in self.games])
        all_rewards = np.concatenate([game[2] for game in self.games])
        all_is_result = np.concatenate([game[3] for game in self.games])
        return ZeroExperienceBuffer(
            states=all_states,
            visit_counts=all_visit_counts,
            rewards=all_rewards,
            is_result=all_is_result
        )


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_is_result = np.concatenate([np.array(c.is_result) for c in collectors])

    return ZeroExperienceBuffer(
        states=combined_states,
        visit_counts=combined_visit_counts,
        rewards=combined_rewards,
        is_result=combined_is_result
    )

def load_experience(h5file):
    return ZeroExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        visit_counts=np.array(h5file['experience']['visit_counts']),
        rewards=np.array(h5file['experience']['rewards']),
        is_result=np.array(h5file['experience']['is_result'])
    )