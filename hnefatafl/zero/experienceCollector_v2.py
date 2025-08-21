import numpy as np
from collections import deque
import torch
from pytorch_lightning.callbacks import Callback
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
        policy_targets = np.array(self.visit_counts) / (visit_sums + 1e-8)  # add a small number to avoid 0 div

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
        policy_targets = np.array(self.visit_counts) / (visit_sums + 1e-8)  # Add small number to avoid 0 division error

        #TODO: select moves that result in a win for black more often
        priorities = np.zeros(len(self.states), dtype=np.float32)
        black_win_mask = (self.rewards >= 0.5) & (self.is_result)
        white_win_mask = (self.rewards <= -0.5) & (self.is_result)
        draw_mask = ~self.is_result | (np.abs(self.rewards) < 0.5)
        priorities[black_win_mask] = 1.5
        priorities[white_win_mask] = 0.7
        priorities[draw_mask] = 0.3

        priorities = np.power(priorities, priority_alpha)
        priorities /= np.sum(priorities + 1e-8)

        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        policy_tensor = torch.tensor(policy_targets, dtype=torch.float32)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)

        dataset = TensorDataset(states_tensor, policy_tensor, rewards_tensor)
        sampler = WeightedRandomSampler(priorities, len(priorities), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    def __len__(self):
        return len(self.states)

class PersistentExperienceBuffer:
    def __init__(self, max_games=500_000):
        self.states = deque(maxlen=max_games)
        self.visit_counts = deque(maxlen=max_games)
        self.rewards = deque(maxlen=max_games)
        self.is_result = deque(maxlen=max_games)

    def add_experience(self, collectors):
        for collector in collectors:
            self.states.extend(collector.states)
            self.visit_counts.extend(collector.visit_counts)
            self.rewards.extend(collector.rewards)
            self.is_result.extend(collector.is_result)

    def get_training_buffer(self):
        return ZeroExperienceBuffer(
            states=np.array(self.states),
            visit_counts=np.array(self.visit_counts),
            rewards=np.array(self.rewards),
            is_result=np.array(self.is_result)
        )

    def clear_states(self):
        self.states.clear()
        self.visit_counts.clear()
        self.rewards.clear()
        self.is_result.clear()

    def get_states(self):
        return {
            'states': list(self.states),
            'visit_counts': list(self.visit_counts),
            'rewards': list(self.rewards),
            'is_result': list(self.is_result)
        }

    def set_states(self, load_states):
        self.clear_states()

        self.states.extend(load_states['states'])
        self.visit_counts.extend(load_states['visit_counts'])
        self.rewards.extend(load_states['rewards'])
        self.is_result.extend(load_states['is_result'])

    def __len__(self):
        return len(self.states)


class ReplayBufferCheckpoint(Callback):
    def __init__(self, buffer: PersistentExperienceBuffer):
        super().__init__()
        self.buffer = buffer
        self.buffer_key = 'experience'

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print('Saving experience buffer...')
        checkpoint[self.buffer_key] = self.buffer.get_states()
        print("Experience buffer saved.")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        print('Loading experience buffer...')
        buffer_states = checkpoint.get(self.buffer_key, None)
        if buffer_states:
            self.buffer.set_states(buffer_states)
            print("Experience buffer loaded.")
        else:
            print("No experience buffer found in checkpoint, starting with an empty buffer.")


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