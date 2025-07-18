import numpy as np
from hnefatafl.core.move import Move


class ExperienceBuffer:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

    @classmethod
    def deserialize(cls, h5file):
        states = h5file['experience']['states'][:]
        actions = h5file['experience']['actions'][:]
        rewards = h5file['experience']['rewards'][:]
        return cls(states, actions, rewards)

class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        self.current_episode_states.append(state)
        encoded_action = action.encode() if isinstance(action, Move) else action
        self.current_episode_actions.append(encoded_action)

    def complete_episode(self, reward):
        self.states.extend(self.current_episode_states)
        self.actions.extend(self.current_episode_actions)
        self.rewards.extend([reward] * len(self.current_episode_states))
        self.begin_episode()

    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards)
        )
