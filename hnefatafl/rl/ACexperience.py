import numpy as np
from hnefatafl.core.move import Move

def combine_experience(collectors):
    combined_states = np.concatenate(
        [np.array(c.states) for c in collectors]
    )

    combined_actions = np.concatenate(
        [np.array(c.actions) for c in collectors]
    )

    combined_rewards = np.concatenate(
        [np.array(c.rewards) for c in collectors]
    )

    combined_advantages = np.concatenate(
        [np.array(c.advantages) for c in collectors]
    )

    return ExperienceBuffer(
        states=combined_states,
        actions=combined_actions,
        rewards=combined_rewards,
        advantages=combined_advantages
    )


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)

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
        self.advantages = []

        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self.current_episode_states.append(state)
        encoded_action = action.encode() if isinstance(action, Move) else action
        self.current_episode_actions.append(encoded_action)
        self.current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states.extend(self.current_episode_states)
        self.actions.extend(self.current_episode_actions)
        self.rewards.extend([reward] * len(self.current_episode_states))

        for i in range(num_states):
            advantage = reward - self.current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self.begin_episode()

    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            advantages=np.array(self.advantages)
        )