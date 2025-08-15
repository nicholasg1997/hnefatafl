import numpy as np
import pytorch_lightning as pl
import torch

from hnefatafl.agents.agent import Agent
from hnefatafl.core.gameTypes import Player


def softmax(x)-> np.ndarray:
    """
    Compute the softmax of a vector.

    :param x: Input array or vector.
    :return: The softmax-transformed vector as a NumPy array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move, encoder):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.encoder = encoder

        legal_moves = state.get_legal_moves()
        self.legal_move_indices = [self.encoder.encode_move(m) for m in legal_moves]
        self.move_to_idx = {idx: i for i, idx in enumerate(self.legal_move_indices)}
        self.priors = np.array([priors[m] for m in legal_moves], dtype=np.float32)
        self.visit_counts = np.zeros(len(legal_moves), dtype=int)
        self.total_values = np.zeros(len(legal_moves), dtype=float)
        self.children = [None] * len(legal_moves)
        self.moves_list = [self.encoder.decode_move_index(idx) for idx in self.legal_move_indices]

    def moves(self):
        return self.moves_list

    def add_child(self, move, child_node):
        move_idx = self.encoder.encode_move(move)
        idx = self.move_to_idx[move_idx]
        self.children[idx] = child_node

    def has_child(self, move):
        move_idx = self.encoder.encode_move(move)
        return move_idx in self.move_to_idx and self.children[self.move_to_idx[move_idx]] is not None

    def get_child(self, move):
        move_idx = self.encoder.encode_move(move)
        if move_idx not in self.move_to_idx:
            return None
        idx = self.move_to_idx[move_idx]
        return self.children[idx]

    def record_visit(self, move, value):
        move_idx = self.encoder.encode_move(move)
        idx = self.move_to_idx[move_idx]
        self.total_visit_count += 1
        self.visit_counts[idx] += 1
        self.total_values[idx] += value

    def visit_count(self, move):
        move_idx = self.encoder.encode_move(move)
        if move_idx in self.move_to_idx:
            idx = self.move_to_idx[move_idx]
            return self.visit_counts[idx]
        return 0

    def is_leaf(self):
        return all(child is None for child in self.children)



class ZeroAgent(Agent):
    """
    A ZeroAgent class for implementing Monte Carlo Tree Search (MCTS) based decision-making
    with functions to train and evaluate moves within a game's decision tree.

    This class is designed to represent an intelligent agent that uses a neural network model
    and MCTS for decision-making in games. The agent utilizes techniques such as priors,
    state caching, value computation, temperature-based exploration, and Dirichlet noise
    integration for effective exploration and exploitation during training and gameplay.

    The agent supports setting up an experience collector, clearing the state cache,
    constructing decision trees, and selecting optimal moves based on MCTS.

    :ivar model: Reference to the neural network model utilized for move priors and value prediction.
    :ivar encoder: Encoder used for converting game state and moves into tensor representations.
    :ivar num_rounds: Number of rounds of tree search to perform for each move selection.
    :ivar c: Exploration constant for balancing exploitation and exploration in tree search.
    :ivar collector: Experience collector used for collecting gameplay data during training. Initialized as None.
    :ivar device: Device (e.g., CPU or GPU) where computations are performed.
    """
    def __init__(self, model, encoder, rounds_per_move=1600, c=3.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.collector = None

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.state_cache = {}

        self.device = next(self.model.parameters()).device
        #print(f"Using device: {self.device}")


    def set_collector(self, collector):
        """
        Set the experience collector for this agent.
        :param collector:
        :return: None
        """
        self.collector = collector

    def clear_cache(self):
        """
        Clears the state cache used for storing previously computed priors and values.
        :return: None
        """
        self.state_cache = {}

    def select_branch(self, node):
        if not node.legal_move_indices:
            print("No legal moves available.")
            return None
        total_n = node.total_visit_count
        sqrt_total_n = np.sqrt(total_n)

        with np.errstate(divide='ignore', invalid='ignore'):
            q_values = np.where(node.visit_counts > 0, node.total_values / node.visit_counts, 0.0)

        scores = q_values + self.c * node.priors * sqrt_total_n / (1 + node.visit_counts)

        best_idx = np.argmax(scores)
        best_move_idx = node.legal_move_indices[best_idx]
        return node.encoder.decode_move_index(best_move_idx)

    def create_node(self, game_state, move=None, parent=None):
        state_hash = hash(str(game_state.board.grid.tobytes()) + str(game_state.next_player))

        if state_hash in self.state_cache:
            priors, value = self.state_cache[state_hash]
        else:
            state_tensor = self.encoder.encode(game_state)
            model_input = torch.tensor(np.array([state_tensor]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                raw_priors, values = self.model(model_input)
            #priors = softmax(raw_priors[0].detach().numpy())
            priors = torch.softmax(raw_priors[0], dim=0).detach().cpu().numpy()
            value = values[0][0]
            self.state_cache[state_hash] = (priors, value)

        legal_moves = game_state.get_legal_moves()
        legal_moves_mask = np.zeros_like(priors, dtype=bool)
        for m in legal_moves:
            legal_moves_mask[self.encoder.encode_move(m)] = 1

        masked_priors = priors * legal_moves_mask

        if np.sum(masked_priors) > 0:
            masked_priors /= np.sum(masked_priors)
        else:
            masked_priors[legal_moves_mask] = 1.0 / np.sum(legal_moves_mask)

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(masked_priors)
            if legal_moves_mask[idx]
        }

        new_node = ZeroTreeNode(
            game_state,
            value,
            move_priors,
            parent,
            move,
            self.encoder
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_move(self, game_state, temperature=1.0, add_noise=False):
        # self.clear_cache()
        root = self.create_node(game_state)

        if add_noise and self.dirichlet_alpha > 0:
            if root.legal_move_indices:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.legal_move_indices))
                root.priors = (1 - self.dirichlet_epsilon) * root.priors + self.dirichlet_epsilon * noise

        for _ in range(self.num_rounds):
            node = root
            path = []

            while True:  # work down the tree until we hit a leaf node
                if not node.moves():
                    print("No legal moves available, breaking out of the loop.")
                    break

                move = self.select_branch(node)

                if node.has_child(move):
                    path.append((node, move))
                    node = node.get_child(move)
                else:
                    break

            parent_node = node

            if parent_node.state.is_over():  # Reached a terminal state
                player_at_terminal_node = parent_node.state.next_player.other
                winner = parent_node.state.winner
                if winner is None:
                    value = 0.0
                elif winner == player_at_terminal_node:
                    value = 1.0
                else:
                    value = -1.0

                #print(f"Move {parent_node.state.move_count}: Terminal state, winner={winner}, move_limit_hit={parent_node.state.move_limit_hit}, current_player={current_player}, value={value}, path={[(n.state.move_count, m) for n, m in path]}")

            else:
                new_state = parent_node.state.apply_move(move)
                child_node = self.create_node(new_state, move=move, parent=parent_node)
                path.append((parent_node, move))
                value = -child_node.value
                #print(f"Move {parent_node.state.move_count}: Expanded to child, value={value}, path={[(n.state.move_count, m) for n, m in path]}")

            for path_node, path_move in reversed(path):
                path_node.record_visit(path_move, value)
                value = -value

        if self.collector is not None:
            visit_counts = np.zeros(self.encoder.num_moves(), dtype=np.float32)
            for idx, count in zip(root.legal_move_indices, root.visit_counts):
                visit_counts[idx] = count
            encoded_state = self.encoder.encode(game_state)
            self.collector.record_decision(encoded_state, visit_counts)

        if not root.moves():
            print("No legal moves available, returning None.")
            return None

        if temperature == 0:
            return max(root.moves(), key=lambda m: root.visit_count(m))
        else:
            moves = root.moves()
            visit_counts = np.array([root.visit_count(m) for m in moves], dtype=np.float32)

            if np.sum(visit_counts) == 0:
                print(
                    f"Warning: All visit counts are zero, returning random move. move: {parent_node.state.move_count + 1}")
                probs = np.ones_like(visit_counts) / len(visit_counts)
            else:
                probs = visit_counts ** (1 / temperature)
                probs /= np.sum(probs)
            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]

    def train(self, experience, batch_size, epochs):
        # TODO: add early stopping, etc.
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, dataloader)
