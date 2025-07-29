import numpy as np
import pytorch_lightning as pl
import torch

from hnefatafl.agents.agent import Agent

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
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            self.branches[move] = Branch(p)
        self.children = {}

    def moves(self) -> list:
        """Return a list of legal moves available from this node."""
        return list(self.branches.keys())

    def add_child(self, move, child_node):
        """Add a child node for the given move."""
        self.children[move] = child_node

    def has_child(self, move) -> bool:
        """Check if a child node exists for the given move."""
        return move in self.children

    def get_child(self, move) -> 'ZeroTreeNode':
        """Get the child node for the given move."""
        return self.children.get(move, None)

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches.get(move, None)
        if branch is None or branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        branch = self.branches.get(move, None)
        if branch is None:
            return 0.0
        return branch.prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def is_leaf(self):
        return len(self.children) == 0


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
        """
        Selects the best branch from the available moves based on a scoring function.

        The scoring function combines the expected value of the move (Q), the prior
        probability (P), and the visit counts, adjusted by the total visits to
        the node. This function is used to determine the most promising move
        to explore further in a decision-making or game-playing algorithm.

        If no moves are available, the function will notify the user and return None.

        :param node: The current node for which to select the best move.
        :return: The move with the highest score among the legal moves available
            in the current node, or None if no moves are available.
        """
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (1 + n)

        if not node.moves():
            print("No legal moves available.")
            return None
        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, move=None, parent=None):
        """
        Creates a new node for the tree structure using the game state, move, and parent node.

        This function either retrieves node priors and value from a cache or computes them
        using an encoder and the model. From the computed or cached priors, a legal
        move mask is applied and normalized priors for legal moves are determined.
        The function then creates a new tree node and optionally links it to a parent node.

        :param game_state: The current state of the game as passed to the node creation.
        :type game_state: GameState
        :param move: The move that leads to this state. Defaults to None.
        :type move: Optional[Move]
        :param parent: The parent node in the tree structure. Defaults to None.
        :type parent: Optional[ZeroTreeNode]
        :return: The newly created node with computed priors, value, and game state.
        :rtype: ZeroTreeNode
        """
        # --- CACHE ---
        state_hash = hash(str(game_state.board.grid.tobytes()) + str(game_state.next_player))

        if state_hash in self.state_cache:
            priors, value = self.state_cache[state_hash]
        else:
            state_tensor = self.encoder.encode(game_state)
            model_input = torch.tensor(np.array([state_tensor]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                raw_priors, values = self.model(model_input)
            priors = softmax(raw_priors[0].cpu().detach().numpy())
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

        #move_priors = {self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors) }
        new_node = ZeroTreeNode(
            game_state,
            value,
            move_priors,
            parent,
            move
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_move(self, game_state, temperature=1.0, add_noise=False):
        """
        Selects a move for the given game state using a Monte Carlo Tree Search (MCTS) process. This function utilizes
        MCTS to compute and select the best move based on several rounds of simulation, taking into account
        factors such as visit counts, value propagation through paths in the tree, and optional Dirichlet noise for
        introducing exploration. The selected move corresponds to the optimal policy under the current search parameters.

        :param game_state: The current state of the game from which the move is to be selected. It allows access to
            legal moves, applying a move to generate a new state, and determining if the game has reached a terminal state.
        :param temperature: A parameter to control exploration versus exploitation during move selection.
            Temperature of 0 results in greedy deterministic move selection, while higher values increase exploration.
            Defaults to 1.0 for balanced exploration.
        :param add_noise: Boolean indicating whether to add Dirichlet noise to the root policy for increased exploration.
            Defaults to False. This is particularly useful during self-play to diversify learning.
        :return: The selected move based on the MCTS process. Returns None if there are no legal moves available.
        :rtype: Optional object corresponding to a valid move in the game's representation.
        """
        #self.clear_cache()
        root = self.create_node(game_state)

        if add_noise and self.dirichlet_alpha > 0:
            moves = root.moves()
            if moves:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
                for i, move in enumerate(moves):
                    root.branches[move].prior = ((1 - self.dirichlet_epsilon) * root.branches[move].prior +
                                                 self.dirichlet_epsilon * noise[i])

        for _ in range(self.num_rounds):
            node = root
            path = []

            while True: # work down the tree until we hit a leaf node
                if not node.moves():
                    print("No legal moves available, breaking out of the loop.")
                    break

                move = self.select_branch(node)

                if node.has_child(move):
                    path.append((node, move))
                    node = node.get_child(move)
                else:
                    break
            #print(f"path length: {len(path)}")

            parent_node = node

            if parent_node.state.is_over():  # we have reached a terminal state
                #print(f"Terminal state reached: {parent_node.state.winner}")
                current_player = parent_node.state.next_player.other
                winner = parent_node.state.winner
                if winner == current_player: # Win
                    value = 1.0
                elif winner == current_player.other:  # Loss
                    value = -1.0
                else: # Draw
                    if parent_node.state.repeating_player == current_player:
                        value = -0.5
                    else:
                        value = 0.0
            else:
                new_state = parent_node.state.apply_move(move)
                child_node = self.create_node(new_state, move=move, parent=parent_node)
                value  = -child_node.value

            for path_node, path_move in reversed(path):
                #print(f"Recording visit for move {path_move} with value {value}")
                path_node.record_visit(path_move, value)
                value = -value

            #print(f"path length: {len(path)}")
            #print(f"tree depth: {len(root.branches)}")

        if self.collector is not None:
            visit_counts = np.zeros(self.encoder.num_moves())
            for move, branch in root.branches.items():
                move_idx = self.encoder.encode_move(move)
                visit_counts[move_idx] = branch.visit_count
            encoded_state = self.encoder.encode(game_state)
            self.collector.record_decision(encoded_state, visit_counts)

        if not root.moves():
            print("No legal moves available, returning None.")
            return None

        if temperature == 0:
            #print(f"path length: {len(root.branches)}")
            return max(root.moves(), key=lambda m: root.visit_count(m))
        else:
            moves = root.moves()
            visit_counts = np.array([root.visit_count(m) for m in moves], dtype=np.float32)

            if np.sum(visit_counts) == 0:
                print("Warning: All visit counts are zero, returning random move.")
                probs = np.ones_like(visit_counts)/len(visit_counts)
            else:
                probs = visit_counts ** (1 / temperature)
                probs /= np.sum(probs)
            #print(f"probs: {probs}")

            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]

    def train(self, experience, batch_size, epochs):
        # TODO: add early stopping, etc.
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, dataloader)
