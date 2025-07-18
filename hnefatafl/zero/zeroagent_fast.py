import numpy as np
import pytorch_lightning as pl
import torch
from sympy.physics.units import temperature

from hnefatafl.agents.agent import Agent


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

    def moves(self):
        return list(self.branches.keys())

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
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
    def __init__(self, model, encoder, rounds_per_move=1600, mcts_batch_size=8, c=4.0):
        assert mcts_batch_size <= rounds_per_move, "MCTS batch size must be less than or equal to rounds per move."

        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.collector = None
        self.mcts_batch_size = mcts_batch_size

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            total_n = node.total_visit_count

            exploration = self.c * p * np.sqrt(total_n) / (1 + n)
            return q + exploration

        if not node.moves():
            return None
        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        device = next(self.model.parameters()).device
        model_input = torch.from_numpy(np.array([state_tensor])).float().to(device)

        with torch.no_grad():
            priors, values = self.model(model_input)

        priors = priors.cpu().numpy()[0]
        value = values.cpu().numpy()[0][0]

        legal_moves = game_state.get_legal_moves()
        legal_moves_mask = np.zeros_like(priors, dtype=bool)
        for m in legal_moves:
            legal_moves_mask[self.encoder.encode_move(m)] = True

        masked_priors = priors * legal_moves_mask
        sum_masked_priors = np.sum(masked_priors)
        if sum_masked_priors > 0:
            masked_priors /= sum_masked_priors
        else:
            masked_priors = legal_moves_mask / np.sum(legal_moves_mask)

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(masked_priors)
            if legal_moves_mask[idx]
        }

        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_move(self, game_state, temp=1.0):
        root = self.create_node(game_state)

        num_loops = max(1, self.num_rounds // self.mcts_batch_size)
        for _ in range(num_loops):
            batch_leaves = []
            for _ in range(self.mcts_batch_size):
                node = root
                next_move = self.select_branch(node)
                while next_move is not None and node.has_child(next_move):
                    node = node.get_child(next_move)
                    next_move = self.select_branch(node)
                batch_leaves.append((node, next_move))

            game_states = [leaf.state.apply_move(move) for leaf, move in batch_leaves if move is not None]
            if not game_states:
                continue

            state_tensors = [self.encoder.encode(gs) for gs in game_states]
            device = next(self.model.parameters()).device
            model_input = torch.from_numpy(np.array(state_tensors)).float().to(device)

            with torch.no_grad():
                priors_batch, values_batch = self.model(model_input)

            priors_batch = priors_batch.cpu().numpy()
            values_batch = values_batch.cpu().numpy()

            leaf_idx = 0
            for (parent_node, move), new_game_state in zip(batch_leaves, game_states):
                if move is None:
                    continue

                priors, value_estimate = priors_batch[leaf_idx], values_batch[leaf_idx][0]
                leaf_idx += 1

                if new_game_state.is_over():
                    if new_game_state.winner == new_game_state.next_player.other:
                        value = 1.0
                    elif new_game_state.winner is None:
                        value = 0.0
                    else:
                        value = -1.0
                else:
                    value = value_estimate

                legal_moves = new_game_state.get_legal_moves()
                legal_moves_mask = np.zeros_like(priors, dtype=bool)
                for m in legal_moves:
                    legal_moves_mask[self.encoder.encode_move(m)] = True

                masked_priors = priors * legal_moves_mask
                sum_masked_priors = np.sum(masked_priors)
                if sum_masked_priors > 0:
                    masked_priors /= sum_masked_priors
                else:
                    if np.sum(legal_moves_mask) > 0:
                        masked_priors = legal_moves_mask / np.sum(legal_moves_mask)

                move_priors = {
                    self.encoder.decode_move_index(idx): p
                    for idx, p in enumerate(masked_priors) if legal_moves_mask[idx]
                }

                child_node = ZeroTreeNode(new_game_state, value, move_priors, parent_node, move)
                parent_node.add_child(move, child_node)

                bp_value = -child_node.value
                temp_node = parent_node
                bp_move = move
                while temp_node is not None:
                    temp_node.record_visit(bp_move, bp_value)
                    bp_move = temp_node.last_move
                    temp_node = temp_node.parent
                    bp_value = -bp_value


        if not root.moves():
            return None


        if temp > 0:
            visit_counts = np.array([root.visit_count(m) for m in root.moves()])
            probs = visit_counts ** (1 / temp)
            probs /= np.sum(probs)
            move_idx = np.random.choice(len(root.moves()), p=probs)
            print(len(root.moves()))
            selected_move = list(root.moves())[move_idx]
        else:
            selected_move = max(root.moves(), key=lambda m: root.visit_count(m))

        if self.collector is not None:
            visit_counts = np.zeros(self.encoder.num_moves())
            for move, branch in root.branches.items():
                move_idx = self.encoder.encode_move(move)
                visit_counts[move_idx] = branch.visit_count
            encoded_state = self.encoder.encode(game_state)
            self.collector.record_decision(encoded_state, visit_counts)

        return selected_move

    def train(self, experience, batch_size, epochs):
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs, accelerator="auto")
        trainer.fit(self.model, dataloader)