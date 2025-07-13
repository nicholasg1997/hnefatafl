import numpy as np
import pytorch_lightning as pl

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
        legal_moves = state.get_legal_moves()
        print(f"legal moves avaliable: {len(legal_moves)}")
        for move, p in priors.items():
            if move in legal_moves: # not sure that this will work but im not sure that its neccesary as we mask out illegal moves
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
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.collector = None

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
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
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model(model_input)
        priors = priors[0] # will proably have to change this for working in PyTorch
        value = values[0][0] # will also have to change this
        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(
            game_state, value,
            move_priors, parent, move
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_move(self, game_state):
        root = self.create_node(game_state)

        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)

            while node.has_child(next_move):
                node = node.get_child(next_move)
                if not node.is_leaf():
                    next_move = self.select_branch(node)
                else:
                    break

            new_game_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_game_state, move=next_move, parent=node)

            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

        visit_counts = np.zeros(self.encoder.num_moves())
        for move, branch in root.branches.items():
            move_idx = self.encoder.encode_move(move)
            visit_counts[move_idx] = branch.visit_count

        #policy_targets = visit_counts / np.sum(visit_counts)

        if self.collector is not None:
            encoded_states = self.encoder.encode(game_state)
            self.collector.record_decision(encoded_states, visit_counts)

        best_move = max(root.moves(), key=root.visit_count)
        return best_move


    def train(self, experience, batch_size, epochs):
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, dataloader)