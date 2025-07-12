import numpy as np

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
            if state.is_valid_move(move): # not sure that this will work but im not sure that its neccesary as we mask out illegal moves
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


class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.num_rounds = rounds_per_move
        self.c = c

    def select_move(self, game_state):
        root = self.create_node(game_state)
        pass

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (1 + n)

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

    def train(self, experience, learning_rate, batch_size):
        # This method should implement the training logic for the model
        # using the collected experience. use PyTorch Lightning

        pass