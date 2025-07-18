import numpy as np
import pytorch_lightning as pl
import torch

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
    def __init__(self, model, encoder, rounds_per_move=1600, c=3.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.collector = None

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.state_cache = {}

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
        # --- CACHE ---
        state_hash = hash(str(game_state.board.grid.tobytes()) + str(game_state.next_player))

        if state_hash in self.state_cache:
            priors, value = self.state_cache[state_hash]
        else:
            state_tensor = self.encoder.encode(game_state)
            model_input = np.array([state_tensor])
            with torch.no_grad():
                priors, values = self.model(model_input)
            priors = priors[0]
            value = values[0][0]
            self.state_cache[state_hash] = (priors, value)
        # --------
        legal_moves = game_state.get_legal_moves()
        legal_moves_mask = np.zeros_like(priors.detach().numpy(), dtype=bool)
        for m in legal_moves:
            legal_moves_mask[self.encoder.encode_move(m)] = True

        masked_priors = priors.detach().numpy() * legal_moves_mask
        masked_priors /= np.sum(masked_priors)

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(masked_priors)
            if legal_moves_mask[idx]
        }

        # --------
        #move_priors = {self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors) }
        new_node = ZeroTreeNode(
            game_state, value,
            move_priors, parent, move
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_move(self, game_state, temperature=1.0, add_noise=False):
        self.state_cache = {}
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
            while not node.is_leaf(): # work down tree until we hit a leaf node
                move = self.select_branch(node)
                if move is None:
                    print("No legal moves available, breaking out of MCTS.")
                    break
                path.append((node, move))
                node = node.get_child(move)

            if node.state.is_over():  # we have reached a terminal state
                if node.state.winner == node.state.next_player: # Win
                    value = 1.0
                elif node.state.winner == node.state.next_player.other:
                    value = -1.0
                else:
                    value = 0.0
            else:
                value = self.expand_leaf_node(node)

            for path_node, path_move in reversed(path):
                path_node.record_visit(path_move, value)
                value = -1 * value # not sure if value swapping should go here or before recording node visit

        # --- Data collection for training ---
        if self.collector is not None:
            visit_counts = np.zeros(self.encoder.num_moves())
            for move, branch in root.branches.items():
                move_idx = self.encoder.encode_move(move)
                visit_counts[move_idx] = branch.visit_count
            encoded_state = self.encoder.encode(game_state)
            self.collector.record_decision(encoded_state, visit_counts)

        if temperature == 0:
            return max(root.moves(), key=lambda m: root.visit_count(m))
        else:
            moves = []
            visit_counts = []
            for move in root.moves():
                moves.append(move)
                visit_counts.append(root.visit_count(move))

            if not moves:
                return None

            visit_counts = np.array(visit_counts, dtype=np.float32)

            if np.sum(visit_counts) == 0:
                print("Warning: All visit counts are zero, returning random move.")
                probs = np.ones_like(visit_counts)/len(visit_counts)
            else:
                probs = visit_counts ** (1 / temperature)
                probs /= np.sum(probs)
            print(f"probs: {probs}")

            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]

    def expand_leaf_node(self, node, device='mps'):
        state_hash = hash(str(node.state.board.grid.tobytes()) + str(node.state.next_player))
        print(f"state_hash: {state_hash}")
        if state_hash in self.state_cache:
            priors, value = self.state_cache[state_hash]
            priors = priors.detach().cpu().numpy()
            print("Using cached priors and values.")
            print(f"priors: {priors}\nvalues: {value}")
        else:
            print("Calculating priors and values.")
            state_tensor = self.encoder.encode(node.state)
            model_input = torch.from_numpy(np.array([state_tensor])).float().to(device)
            with torch.no_grad():
                priors_tensor, values_tensor = self.model(model_input)
                print(f"priors: {priors_tensor}\nvalues: {values_tensor}")

            priors = priors_tensor[0].detach().cpu().numpy()
            value = values_tensor[0][0].detach().cpu().numpy()
            self.state_cache[state_hash] = (priors, value)

        legal_moves = node.state.get_legal_moves()
        legal_moves_mask = np.zeros_like(priors, dtype=bool)
        for m in legal_moves:
            legal_moves_mask[self.encoder.encode_move(m)] = 1

        masked_priors = priors * legal_moves_mask
        print(f"masked_priors: {masked_priors}")
        sum_masked_priors = np.sum(masked_priors)
        print(f"sum_masked_priors: {sum_masked_priors}")
        if sum_masked_priors > 0:
            masked_priors /= sum_masked_priors

        for move in legal_moves:
            move_idx = self.encoder.encode_move(move)
            prior_p = masked_priors[move_idx]
            node.branches[move] = Branch(prior_p)

        return value



    def train(self, experience, batch_size, epochs):
        # TODO: add early stopping, learning rate scheduling, etc.
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, dataloader)