import numpy as np
import pytorch_lightning as pl
import torch

from hnefatafl.agents.agent import Agent


def softmax(x) -> np.ndarray:
    """Compute the softmax of a vector (kept for backward compatibility)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ZeroTreeNode:
    """
    A single MCTS node, stored with parallel arrays indexed by *legal-move position*.

    Everything the search needs (priors, visit counts, values, children) is aligned to
    the same index order as ``moves_list``, so the hot path navigates by integer index
    and never re-encodes/decodes a Move. ``legal_move_indices`` is kept only so the final
    visit-count vector can be scattered back into the full action space for the collector.
    """

    def __init__(self, state, value, legal_moves, legal_move_indices, priors, parent, last_move):
        self.state = state
        self.value = value                      # NN value, from this node's player-to-move perspective
        self.parent = parent
        self.last_move = last_move
        self.encoder_indices = legal_move_indices

        self.moves_list = legal_moves           # list[Move], aligned with the arrays below
        n = len(legal_moves)
        self.priors = priors                    # np.float32[n], masked + normalized
        self.visit_counts = np.zeros(n, dtype=np.int64)
        self.total_values = np.zeros(n, dtype=np.float64)
        self.children = [None] * n
        self.total_visit_count = 1

        # Cache terminality once; state.is_over() re-runs the (expensive) endgame checks
        # every call when the game is NOT over.
        self.is_terminal = state.is_over()


class ZeroAgent(Agent):
    """
    Batched AlphaZero-style MCTS agent.

    Each search "round" still corresponds to one simulation, but simulations are gathered
    into batches: we descend to ``batch_size`` distinct leaves (using virtual loss to keep
    the batch diverse), run a SINGLE network forward pass over all of them, then expand and
    back-propagate. Batching is what makes a GPU/MPS device worthwhile -- one forward over 32
    boards instead of 32 forwards over one board.

    The model should already be on the desired device (CPU / cuda / mps); the agent reads the
    device from the model's parameters and moves encoded states there for inference only.
    """

    def __init__(self, model, encoder, rounds_per_move=800, c=1.5,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 batch_size=32, virtual_loss=1.0):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.collector = None

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss

        self.state_cache = {}
        self.device = next(self.model.parameters()).device

    def set_collector(self, collector):
        self.collector = collector

    def clear_cache(self):
        self.state_cache = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _state_hash(game_state):
        return hash(game_state.board.grid.tobytes() + bytes([game_state.next_player.value]))

    def _evaluate(self, states):
        """
        Run the network on a list of GameStates and return a list of (priors, value) pairs,
        where ``priors`` is the full-action-space softmax (np.float32) and ``value`` is a float.

        De-duplicates within the batch and caches across the whole search, so the (single)
        batched forward pass only sees states it has never scored before.
        """
        results = [None] * len(states)
        unique_states = []
        unique_targets = []      # parallel to unique_states: list of result-indices sharing that state
        seen = {}

        for i, s in enumerate(states):
            h = self._state_hash(s)
            cached = self.state_cache.get(h)
            if cached is not None:
                results[i] = cached
            elif h in seen:
                unique_targets[seen[h]].append(i)
            else:
                seen[h] = len(unique_states)
                unique_states.append((h, s))
                unique_targets.append([i])

        if unique_states:
            batch = np.stack([self.encoder.encode(s) for _, s in unique_states])
            model_input = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw_priors, raw_values = self.model(model_input)
                priors = torch.softmax(raw_priors, dim=1).cpu().numpy()
                values = raw_values.squeeze(1).cpu().numpy()

            for k, (h, _) in enumerate(unique_states):
                entry = (priors[k], float(values[k]))
                self.state_cache[h] = entry
                for i in unique_targets[k]:
                    results[i] = entry

        return results

    def _make_node(self, state, full_priors, value, parent, move):
        """Build a node from a full-action-space prior vector, gathering it onto legal moves."""
        legal_moves = state.get_legal_moves()
        legal_indices = [self.encoder.encode_move(m) for m in legal_moves]
        legal_priors = full_priors[legal_indices].astype(np.float32)

        total = legal_priors.sum()
        if total > 0:
            legal_priors /= total
        elif len(legal_priors) > 0:
            legal_priors[:] = 1.0 / len(legal_priors)

        return ZeroTreeNode(state, value, legal_moves, legal_indices, legal_priors, parent, move)

    def _select_child_idx(self, node):
        """PUCT: pick the legal-move position maximizing Q + U."""
        visits = node.visit_counts
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.where(visits > 0, node.total_values / visits, 0.0)
        u = self.c * node.priors * np.sqrt(node.total_visit_count) / (1 + visits)
        return int(np.argmax(q + u))

    def _terminal_value(self, terminal_state, mover):
        """Value of a finished game from the perspective of ``mover`` (the player who is to move
        at the node whose edge we are scoring)."""
        winner = terminal_state.winner
        if winner is None:
            return 0.0
        return 1.0 if winner == mover else -1.0

    def _backpropagate(self, path, value):
        """
        Walk the path leaf->root. For each edge we (a) undo the virtual loss applied during
        descent and (b) add the real result. The +1 visit added during descent is the real
        visit, so visit counts are NOT touched here -- only the value is corrected. This keeps
        visit statistics exact, which matters because they become the policy target.
        """
        for node, idx in reversed(path):
            node.total_values[idx] += self.virtual_loss + value
            value = -value

    # ------------------------------------------------------------------ search

    def select_move(self, game_state, temperature=1.0, add_noise=False):
        full_priors, root_value = self._evaluate([game_state])[0]
        root = self._make_node(game_state, full_priors, root_value, parent=None, move=None)

        if add_noise and self.dirichlet_alpha > 0 and len(root.moves_list) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.moves_list))
            root.priors = ((1 - self.dirichlet_epsilon) * root.priors
                           + self.dirichlet_epsilon * noise).astype(np.float32)

        sims_done = 0
        while sims_done < self.num_rounds:
            this_batch = min(self.batch_size, self.num_rounds - sims_done)
            expand_leaves = []   # (path, new_state, node, idx)

            for _ in range(this_batch):
                sims_done += 1
                node = root
                path = []          # list[(node, child_idx)]

                while True:
                    if node.is_terminal or len(node.moves_list) == 0:
                        # Existing terminal node reached: score it directly.
                        if path:
                            mover = path[-1][0].state.next_player
                            self._backpropagate(path, self._terminal_value(node.state, mover))
                        break

                    idx = self._select_child_idx(node)
                    # Virtual loss: temporarily make this edge look visited-and-losing so the
                    # other leaves in this batch are pushed toward different parts of the tree.
                    node.visit_counts[idx] += 1
                    node.total_values[idx] -= self.virtual_loss
                    node.total_visit_count += 1
                    path.append((node, idx))

                    child = node.children[idx]
                    if child is None:
                        # Unexpanded edge: this is our expansion target.
                        move = node.moves_list[idx]
                        new_state = node.state.apply_move(move)
                        if new_state.is_over():
                            mover = node.state.next_player
                            self._backpropagate(path, self._terminal_value(new_state, mover))
                        else:
                            expand_leaves.append((path, new_state, node, idx))
                        break

                    node = child

            if not expand_leaves:
                continue

            evaluations = self._evaluate([ns for _, ns, _, _ in expand_leaves])
            for (path, new_state, node, idx), (child_priors, child_value) in zip(expand_leaves, evaluations):
                child = self._make_node(new_state, child_priors, child_value, parent=node, move=node.moves_list[idx])
                node.children[idx] = child
                # child_value is from the child's player-to-move perspective; the edge belongs
                # to ``node``'s player, so the value to back up is -child_value.
                self._backpropagate(path, -child_value)

        # Record the search policy (visit counts scattered into the full action space).
        if self.collector is not None:
            visit_counts = np.zeros(self.encoder.num_moves(), dtype=np.float32)
            for move_idx, count in zip(root.encoder_indices, root.visit_counts):
                visit_counts[move_idx] = count
            self.collector.record_decision(self.encoder.encode(game_state), visit_counts)

        if len(root.moves_list) == 0:
            print("No legal moves available, returning None.")
            return None

        if temperature == 0:
            return root.moves_list[int(np.argmax(root.visit_counts))]

        counts = root.visit_counts.astype(np.float32)
        if counts.sum() == 0:
            probs = np.ones_like(counts) / len(counts)
        else:
            probs = counts ** (1.0 / temperature)
            probs /= probs.sum()
        return root.moves_list[np.random.choice(len(root.moves_list), p=probs)]

    def train(self, experience, batch_size, epochs):
        dataloader = experience.get_dataloader(batch_size)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, dataloader)
