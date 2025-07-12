from hnefatafl.agents.agent import Agent
import torch


class ACAgent(Agent):
    def __init__(self, model, encoder, device='cpu'):
        self.model = model.to(device)
        self.encoder = encoder
        self.collector = None
        self.last_state_value = None

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        X = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value_est = self.model(X)
            policy_logits = policy_logits.squeeze(0)
            value = value_est.item()

        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            print("No legal moves available.")
            return None

        legal_move_indices = [self.encoder.encode_move(move) for move in legal_moves]

        logit_mask = torch.ones_like(policy_logits) * -1e9
        logit_mask[legal_move_indices] = 0
        masked_logits = policy_logits + logit_mask

        probs = torch.softmax(masked_logits, dim=-1)
        move_index = torch.multinomial(probs, num_samples=1).item()
        move = self.encoder.decode_move_index(move_index)

        if self.collector is not None:
            self.collector.record_decision(
                state=board_tensor,
                action=move_index,
                estimated_value=value,
            )

        return move

    def train(self, experience, lr=0.1, batch_size=128):
        pass

    def save(self, path):
        pass

    def diagnostic(self, game_state):
        return {'value': self.last_state_value}

def load_ac_agent(path):
    pass


