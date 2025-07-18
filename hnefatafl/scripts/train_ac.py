import copy

import torch
import torch.nn.functional as F
import numpy as np

from hnefatafl.models.acmodel import ACModel
from hnefatafl.agents.rl.acagent import ACAgent
from hnefatafl.rl.ACexperience import ExperienceCollector, combine_experience, ExperienceBuffer
from hnefatafl.utils.nnTrainingUtils import simulate_game
from hnefatafl.core.gameTypes import Player
from hnefatafl.agents.agent import RandomAgent


def evaluate_model(agent, encoder, old_agent=None, num_games=50):
    print("\n--- Evaluating Model ---")
    random_agent = RandomAgent()
    wins_as_white, wins_as_black = 0, 0

    for _ in range(num_games // 2):
        winner = simulate_game(black_player=agent, white_player=random_agent)
        if winner == Player.white:
            wins_as_white += 1
    for _ in range(num_games // 2):
        winner = simulate_game(black_player=random_agent, white_player=agent)
        if winner == Player.black:
            wins_as_black += 1

    print(f"vs Random Agent ({num_games} games):")
    print(f"  Wins as White: {wins_as_white}/{num_games // 2}")
    print(f"  Wins as Black: {wins_as_black}/{num_games // 2}")

    # Evaluation against a previous version of itself
    if old_agent:
        wins_as_white, wins_as_black = 0, 0
        for _ in range(num_games // 2):
            winner = simulate_game(black_player=agent, white_player=old_agent)
            if winner == Player.white:
                wins_as_white += 1
        for _ in range(num_games // 2):
            winner = simulate_game(black_player=old_agent, white_player=agent)
            if winner == Player.black:
                wins_as_black += 1

        print(f"vs Previous Version ({num_games} games):")
        print(f"  Wins as White: {wins_as_white}/{num_games // 2}")
        print(f"  Wins as Black: {wins_as_black}/{num_games // 2}")
    print("--- End Evaluation ---\n")


def train_ac(encoder, num_eps=100, learning_rate=0.001, batch_size=128,
             update_freq=10, eval_freq = 10, save_path=None, grad_clip=1.0):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training, no MPS device available.")

    device = torch.device("cpu")  # force to use cpu for now
    model = ACModel(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    black_agent = ACAgent(model, encoder)
    white_agent = ACAgent(model, encoder)

    old_model = copy.deepcopy(model)
    old_agent = ACAgent(old_model, encoder)

    experience_buffers = []

    for i in range(num_eps):
        print(f"--- starting episode {i + 1}/{num_eps} ---")

        if (i + 1) % eval_freq == 0:
            model.eval()
            evaluate_model(black_agent, encoder, old_agent=old_agent, num_games=50)
            old_model.load_state_dict(model.state_dict())
            model.train()

        black_collector = ExperienceCollector()
        white_collector = ExperienceCollector()
        black_agent.set_collector(black_collector)
        white_agent.set_collector(white_collector)

        black_collector.begin_episode()
        white_collector.begin_episode()

        game_winner = simulate_game(black_agent, white_agent)

        if game_winner == Player.black:
            black_collector.complete_episode(reward=1)
            white_collector.complete_episode(reward=-1)
        elif game_winner == Player.white:
            black_collector.complete_episode(reward=-1)
            white_collector.complete_episode(reward=1)
        else:
            black_collector.complete_episode(reward=0)
            white_collector.complete_episode(reward=0)

        experience_buffers.append(black_collector.to_buffer())
        experience_buffers.append(white_collector.to_buffer())

        if (i + 1) % update_freq != 0:
            continue

        experience = combine_experience(experience_buffers)
        experience_buffers = []

        if len(experience.states) == 0:
            print("No experience to train on.")
            continue

        states = torch.tensor(np.array(experience.states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(experience.actions), dtype=torch.long, device=device)
        advantages = torch.tensor(np.array(experience.advantages), dtype=torch.float32, device=device)
        target_values = torch.tensor(np.array(experience.rewards), dtype=torch.float32, device=device)

        model.train()

        num_samples = len(states)
        indices = torch.randperm(num_samples)

        total_value_loss = 0
        total_actor_loss = 0
        num_batches = 0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_target_values = target_values[batch_indices]

            policy_logits, values = model(batch_states)
            values = values.squeeze(-1)

            value_loss = F.mse_loss(values, batch_target_values)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)

            actor_loss = -(selected_log_probs * batch_advantages.detach()).mean(dim=0)

            total_loss = value_loss + actor_loss

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_value_loss += value_loss.item()
            total_actor_loss += actor_loss.item()
            num_batches += 1

        model.eval()
        avg_v_loss = total_value_loss / num_batches if num_batches > 0 else 0
        avg_a_loss = total_actor_loss / num_batches if num_batches > 0 else 0
        print(f"Update after episode {i + 1}: Avg Value Loss: {avg_v_loss:.4f}, Avg Actor Loss: {avg_a_loss:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder

    encoder = SevenPlaneEncoder(11)
    train_ac(encoder, num_eps=10_000, learning_rate=0.0001, eval_freq=1000,batch_size=2048, update_freq=50, grad_clip=1.0)