import wandb
from hnefatafl.core.gameTypes import Player

class StatTracker:
    def __init__(self):
        self.game_results = []
        self.game_lengths = []
        self.win_types = {
            "escape": 0,
            "capture": 0,
            "move_limit": 0,
            "repetition_white": 0,
            "repetition_black": 0,
        }

    def log_game(self, data):
        self.game_results.append(data["winner"])
        self.game_lengths.append(data["move_count"])
        if data["repetition_hit"]:
            if data["repeating_player"] == Player.white:
                self.win_types["repetition_white"] += 1
            elif data["repeating_player"] == Player.black:
                self.win_types["repetition_black"] += 1
        elif data["move_limit_hit"]:
            self.win_types["move_limit"] += 1
        elif data["winner"] == Player.black:
            self.win_types["capture"] += 1
        elif data["winner"] == Player.white:
            self.win_types["escape"] += 1


    def summarize_generation(self, generation):
        num_games = len(self.game_results)
        if num_games == 0:
            print("No games played in this generation.")
            return

        white_wins = self.game_results.count(Player.white)
        black_wins = self.game_results.count(Player.black)
        white_win_pct = white_wins / num_games
        black_win_pct = black_wins / num_games

        avg_game_length = sum(self.game_lengths) / num_games

        print(f"\n--- Generation {generation} Summary ---")
        print(f"White Win Rate: {white_win_pct:.2%}")
        print(f"Black Win Rate: {black_win_pct:.2%}")
        print(f"Average Game Length: {avg_game_length:.1f} moves")
        print(f"Win Types: {self.win_types}")
        print("---------------------------\n")

        wandb.log({
            "generation": generation,
            "white_win_rate": white_win_pct,
            "black_win_rate": black_win_pct,
            "average_game_length": avg_game_length,
            "wins_by_escape": self.win_types["escape"],
            "wins_by_capture": self.win_types["capture"],
            "move_limit_hits": self.win_types["move_limit"],
            "repetition_white": self.win_types["repetition_white"],
            "repetition_black": self.win_types["repetition_black"],
        })

        self._reset_generation_stats()

    def _reset_generation_stats(self):
        self.game_results.clear()
        self.game_lengths.clear()
        self.win_types = {k:0 for k in self.win_types}


    def close(self):
        wandb.finish()
        print("WandB session closed.")


