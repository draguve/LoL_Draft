import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class LeagueDataset(Dataset):
    def __init__(self, games, tokenizer):
        super().__init__()
        self.tokenizer: Tokenizer = tokenizer
        self.games: list = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        game = self.games[index]
        game_tokens = self.tokenizer.tokenize_game(game)
        blue_win = game["blue_win"]
        return (
            torch.tensor(game_tokens, dtype=torch.long),
            torch.tensor(blue_win, dtype=torch.float),
        )


def collate_fn(batch, pad_token=0):
    tokens, wins = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=pad_token
    )
    return padded, torch.tensor(wins)
