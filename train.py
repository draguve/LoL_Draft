# %% Imports
from torch.utils.data import DataLoader

from dataloader import LeagueDataset, collate_fn
from dataset import get_games, load_dataset
from tokenizer import Tokenizer

DATASET_LOCATION = "Dataset"

# %% Load dataset
df = load_dataset()
df_complete = df.loc[df["datacompleteness"] == "complete"]

# %% Parse games and build tokenizer
games, unparseable_games, ungrouped_games = get_games(df)
print(
    f"Valid Games: {len(games)} Unparsable:{len(unparseable_games)} Ungrouped {len(ungrouped_games)}"
)
tokenizer = Tokenizer(df)
print(f"Vocab Size: {tokenizer.vocab_size()}")

# %% Check Tokenizer
game_tokens = tokenizer.tokenize_game(games[-1])
tokenizer.print_tokens(game_tokens)

# %% Check Dataset
dataset = LeagueDataset(games, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

for tokens, results in dataloader:
    print(tokens.shape, results.shape)

# # %% Check
# print(list(game["champs"] for game in games))
