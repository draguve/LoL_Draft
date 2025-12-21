# %% Imports
from pprint import pprint

from dataset import get_games, load_dataset
from tokenizer import Tokenizer

DATASET_LOCATION = "Dataset"

# %% Load dataset
df = load_dataset()
df_complete = df.loc[df["datacompleteness"] == "complete"]
print(df_complete.shape)

# %% C
games, unparseable_games, ungrouped_games = get_games(df)
print(len(games), len(unparseable_games), len(ungrouped_games))
# %% Check
tokenizer = Tokenizer(df)

# for game in games:
#     print(game["gameid"])

# %% Fearless
game_tokens = tokenizer.tokenize_game(games[-1])
parsed = tokenizer.parse_tokens(game_tokens)
for token, parsed in zip(game_tokens, parsed):
    print(f"{token} ----- {parsed}")
