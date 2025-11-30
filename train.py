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
pprint(len(tokenizer.player_to_id))

# for game in games:
#     print(game["gameid"])

# %% Fearless
pprint(games[-1])
