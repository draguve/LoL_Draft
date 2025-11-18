# %% Imports
from pprint import pprint

from dataset import load_dataset, parse_group

DATASET_LOCATION = "Dataset"


df = load_dataset()
df_complete = df.loc[df["datacompleteness"] == "complete"]
print(df_complete.shape)

# %% C

for gameid, group in df_complete.groupby("gameid"):
    game = parse_group(group)
    pprint(game)
    break
