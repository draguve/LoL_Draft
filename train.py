# %% Imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import LeagueDataset, collate_fn
from dataset import get_games, load_dataset
from model import LeagueModel
from tokenizer import Tokenizer

DATASET_LOCATION = "Dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f"Using device: {device}")


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

# %% Dataloaders
dataset = LeagueDataset(games, tokenizer)
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
)

# %% Model
model = LeagueModel(vocab_size=tokenizer.vocab_size()).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# %% Train Model
for epoch in range(NUM_EPOCHS):
    model.train()
    for tokens, blue_wins in train_dataloader:
        # todo fix blue win size here
        length = tokens.shape
        tokens = tokens.to(device)
        blue_wins = blue_wins.to(device)

        pred = model(tokens)
        loss = criterion(pred, blue_wins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for tokens, blue_wins in test_dataloader:
            # todo fix blue win size here
            length = tokens.shape
            tokens = tokens.to(device)
            blue_wins = blue_wins.to(device)

            val_pred = model(tokens)
            loss = criterion(val_pred, blue_wins)

# # %% Check
# print(list(game["champs"] for game in games))
