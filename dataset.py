import os
from datetime import datetime
from itertools import chain
from pprint import pprint

import pandas as pd
from tqdm import tqdm

DATASET_LOCATION = "Dataset"


def load_dataset(on_and_after=2018):
    csvs_to_load = []
    for root, _, files in os.walk(DATASET_LOCATION):
        for file in files:
            if file.endswith(".csv"):
                year = int(file.split("_")[0])
                if year >= on_and_after:
                    csvs_to_load.append(os.path.join(root, file))

    data_frames = []
    for csv in csvs_to_load:
        df = pd.read_csv(csv, low_memory=False)
        data_frames.append(df)

    df = pd.concat(data_frames)
    return df


def parse_group(group):
    data = {}
    data["gameid"] = group["gameid"].iloc[0]
    data["patch"] = group["patch"].iloc[0]
    date_time = datetime.strptime(group["date"].iloc[0], "%Y-%m-%d %H:%M:%S")
    data["date"] = date_time
    data["blue_team"] = group.loc[group["side"] == "Blue", "teamname"].iloc[0]
    data["red_team"] = group.loc[group["side"] == "Red", "teamname"].iloc[0]
    data["blue_id"] = group.loc[group["side"] == "Blue", "teamid"].iloc[0]
    data["red_id"] = group.loc[group["side"] == "Red", "teamid"].iloc[0]
    data["game_in_series"] = group["game"].iloc[0]
    champ_dict = {
        row["champion"]: {
            "playerid": row["playerid"],
            "playername": row["playername"],
        }
        for _, row in group.iterrows()
        if row["position"] != "team"  # skip the team summary rows
    }
    assert len(champ_dict) == 10
    data["champs"] = champ_dict

    # Get the blue and red team rows (the 'team' summary rows)
    blue_team = group[
        (group["side"] == "Blue") & (group["position"] == "team")
    ].iloc[0]
    red_team = group[
        (group["side"] == "Red") & (group["position"] == "team")
    ].iloc[0]

    # Extract bans and picks as lists
    blue_bans = [
        blue_team[f"ban{i}"]
        for i in range(1, 6)
        if pd.notna(blue_team[f"ban{i}"])
    ]
    red_bans = [
        red_team[f"ban{i}"]
        for i in range(1, 6)
        if pd.notna(red_team[f"ban{i}"])
    ]
    blue_picks = [
        blue_team[f"pick{i}"]
        for i in range(1, 6)
        if pd.notna(blue_team[f"pick{i}"])
    ]
    red_picks = [
        red_team[f"pick{i}"]
        for i in range(1, 6)
        if pd.notna(red_team[f"pick{i}"])
    ]
    assert len(blue_bans) == 5
    assert len(red_bans) == 5
    assert len(blue_picks) == 5
    assert len(red_picks) == 5
    data["champ_pool"] = set(blue_picks + red_picks)

    # # --- Draft order reconstruction ---
    draft = []

    # --- Ban Phase 1 ---
    for i in range(3):
        draft.append({"type": "Ban", "champion": blue_bans[i]})
        draft.append({"type": "Ban", "champion": red_bans[i]})

    # --- Pick Phase 1 ---
    draft.append({"type": "Pick", "champion": blue_picks[0]})
    draft.append({"type": "Pick", "champion": red_picks[0]})
    draft.append({"type": "Pick", "champion": red_picks[1]})
    draft.append({"type": "Pick", "champion": blue_picks[1]})
    draft.append({"type": "Pick", "champion": blue_picks[2]})
    draft.append({"type": "Pick", "champion": red_picks[2]})

    # --- Ban Phase 2 ---
    draft.append({"type": "Ban", "champion": red_bans[3]})
    draft.append({"type": "Ban", "champion": blue_bans[3]})
    draft.append({"type": "Ban", "champion": red_bans[4]})
    draft.append({"type": "Ban", "champion": blue_bans[4]})

    # --- Pick Phase 2 ---
    draft.append({"type": "Pick", "champion": red_picks[3]})
    draft.append({"type": "Pick", "champion": blue_picks[3]})
    draft.append({"type": "Pick", "champion": blue_picks[4]})
    draft.append({"type": "Pick", "champion": red_picks[4]})

    assert len(draft) == 20
    data["draft"] = draft
    data["blue_win"] = blue_team["result"] == 1
    return data


def find_fearless(games):
    clean_games = []
    problem_games = []
    sorted_games = sorted(games, key=lambda x: x["date"])
    last_games = {}
    for game in tqdm(sorted_games, "Finding fearless"):
        game_id = game["gameid"]
        blue_id = game["blue_id"]
        red_id = game["red_id"]
        team_set = frozenset([blue_id, red_id])
        if game["game_in_series"] == 1:
            last_games[team_set] = [game]
        elif team_set in last_games:
            last_games[team_set].append(game)
        else:
            problem_games.append(game)
            continue

        if len(last_games[team_set]) != game["game_in_series"]:
            problem_games.append(game)
            continue

        game["prev_games"] = [
            iter["gameid"] for iter in last_games[team_set][0:-1]
        ]
        all_champ_pools = [iter["champ_pool"] for iter in last_games[team_set]]
        all_champs = list(chain.from_iterable(all_champ_pools))
        # print(all_champs)
        is_fearless = len(all_champs) == len(set(all_champs))
        game["is_fearless"] = is_fearless
        if is_fearless:
            prev_champs = [
                iter["champ_pool"] for iter in last_games[team_set][0:-1]
            ]
            game["fearless_banned"] = set().union(*prev_champs)
            assert (
                len(game["fearless_banned"])
                == (game["game_in_series"] - 1) * 10
            )

        clean_games.append(game)

    return clean_games, problem_games


def get_games(df):
    games = []
    unparseable = []
    for gameid, group in tqdm(df.groupby("gameid"), "Parsing games"):
        try:
            game = parse_group(group)
            games.append(game)
        except:
            unparseable.append(gameid)

    games, ungrouped = find_fearless(games)
    return games, unparseable, ungrouped
