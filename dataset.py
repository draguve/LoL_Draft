import os

import pandas as pd

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
    data["patch"] = group["patch"].iloc[0]
    data["date"] = group["date"].iloc[0]
    data["blue_team"] = group.loc[group["side"] == "Blue", "teamname"].iloc[0]
    data["red_team"] = group.loc[group["side"] == "Red", "teamname"].iloc[0]
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
