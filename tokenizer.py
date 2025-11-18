def build_tokeizer(df, games):
    pass


class Tokenizer:
    def __init__(self, df):
        champs = df["champion"].unique()
        cleaned_champs = [champ for champ in champs if champ == champ]
        sorted_champs = sorted(cleaned_champs)
        self.champion_to_id = {
            champ: i for i, champ in enumerate(sorted_champs)
        }

        players = df["playerid"].unique()
        cleaned_players = [player for player in players if player == player]
        self.player_to_id = {
            player: i for i, player in enumerate(cleaned_players)
        }
