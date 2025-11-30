def build_tokeizer(df, games):
    pass


class Tokenizer:
    def __init__(self, df):
        champs = df["champion"].unique()
        cleaned_champs = [champ for champ in champs if champ == champ]
        sorted_champs = sorted(cleaned_champs)
        self.champion_to_id = {
            champ: i + 1 for i, champ in enumerate(sorted_champs)
        }

        players = df["playerid"].unique()
        cleaned_players = [player for player in players if player == player]
        self.player_to_id = {
            player: i + 1 for i, player in enumerate(cleaned_players)
        }

        versions = df["patch"].unique()
        cleaned_version = [
            version for version in versions if version == version
        ]
        self.version_to_id = {
            version: i + 1 for i, version in enumerate(cleaned_version)
        }

    def tokenize_game(self, game):
        tokens = []
        tokens.append(("patch", self.version_to_id[game["patch"]]))
        if game["is_fearless"]:
            for champ in list(sorted(game["fearless_banned"])):
                tokens.append(("ban", self.champion_to_id[champ]))
        for item in game["draft"]:
            champ = item["champion"]
            champ_token = self.champion_to_id[champ]
            match item["type"]:
                case "Pick":
                    playerid = game["champs"][champ]["playerid"]
                    player_token = self.player_to_id[playerid]
                    tokens.append(("pick", champ_token, player_token))
                case "Ban":
                    tokens.append(("ban", champ_token))
        return tokens
