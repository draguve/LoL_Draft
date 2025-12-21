import random


class Tokenizer:
    SPECIAL_TOKENS = ["<pad>", "<unk>"]  # IDs 0, 1
    ACTION_TOKENS = [
        "pick_blue",
        "pick_red",
        "ban_blue",
        "ban_red",
        "fearless_ban",
    ]

    def __init__(self, df):
        self.token_to_id = {}
        next_id = 0

        # 1) Special tokens
        for tok in self.SPECIAL_TOKENS:
            self.token_to_id[tok] = next_id
            next_id += 1
        # next_id now = 2

        # 2) Champions (start at ID 2)
        champs = sorted(x for x in df["champion"].dropna().unique())
        self.champion_start_id = next_id
        for c in champs:
            self.token_to_id[f"champ:{c}"] = next_id
            next_id += 1
        self.champion_end_id = next_id - 1

        # 3) Patches
        versions = sorted(x for x in df["patch"].dropna().unique())
        self.version_start_id = next_id
        for v in versions:
            self.token_to_id[f"patch:{v}"] = next_id
            next_id += 1
        self.version_end_id = next_id - 1

        # 4) Actions (pick/ban tokens)
        self.actions_start_id = next_id
        for a in self.ACTION_TOKENS:
            self.token_to_id[a] = next_id
            next_id += 1
        self.actions_end_id = next_id - 1

        # 5) Players (moved to the end)
        players = sorted(x for x in df["playerid"].dropna().unique())
        self.player_start_id = next_id
        for p in players:
            self.token_to_id[f"player:{p}"] = next_id
            next_id += 1
        self.player_end_id = next_id - 1

        # Reverse mapping
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def tokenize_game(self, game, player_token_prob=1.0):
        tokens = []
        tokens.append(self.token_to_id[f"patch:{game['patch']}"])
        if game["is_fearless"]:
            for champ in list(sorted(game["fearless_banned"])):
                tokens.extend(
                    [
                        self.token_to_id["fearless_ban"],
                        self.token_to_id[f"champ:{champ}"],
                    ]
                )
        for item in game["draft"]:
            side = item["side"]
            champ = item["champion"]
            champ_token = self.token_to_id[f"champ:{champ}"]
            match item["type"]:
                case "Pick":
                    tokens.extend(
                        [
                            self.token_to_id[f"pick_{side}"],
                            champ_token,
                        ]
                    )
                    if (
                        player_token_prob == 1.0
                        or random.random() < player_token_prob
                    ):
                        playerid = game["champs"][champ]["playerid"]
                        player_token = self.token_to_id[f"player:{playerid}"]
                        tokens.append(player_token)
                case "Ban":
                    tokens.extend(
                        [
                            self.token_to_id[f"ban_{side}"],
                            champ_token,
                        ]
                    )
        return tokens

    def parse_tokens(self, tokens) -> list[str]:
        parsed: list[str] = []
        for token in tokens:
            parsed.append(self.id_to_token[token])
        return parsed

    def print_tokens(self, tokens):
        parsed = self.parse_tokens(tokens)
        for token, parsed in zip(tokens, parsed):
            print(f"{token:<30} - {parsed}")

    def vocab_size(self):
        return len(self.token_to_id)
