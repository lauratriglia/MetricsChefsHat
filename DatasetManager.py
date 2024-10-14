import pandas as pd
import numpy as np



def calculate_new_data (df, game):
    df['Number_Pass'] = 0
    df['Number_Discard'] = 0
    df['Pizza_Player'] = 0
    all_df = []
    for round_number, round_df in game_df.groupby('Round'):
        players = round_df['Source'].tolist()
        action_desc = round_df['Action_Description'].tolist()
        action_type = round_df['Action_Type'].tolist()

        # Initialize dictionaries for storing actions
        player_pass = {}
        player_discard = {}
        player_pizza = None  # To store the pizza player as a single value

        # Loop through each action description and player
        for i, player in enumerate(players):
            if player is not None:
                # Initialize pass, discard if not set
                if player not in player_pass:
                    player_pass[player] = 0
                if player not in player_discard:
                    player_discard[player] = 0

                # Process 'pass' action
                if action_desc[i] == 'pass':
                    player_pass[player] += 1

                # Process discard based on 'Q' value in action description
                elif isinstance(action_desc[i], str):
                    parts = action_desc[i].split(';')
                    for part in parts:
                        if 'Q' in part:
                            discard_count = int(part[1:])
                            player_discard[player] += discard_count

                # Check if action_type is 'declare_pizza' and action_desc is NaN
                elif pd.isna(action_desc[i]) and action_type[i] == 'DECLARE_PIZZA':
                    player_pizza = player

        round_rows = []
        for player in player_pass.keys():
            round_rows.append({
                'Match': game,
                'Round': round_number,
                'Source': player,
                'Number_Pass': player_pass[player],
                'Number_Discard': player_discard[player],
                'Pizza_Player': (player == player_pizza)  # Mark True if this is the pizza player
            })


        round_df = pd.DataFrame(round_rows)
        all_df.append(round_df)



# Final result dataframe
    result_df = pd.concat(all_df, ignore_index=True)

# result_df = df[['Round Number', 'Source', 'Number_Pass', 'Number_Discard', 'Pizza_Player']]

    return result_df


def standardize_names(df):
    df['Source'] = df['Source'].replace({
        r'Random_\d+': 'Random',          # Any player like Random_01, Random_02 becomes 'Random'
        r'DQL_vsEveryone.*': 'DQL_vsEveryone',  # Standardize DQL_vsEveryone
        r'PPO_vsEveryone.*': 'PPO_vsEveryone'   # Standardize PPO_vsEveryone
    }, regex=True)
    return df

df1 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_1.pkl")
df2 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_2.pkl")
df3 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_3.pkl")
df4 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_4.pkl")
# Apply the function to each dataframe
df1 = standardize_names(df1)
df2 = standardize_names(df2)
df3 = standardize_names(df3)
df4 = standardize_names(df4)

# Step 2: Merge the dataframes
# Assuming the dataframes share a common key like 'round' or 'game_id'
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

base_path = '/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/PpovsRandom/Dataset.pkl'

#df = pd.read_pickle(base_path)
#df = df.reset_index(drop=True)
final_df = []


for game in df['Match'].unique():
    if game == 0:
        continue
    game_df = df[(df['Match'] == game) & (df['Source'] != 'SYSTEM')]

    # Find the index where Player_Finished is True
    finish_index = game_df[game_df['Player_Finished'] == True].index.min()

    # Get the Round value for the finish_index row
    finish_round = game_df.loc[finish_index, 'Round']
    next_round = finish_round + 1
    next_round_index = game_df[game_df['Round'] == next_round].index.min()

    game_df = game_df.loc[:next_round_index-1]
    # Group by round and calculate Attack, Defence, and Vitality
    result = calculate_new_data(game_df, game)
    final_df.append(result)

final_df = pd.concat(final_df, ignore_index=True)
player_counts = final_df['Source'].value_counts()

# Display the counts
print(player_counts)
final_df.to_csv("MetricsDataset/250GamesLargerValue/250GamesLargerValue_OldMetrics.csv")






