import pandas as pd
import numpy as np
import os


def calculate_metrics(df, game):
    action_df = pd.read_pickle('random.pkl')
    action_df['Action'] = action_df['Action'].astype(str)
    max_value = action_df['Count'].max()
    df['Attack'] = 0
    df['Defense'] = 0
    df['Vitality'] = 0
    df['Eccentricity'] = 0.0  # New column for eccentricity, initialized as float
    df = df[(df['Match'] == game) & (df['Source'] != 'SYSTEM') & (df['Action_Type'] != 'DECLARE_PIZZA')]

    # Group by round and calculate metrics
    for round_number, round_df in df.groupby('Round'):
        player_attack = {}
        player_defense = {}
        visualization_data = []

        # Extract relevant columns
        actions = round_df['Action_Description'][round_df['Action_Description'].notna()].tolist()
        players = round_df['Source'].tolist()

        for i in range(len(actions)):
            if players[i] is not None:
                player = players[i]

                # Initialize the player's attack and defense counts if not already
                if player not in player_attack:
                    player_attack[player] = 0
                if player not in player_defense:
                    player_defense[player] = 0

                if actions[i] != 'pass':  # DISCARD or any other action except PASS
                    # Count defense as the number of 'PASS' actions before the current action
                    player_defense[player] = actions[:i].count('pass')

                    # Count attack as the number of 'PASS' actions after the current action
                    if any(action != 'pass' for action in actions[i + 1:]):
                        next_action_index = next(
                            j for j, action in enumerate(actions[i + 1:], i + 1) if action != 'pass')
                        player_attack[player] = actions[i + 1:next_action_index].count('pass')
                    else:
                        player_attack[player] = actions[i + 1:].count('pass')

                # Calculate eccentricity
                possible_actions = round_df.iloc[i]['Possible_Actions']
                action_done = round_df.iloc[i]['Action_Description']
                action_counts = {}

                for action in possible_actions:
                    if action == 'pass':
                        action_counts[action] = -max_value  # Special handling for 'pass'
                    else:
                        count = action_df[action_df['Action'] == action]['Count'].values
                        action_counts[action] = count[0] if len(count) > 0 else 0

                # Add action_done to action_counts if it's not in possible_actions
                if action_done == 'pass':
                    action_counts[action_done] = -max_value

                # Calculate differences (eccentricity)
                highest_prob = max(action_counts.values())
                differences = highest_prob - action_counts.get(action_done, 0)

                # Additional conditions for 'pass' action
                if highest_prob == -max_value and action_done == 'pass':
                    differences = -0.01
                if highest_prob != -max_value and action_done == 'pass':
                    differences = -0.03

                # Store the eccentricity, cast to float to avoid dtype warning
                df.loc[(df['Round'] == round_number) & (df['Source'] == player), 'Eccentricity'] = float(differences)

        # Calculate vitality: Count of actions that are not 'pass'
        vitality = round_df.groupby('Source')['Action_Description'].apply(
            lambda x: x[(x.notna()) & (x != 'pass')].count()
        )

        metrics = {
            'Attack': player_attack,
            'Defense': player_defense,
            'Vitality': vitality
        }
        encountered_players = set()
        for index, row in round_df.iterrows():
            player = row['Source']

            if player is not None:
                # Check if the player has already been encountered
                if player in encountered_players:
                    # If already encountered, set the metrics for this specific row to NaN
                    df.loc[index, ['Attack', 'Defense', 'Vitality']] = np.nan
                else:
                    # If not encountered, save the metric values
                    for metric, values in metrics.items():
                        df.loc[index, metric] = values[player]
                    # Add player to the set of encountered players
                    encountered_players.add(player)

    # Extract the result DataFrame with the new eccentricity column
    result_df = df[['Match', 'Round', 'Source', 'Attack', 'Defense', 'Vitality', 'Eccentricity']]
    aggr_df = df[['Match', 'Round', 'Source', 'Attack', 'Defense', 'Vitality']]

    aggr_df = df.groupby(['Match', 'Round', 'Source'], as_index=False).agg({
        'Attack': 'sum',
        'Defense': 'sum',
        'Vitality': 'first'
        # Keep the first occurrence of Vitality (or you can use 'sum' if you want to sum across rounds)
    })
    return result_df, aggr_df

def standardize_names(df):
    df['Source'] = df['Source'].replace({
        r'Random_\d+': 'Random',          # Any player like Random_01, Random_02 becomes 'Random'
        r'DQL_vsEveryone.*': 'DQL_vsEveryone',  # Standardize DQL_vsEveryone
        r'PPO_vsEveryone.*': 'PPO_vsEveryone'   # Standardize PPO_vsEveryone
    }, regex=True)
    return df

# df1 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_1.pkl")
# df2 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_2.pkl")
# df3 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_3.pkl")
# df4 = pd.read_pickle("/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/250GamesLargerValue/Dataset_4.pkl")
# # Apply the function to each dataframe
# df1 = standardize_names(df1)
# df2 = standardize_names(df2)
# df3 = standardize_names(df3)
# df4 = standardize_names(df4)

# Step 2: Merge the dataframes
# Assuming the dataframes share a common key like 'round' or 'game_id'
# df = pd.concat([df1, df2, df3, df4], ignore_index=True)


base_path = "/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/Datasets/testing/all_rooms_concatenated.pkl"
# Read CSV and save as pickle
# df = pd.read_csv(base_path)
# df = df.reset_index(drop=True)
# pickle_path = base_path.replace('.csv', '.pkl')  # Remove .csv extension for pickle
# df.to_pickle(pickle_path)
final_df = []
aggr_df = []
df = pd.read_pickle(base_path)
# Calculate scores
for game in df['Match'].unique():
    if game == 0:
        continue
    # Filter the DataFrame for the current game
    game_df = df[(df['Match'] == game) & (df['Source'] != 'SYSTEM')]

    # Find the index where Player_Finished is True
    finish_index = game_df[game_df['Player_Finished'] == True].index.min()

    # Get the Round value for the finish_index row
    finish_round = game_df.loc[finish_index, 'Round']
    next_round = finish_round + 1
    next_round_index = game_df[game_df['Round'] == next_round].index.min()

    game_df = game_df.loc[:next_round_index - 1]
    # Group by round and calculate Attack, Defence, and Vitality
    result, summed = calculate_metrics(game_df, game)
    final_df.append(result)
    aggr_df.append(summed)

    # Create a PlayerAnalysis instance and generate the plot for Attack, Defense, Vitality and Eccentricity
    # There are multiple possible plots:
    # - radar_chart_tot: Plot a single radar chart FOR GAME that is the collection of the mean of the three metrics
    # - radar_chart: Plot a single radar cart FOR ROUND with the sum of each of the metrics for the specific round
    # - self_plots_tot: Plot eccentricity metric as a boxplot for a game
    # - self_plots: Plot eccentricity metric as barplot for each action done by each of the player follow the sequence
    #   of the actions
    # - stack_plots_sing: Plot singular plot of attack, defense and vitality as lineplot for each player

final_df = pd.concat(final_df, ignore_index=True)

output_dir = "MetricsDataset/Training/"
os.makedirs(output_dir, exist_ok=True)
final_df.to_csv(os.path.join(output_dir, "DQL_atk_all.csv"), index=False)
player_counts = final_df['Source'].value_counts()
print(player_counts)







