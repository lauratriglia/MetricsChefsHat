
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from PlotManager import PlayerAnalysis
from scipy.stats import entropy


def assign_cluster_tens(row, max_round):
    # Calculate the thresholds for dividing into three parts
    first_part = max_round / 3
    second_part = 2 * max_round / 3

    # Divide each section (Beginning, Middle, End) into 10 sub-parts
    if row['Round'] <= first_part:
        # Scale the round number to a value between 1 and 10 for the beginning
        sub_part = int((row['Round'] / first_part) * 5)
        return f'B{sub_part if sub_part > 0 else 1}'  # Handle rounding down to 0
    elif row['Round'] <= second_part:
        # Scale the round number to a value between 1 and 10 for the middle
        sub_part = int(((row['Round'] - first_part) / (second_part - first_part)) * 5)
        return f'M{sub_part if sub_part > 0 else 1}'
    else:
        # Scale the round number to a value between 1 and 10 for the end
        sub_part = int(((row['Round'] - second_part) / (max_round - second_part)) * 5)
        return f'E{sub_part if sub_part > 0 else 1}'


def assign_cluster(row, max_round):
    # Calculate the thresholds for dividing into three parts
    first_part = max_round / 3
    second_part = 2 * max_round / 3

    if row['Round'] <= first_part:
        return 'Beginning'
    elif row['Round'] <= second_part:
        return 'Middle'
    else:
        return 'End'

def clean_dataset(final_df):
    final_df = final_df.dropna(how='any')
    expected_players = set(final_df['Source'].unique())

    # Group the dataframe by 'Match' and 'Round'
    match_round_groups = (final_df.groupby(['Match', 'Round']))

    # Function to check if the players in a match-round pair match the expected set
    def check_match_round(round_data):
        players_in_round = set(round_data['Source'])
        return players_in_round == expected_players and len(round_data) == 4

    # Find match-round pairs where there is a mismatch or repetition
    invalid_match_rounds = [(match, round_number)
                            for (match, round_number), round_data in match_round_groups
                            if not check_match_round(round_data)]

    # If there are invalid match-round pairs, filter the dataframe to remove them
    if invalid_match_rounds:
        # Create a filter to exclude the invalid match-round pairs
        final_df_cleaned = final_df[~final_df[['Match', 'Round']].apply(tuple, axis=1).isin(invalid_match_rounds)]
    else:
        final_df_cleaned = final_df  # No issues found, keep the original dataframe
    player_counts = final_df_cleaned['Source'].value_counts()

    # Display the counts
    print(player_counts)

    # final_df_cleaned.to_csv("MetricsDataset/PpoAllRandom/PpoAllRandom_corr.csv", index=False)
    return final_df_cleaned

# Apply the clustering logic for each match
def filtering_multiple_df(files):
    player_names = ['LARGER_VALUE_01', 'LARGER_VALUE_04']

    # New names to assign after filtering
    new_player_names = ['LargerValue_R', 'LargerValue_Dql', 'LargerValue_Ppo']

    # Create an empty list to store filtered DataFrames
    filtered_dfs = []

    # Loop through the CSV files and new names
    for i, file in enumerate(files):
        # Load the CSV file
        df = pd.read_csv(file)

        # Clean the dataset using the clean_dataset function
        df = clean_dataset(df)

        # Calculate the max round for each match and assign the corresponding cluster
        df['Cluster'] = df.groupby('Match')['Round'].transform(lambda x: x.max())
        df['Cluster'] = df.apply(lambda row: assign_cluster_tens(row, row['Cluster']), axis=1)

        # Replace negative Eccentricity values with NaN
        # df['Eccentricity'] = df['Eccentricity'].apply(lambda x: np.nan if x < 0 else x)

        # Calculate the round percentage using the calculate_round_percentage function
        df = df.groupby('Match').apply(calculate_round_percentage)

        # Loop through each player_name in player_names list and filter the DataFrame
        for player_name in player_names:
            # Filter the DataFrame for the specific player
            df_filtered = df[df['Source'] == player_name].copy()

            # Rename the player in the filtered DataFrame
            if i < len(new_player_names):  # Ensure we don't go out of bounds in case there are fewer files than names
                df_filtered['Source'] = new_player_names[i]

            # Append the filtered DataFrame to the list
            filtered_dfs.append(df_filtered)

    # Merge all filtered DataFrames together (concatenate)
    final_df = pd.concat(filtered_dfs)

    # Display the merged DataFrame
    return final_df

def calculate_round_percentage(sub_df):
    # Find the min and max round for the match
    min_round = sub_df['Round'].min()
    max_round = sub_df['Round'].max()

    # Scale the rounds from 0% to 100%
    sub_df['Round_Percent'] = ((sub_df['Round'] - min_round) / (max_round - min_round)) * 100
    return sub_df



# final_df = filtering_multiple_df(csv_files)
df = pd.read_csv("MetricsDataset/250GamesLargerValue/250GamesLargerValue_OldMetrics.csv")
df = clean_dataset(df)
df['Cluster'] = df.groupby('Match')['Round'].transform(lambda x: x.max())  # Get the max round for each match
df['Cluster'] = df.apply(lambda row: assign_cluster_tens(row, row['Cluster']), axis=1)
# df['Eccentricity'] = df['Eccentricity'].apply(lambda x: np.nan if x < 0 else x)

# df = df.groupby('Match').apply(calculate_round_percentage)
# Assume 'agg_data' is the DataFrame with mean_attack, mean_defense, mean_vitality for each player
"""
agg_data = df.groupby(['Cluster', 'Source']).agg(
                mean_attack=('Attack', 'mean'),
                mean_defense=('Defense', 'mean'),
                mean_vitality=('Vitality', 'mean'),
                mean_eccentricity=('Eccentricity', 'mean'),
            ).reset_index()
"""
agg_data = df.groupby(['Cluster', 'Source']).agg(
        mean_pass = ('Number_Pass', 'mean'),
        mean_discard = ('Number_Discard', 'mean'),
).reset_index()
# Step 1: Extract the metrics for two players, e.g., Player 1 and Player 2
player1 = agg_data[agg_data['Source'] == 'DQL_vsEveryone'][['mean_pass', 'mean_discard']].values[0]
player2 = agg_data[agg_data['Source'] == 'PPO_vsEveryone'][['mean_pass', 'mean_discard']].values[0]

# Step 2: Normalize the vectors so they sum to 1 (turn them into probability distributions)
player1_prob = player1 / np.sum(player1)
player2_prob = player2 / np.sum(player2)

# Step 3: Calculate KL divergence from Player 1 to Player 2
kl_divergence = entropy(player1_prob, player2_prob)

# Output the KL divergence
print(f"KL Divergence between DQL and PPO: {kl_divergence}")
player1 = agg_data[agg_data['Source'] == 'DQL_vsEveryone'][['mean_pass', 'mean_discard']].values[0]
player2 = agg_data[agg_data['Source'] == 'RANDOM_01'][['mean_pass', 'mean_discard']].values[0]

# Step 2: Normalize the vectors so they sum to 1 (turn them into probability distributions)
player1_prob = player1 / np.sum(player1)
player2_prob = player2 / np.sum(player2)

# Step 3: Calculate KL divergence from Player 1 to Player 2
kl_divergence = entropy(player1_prob, player2_prob)

# Output the KL divergence
print(f"KL Divergence between DQL and Random: {kl_divergence}")
player1 = agg_data[agg_data['Source'] == 'DQL_vsEveryone'][['mean_pass', 'mean_discard']].values[0]
player2 = agg_data[agg_data['Source'] == 'LARGER_VALUE_03'][['mean_pass', 'mean_discard']].values[0]

# Step 2: Normalize the vectors so they sum to 1 (turn them into probability distributions)
player1_prob = player1 / np.sum(player1)
player2_prob = player2 / np.sum(player2)

# Step 3: Calculate KL divergence from Player 1 to Player 2
kl_divergence = entropy(player1_prob, player2_prob)

# Output the KL divergence
print(f"KL Divergence between DQL and LargerValue: {kl_divergence}")

player1 = agg_data[agg_data['Source'] == 'PPO_vsEveryone'][['mean_pass', 'mean_discard']].values[0]
player2 = agg_data[agg_data['Source'] == 'LARGER_VALUE_03'][['mean_pass', 'mean_discard']].values[0]

# Step 2: Normalize the vectors so they sum to 1 (turn them into probability distributions)
player1_prob = player1 / np.sum(player1)
player2_prob = player2 / np.sum(player2)

# Step 3: Calculate KL divergence from Player 1 to Player 2
kl_divergence = entropy(player1_prob, player2_prob)

# Output the KL divergence
print(f"KL Divergence between PPO and LargerValue: {kl_divergence}")
player1 = agg_data[agg_data['Source'] == 'PPO_vsEveryone'][['mean_pass', 'mean_discard']].values[0]
player2 = agg_data[agg_data['Source'] == 'RANDOM_01'][['mean_pass', 'mean_discard']].values[0]

# Step 2: Normalize the vectors so they sum to 1 (turn them into probability distributions)
player1_prob = player1 / np.sum(player1)
player2_prob = player2 / np.sum(player2)

# Step 3: Calculate KL divergence from Player 1 to Player 2
kl_divergence = entropy(player1_prob, player2_prob)

# Output the KL divergence
print(f"KL Divergence between PPO and Random: {kl_divergence}")
plots = PlayerAnalysis(df)
# plots.chosen_mandatory_pass("MetricsPlots/AIACIMP/")
# plots.plots_3d_time_metrics("MetricsPlots/250GamesLargerValue/")
# plots.plots_3d_metrics("MetricsPlots/250GamesLargerValue/")
# plots.shadow_mean_line("MetricsPlots/LargerValuevsRandom/All" , 'Attack')
# plots.shadow_mean_line("MetricsPlots/LargerValuevsRandom/All" , 'Defense')
# plots.shadow_mean_line("MetricsPlots/LargerValuevsRandom/All" , 'Vitality')
# plots.shadow_mean_line("MetricsPlots/LargerValuevsRandom/All" , 'Eccentricity')
#plots.lineplot_bme("MetricsPlots/250GamesLargerValue/" , 'Attack')
#plots.lineplot_bme("MetricsPlots/250GamesLargerValue/" , 'Defense')
#plots.lineplot_bme("MetricsPlots/250GamesLargerValue/" , 'Vitality')
#plots.lineplot_bme("MetricsPlots/250GamesLargerValue/" , 'Eccentricity')
# plots.percentage_rounds("MetricsPlots/LargerValuevsRandom/Prova.png")
#plots.boxplot_players("MetricsPlots/DqlAllRandom/", 'Attack')
#plots.boxplot_players("MetricsPlots/DqlAllRandom/", 'Defense')
#plots.boxplot_players("MetricsPlots/DqlAllRandom/", 'Vitality')
#plots.boxplot_players("MetricsPlots/DqlAllRandom/", 'Eccentricity')
# plots.boxplot_players("MetricsPlots/LargerValueAll/", 'Number_Pass')
# plots.boxplot_players("MetricsPlots/LargerValueAll/", 'Number_Discard')
#plots.stack_plots_sing("MetricsPlots/DQL_Def/dql_def_vsRandom")
#plots.stack_barplots_sing("MetricsPlots/DQL_Def/dql_def_vsRandom")
plots.barplot_overall_game("MetricsPlots/DQL_Vit/dql_vit_1k_box", metrics='Vitality', plot_type='box')


#print("Mean Attack for :", df[df['Source'] == 'DQL_attack']['Attack'].mean())



# Statistical test between players on Attack
from scipy.stats import f_oneway, ttest_ind
players = df['Source'].unique()
attack_groups = [df[df['Source'] == player]['Vitality'].dropna() for player in players]
anova_result = f_oneway(*attack_groups)
print('ANOVA result for Vitality between players:', anova_result)

# Pairwise t-tests
for i in range(len(players)):
    for j in range(i+1, len(players)):
        t_stat, p_val = ttest_ind(attack_groups[i], attack_groups[j], equal_var=False)
        print(f'T-test between {players[i]} and {players[j]}: t={t_stat:.3f}, p={p_val:.3g}')