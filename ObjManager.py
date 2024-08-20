import pandas as pd
from PlotManager import PlayerAnalysis


def calculate_scores(df, game):
    ## Function to create the df with Attack, Defense and Vitality
    df['Attack'] = 0
    df['Defense'] = 0
    df['Vitality'] = 0
    df = df[(df['Match'] == game) & (df['Source'] != 'SYSTEM')]

    # Group by round and calculate Attack, Defence, and Vitality
    for round_number, round_df in df.groupby('Round'):
        player_attack = {}
        player_defense = {}

        # Extracting relevant columns
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

        # Calculating vitality: Count of actions that are not 'PASS'
        vitality = round_df.groupby('Source')['Action_Description'].apply(
            lambda x: x[(x.notna()) & (x != 'pass')].count()
        )

        for i, player in enumerate(players):
            if player is not None:
                df.loc[(df['Round'] == round_number) & (df['Source'] == player), 'Attack'] = player_attack[
                    player]
                df.loc[(df['Round'] == round_number) & (df['Source'] == player), 'Defense'] = \
                    player_defense[player]
                df.loc[(df['Round'] == round_number) & (df['Source'] == player), 'Vitality'] = vitality[
                    player]
            else:
                df.loc[(df['Round'] == round_number) & (df['Source'] == player), ['Attack', 'Defense',
                                                                                              'Vitality']] = 0

    # df = df[df['Action_Description'].notna()]
    result_df = df[['Match', 'Round', 'Source', 'Attack', 'Defense', 'Vitality']]

    return result_df


base_path = "/usr/local/src/robot/cognitiveInteraction/SimulationChefsHat/MetricsChefsHat/Datasets/Dataset.pkl"  # change
# here with the path of your dataset

df = pd.read_pickle(base_path)
df = df.reset_index(drop=True)



# Calculate scores
for game in df['Match'].unique():
    if game == 0:
        continue
    # Filter the DataFrame for the current game
    game_df = df[(df['Match'] == game) & (df['Source'] != 'SYSTEM')]
    finish_index = game_df[game_df['Player_Finished'] == True].index.min()
    game_df = game_df.loc[:finish_index]

    # Calculate the metrics Attack, Defense and Vitality for the current game
    results = calculate_scores(game_df, game)

    # Create a PlayerAnalysis instance and generate the plot for Attack, Defense, Vitality and Eccentricity
    # There are multiple possible plots:
    # - radar_chart_tot: Plot a single radar chart FOR GAME that is the collection of the mean of the three metrics
    # - radar_chart: Plot a single radar cart FOR ROUND with the sum of each of the metrics for the specific round
    # - self_plots_tot: Plot eccentricity metric as a boxplot for a game
    # - self_plots: Plot eccentricity metric as barplot for each action done by each of the player follow the sequence
    #   of the actions
    # - stack_plots_sing: Plot singular plot of attack, defense and vitality as lineplot for each player

    metrics = PlayerAnalysis(results)
    # Example of using one of the plot functions for Attack, Defense and Vitality
    metrics.radar_chart(f'test{game}.png')


    # Example of eccentricity plot
    analysis = PlayerAnalysis(game_df)
    # analysis.self_plots(f'singletest{game}.png')

