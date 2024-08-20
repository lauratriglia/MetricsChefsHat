import pandas as pd
from PlotManager import PlayerAnalysis
import CalculateMetrics

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
    results = CalculateMetrics.calculate_scores(game_df, game)

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
    metrics.radar_chart_tot(f'test{game}.png')

    # Example of eccentricity plot
    analysis = PlayerAnalysis(game_df)
    analysis.self_plots(f'singletest{game}.png')
