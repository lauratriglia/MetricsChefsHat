from sys import platform

import CalculateMetrics
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.interpolate import griddata

class PlayerAnalysis:
    def __init__(self, df):
        self.df = df

    def radar_chart(self, filename):
        # Plot a single radar cart FOR ROUND with the sum of each of the metrics for the specific round
        # The original df is cut when a player has finished
        metrics = ['Attack', 'Defense', 'Vitality']
        player_types = self.df['Source'].unique()
        palette = sns.color_palette("Set2", len(player_types))
        colors = dict(zip(player_types, palette))
        rounds = sorted(self.df['Round'].unique())
        fig, axs = plt.subplots(nrows=1, ncols=len(rounds), figsize=(20, 6), subplot_kw=dict(polar=True))
        if len(rounds) == 1:
            axs = [axs]  # Convert to list for consistency

        for i, round_num in enumerate(rounds):
            self._plot_radar(axs[i], self.df, metrics, player_types, round_num, colors, include_legend=(i == 0))

        plt.tight_layout()
        fig.suptitle('Radar Chart by Round and Source', size=20, y=1.05)
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig

    def radar_chart_tot(self, filename):
        # Plot a single radar chart FOR GAME that is the collection of the mean of the three metrics
        # The mean is related to the specific player so the plot will have a different color for each of the players
        metrics = ['Attack', 'Defense', 'Vitality']
        player_types = self.df['Source'].unique()
        palette = sns.color_palette("Set2", len(player_types))
        colors = dict(zip(player_types, palette))
        mean_values = self.df.groupby('Source')[metrics].mean()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        self._plot_radar_tot(ax, mean_values, metrics, player_types, colors)
        plt.tight_layout()
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig

    def self_plots_tot(self, filename):
        # Plot eccentricity metric as boxplot for each game
        visualization_df, _ = CalculateMetrics.eccentricity_df(self.df)

        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(x='Source', y='Differences', data=visualization_df, palette="Set2")
        plt.xlabel('Player Type', fontsize=14)
        plt.ylabel('Eccentricity', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig


    def self_plots(self, filename):
        # Plot eccentricity metric as barplot for each action done by each of the player follow the sequence
        # of the actions
        visualization_df, max_value = CalculateMetrics.eccentricity_df(self.df)
        player_types = visualization_df['Source'].unique()
        palette = sns.color_palette("Set2", len(player_types))  # Use a predefined palette
        color_mapping = dict(zip(player_types, palette))

        # Create the main figure
        rounds = sorted(visualization_df['Round'].unique())
        fig, axes = plt.subplots(nrows=1, ncols=len(rounds), figsize=(14, 5), sharey=True)


        # Add subplots for the bar plots
        for i, round_value in enumerate(rounds):
            ax = axes[i]
            subset = visualization_df[visualization_df['Round'] == round_value]

            # Create bar plot with adjusted settings
            sns.barplot(data=subset, x='Action Count', y='Differences', hue='Source',
                        palette=color_mapping, ax=ax, dodge=True)

            ax.legend_.remove()

            # Add titles and labels
            ax.set_title(f'Round {round_value}')
            ax.set_xlabel('Action Count')
            if i == 0:
                ax.set_ylabel('Differences')
            ax.set_ylim(-0.03, max_value)

        # Adjust layout to prevent overlap
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title='Player')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin for the legend
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig

    def stack_plots_sing(self, filename):
        # Plot singular plot of attack, defense and vitality as lineplot for each player
        pivoted_attack = self.df.pivot_table(index='Round', columns='Source', values='Attack', fill_value=0)
        pivoted_defense = self.df.pivot_table(index='Round', columns='Source', values='Defense', fill_value=0)
        pivoted_vitality = self.df.pivot_table(index='Round', columns='Source', values='Vitality', fill_value=0)

        rounds = pivoted_attack.index
        player_types = self.df['Source'].unique()
        palette = sns.color_palette("Set2", len(player_types))
        color_mapping = dict(zip(player_types, palette))

        self._plot_stat(pivoted_attack, 'Attack', rounds, player_types, color_mapping, f'{filename}_attack.png')
        self._plot_stat(pivoted_defense, 'Defense', rounds, player_types, color_mapping, f'{filename}_defense.png')
        self._plot_stat(pivoted_vitality, 'Vitality', rounds, player_types, color_mapping, f'{filename}_vitality.png')


    def _plot_radar(self, ax, data, metrics, player_types, round_num, colors, include_legend):
        # Plot radar chart
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Plot each player type
        for player_type in player_types:
            subset = data[data['Source'] == player_type]
            means = subset[subset['Round'] == round_num][metrics].mean()

            values = means.tolist()
            values += values[:1]  # Close the circle
            ax.plot(angles, values, label=player_type, color=colors[player_type], linewidth=2)
            ax.fill(angles, values, color=colors[player_type], alpha=0.25)

        # Set labels
        ax.set_ylim(0, 3)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f'Round {round_num}', size=16, color='black')

        # Add legend only if specified
        if include_legend:
            ax.legend(loc='lower right', title='Player')


    def _plot_radar_tot(self, ax, mean_values, metrics, player_types, colors):
        num_vars = len(metrics)
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Plot each player type
        for player_type in player_types:
            values = mean_values.loc[player_type].tolist()
            values += values[:1]  # Close the circle

            ax.plot(angles, values, label=player_type, color=colors[player_type], linewidth=2)
            ax.fill(angles, values, color=colors[player_type], alpha=0.25)

        # Set labels
        ax.set_ylim(0, 3)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Metrics by Player Type', size=16, color='black')

        ax.legend(loc='lower right', title='Player Type')

    def _plot_stat(self, data, stat_name, rounds, player_types, colors, filename):
        fig =  plt.figure(figsize=(10, 6))
        # Loop over each player type and plot using seaborn lineplot
        for player_type in player_types:
            sns.lineplot(x=rounds, y=data[player_type], label=player_type, color=colors[player_type])

        # Titles and labels 
        plt.ylim(0, 1.5)
        plt.title(f'{stat_name} by Round')
        plt.xlabel('Round')
        plt.ylabel(stat_name)
        plt.legend(loc='upper right')
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig

    def stack_barplots_sing(self, filename_prefix):
        # Plot singular barplot of attack, defense and vitality for each player
        pivoted_attack = self.df.pivot_table(index='Round', columns='Source', values='Attack', fill_value=0)
        pivoted_defense = self.df.pivot_table(index='Round', columns='Source', values='Defense', fill_value=0)
        pivoted_vitality = self.df.pivot_table(index='Round', columns='Source', values='Vitality', fill_value=0)

        rounds = pivoted_attack.index
        player_types = self.df['Source'].unique()
        palette = sns.color_palette("Set2", len(player_types))
        color_mapping = dict(zip(player_types, palette))

        self._plot_stat_bar(pivoted_attack, 'Attack', rounds, player_types, color_mapping, f'{filename_prefix}_attack_bar.png')
        self._plot_stat_bar(pivoted_defense, 'Defense', rounds, player_types, color_mapping, f'{filename_prefix}_defense_bar.png')
        self._plot_stat_bar(pivoted_vitality, 'Vitality', rounds, player_types, color_mapping, f'{filename_prefix}_vitality_bar.png')

    def _plot_stat_bar(self, data, stat_name, rounds, player_types, colors, filename):
        fig = plt.figure(figsize=(10, 6))
        bar_width = 0.8 / len(player_types)
        # Calculate std for error bars
        std_data = self.df.pivot_table(index='Round', columns='Source', values=stat_name, aggfunc='std', fill_value=0)
        for idx, player_type in enumerate(player_types):
            plt.bar(
                rounds + idx * bar_width,
                data[player_type],
                width=bar_width,
                label=player_type,
                color=colors[player_type],
                yerr=std_data[player_type],
                capsize=5
            )
        plt.ylim(0, 1.5)
        plt.title(f'{stat_name} by Round (Barplot)')
        plt.xlabel('Round')
        plt.ylabel(stat_name)
        plt.legend(loc='upper right')
        plt.savefig(filename)
        print("Figure saved")
        return fig, True

    def barplot_overall_game(self, filename, metrics=None, plot_type='bar'):
        """
        Plot barplot of overall game metrics for each player, with error bars (std).
        You can choose which metrics to plot by passing a list to 'metrics'.
        Error bars are automatically calculated by seaborn (ci argument).
        """
        if metrics is None:
            metrics = ['Attack', 'Defense', 'Vitality']
            # Aggregate mean and std for each player and metric
        mean_metrics = self.df.groupby('Source')[metrics].mean().reset_index()
        std_metrics = self.df.groupby('Source')[metrics].std().reset_index()

        sources = mean_metrics['Source'].tolist()
        n_sources = len(sources)
        if metrics is None:
            metrics = ['Attack', 'Defense', 'Vitality']
        if plot_type == 'box':
            melted = self.df.melt(id_vars='Source', value_vars=metrics, var_name='Metric', value_name='Value')
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=melted, x='Source', y='Value', hue='Metric', palette="Set2")
            plt.title('Overall Game Metrics by Player (Boxplot)')
            plt.xlabel('Player')
            plt.ylabel('Value')
            plt.legend(title='Metric')
            plt.tight_layout()
            plt.savefig(filename)
            print("Figure saved")
            return plt.gcf(), True
        
        mean_metrics = self.df.groupby('Source')[metrics].mean().reset_index()
        std_metrics = self.df.groupby('Source')[metrics].std().reset_index()
        sources = mean_metrics['Source'].tolist()
        n_sources = len(sources)
        n_metrics = len(metrics)
        group_width = 0.8
        bar_width = group_width / n_metrics

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(n_sources)
        palette = sns.color_palette("Set2", n_metrics)

        for i, metric in enumerate(metrics):
            means = mean_metrics[metric].values
            stds = std_metrics[metric].values
            ax.bar(x + i * bar_width - group_width/2 + bar_width/2, means, bar_width,
                    yerr=stds, capsize=5, label=metric, color=palette[i])

        ax.set_xticks(x)
        ax.set_xticklabels(sources)
        ax.set_title('Overall Game Metrics by Player')
        ax.set_xlabel('Player')
        ax.set_ylabel('Mean Value')
        ax.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(filename)
        print("Figure saved")
        return fig, True

    def plot_multiple_line(self, filename):
        player_name = 'LARGER_VALUE_01'
        player_data = self.df[self.df['Source'] == player_name]
        # Get the unique games
        games = player_data['Match'].unique()

        # Number of games
        n_games = len(games)

        # Create subplots - define rows and columns dynamically depending on number of games
        fig, axes = plt.subplots(nrows=n_games, ncols=1, figsize=(10, 5 * n_games), sharex=True)

        # If there's only one game, axes may not be a list, so we wrap it in a list
        if n_games == 1:
            axes = [axes]

        # Loop over each game and create a subplot for each one
        for i, game in enumerate(games):
            # Filter the data for the current game
            game_data = player_data[player_data['Match'] == game]

            # Plot the attack values for this game
            axes[i].plot(game_data['Round'], game_data['Attack'], label=f'Attack (Game {game})', marker='o')

            # Set titles and labels for each subplot
            axes[i].set_title(f'Attack for Game {game}')
            axes[i].set_ylabel('Attack')
            axes[i].legend()

        # Set common x-label (Rounds) at the bottom
        plt.xlabel('Round')

        # Show the plot for this player
        plt.tight_layout()
        plt.savefig(filename)


    def boxplot_players(self, filename, metric):
        players = self.df['Source'].unique()
        n_players = len(players)

        # Calculate the number of rows and columns for the grid (for example 2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots - grid structure based on number of rows and columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Loop through each player and plot the boxplot in the corresponding subplot
        for i, player in enumerate(players):
            player_data = self.df[self.df['Source'] == player]
            # Filter the data for the current player
            if metric == 'Number_Pass':
                pass_count = player_data.groupby('Cluster')['Number_Pass'].mean().reindex(['Beginning', 'Middle', 'End'],
                                                                                         fill_value=0)

                sns.barplot(x=pass_count.index, y=pass_count.values, ax=axes[i])
                axes[i].set_title(f'{metric} Values for {player}')
                axes[i].set_xlabel('Cluster')
                axes[i].set_ylabel(f'{metric}')

            else:

                # Plot a boxplot for 'attack' grouped by the 'Cluster' (Beginning, Middle, End) in the current subplot
                sns.boxplot(x='Cluster', y=metric, data=player_data, order=['Beginning', 'Middle', 'End'], ax=axes[i])

                # Set title and labels for each subplot
                axes[i].set_title(f'{metric} Values for {player}')
                axes[i].set_xlabel('Cluster')
                axes[i].set_ylabel(f'{metric}')

        # Hide any unused subplots if the number of players is not a perfect multiple of ncols
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the figure
        plt.savefig(f"{filename}{metric}.png")

    def boxplots_oldMetrics(self,filename, metrics=['Number_passes', 'Number_discard']):
        players = self.df['Source'].unique()  # Unique players in the 'Source' column
        n_players = len(players)

        # Calculate the number of rows and columns for the grid (e.g., 2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots with a grid structure based on the number of rows and columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Loop through each player and plot the boxplot in the corresponding subplot
        for i, player in enumerate(players):
            # Filter the data for the current player
            player_data = self.df[self.df['Source'] == player]

            # Melt the DataFrame to long format for plotting multiple metrics in the same boxplot
            player_melted = player_data.melt(id_vars=['Source'], value_vars=metrics, var_name='Metric',
                                             value_name='Value')

            # Plot a boxplot for the specified metrics in the current subplot
            sns.boxplot(x='Metric', y='Value', data=player_melted, ax=axes[i])

            # Set title and labels for each subplot
            axes[i].set_title(f'{", ".join(metrics)} for {player}')
            axes[i].set_xlabel('Metric')
            axes[i].set_ylabel('Values')

        # Hide any unused subplots if the number of players is not a perfect multiple of ncols
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the figure
        plt.savefig(f"{filename}.png")


    def percentage_rounds(self, filename):
        players = self.df['Source'].unique()

        for player in players:
            player_data = self.df[self.df['Source'] == player]

            plt.figure(figsize=(10, 6))

            # Plot each match's line for the 'Attack' metric
            for match in player_data['Match'].unique():
                match_data = player_data[player_data['Match'] == match]
                plt.plot(match_data['Round_Percent'], match_data['Attack'])

            # Set plot details
            plt.title(f'Attack Over Time (as Percentage) for Player {player}')
            plt.xlabel('Round Percentage (%)')
            plt.ylabel('Attack')
            plt.grid(True)
            plt.tight_layout()

            # Display the plot
            plt.savefig(filename)


    def lineplot_bme(self,filename, metric):
        players = self.df['Source'].unique()

        n_players = len(players)

        # Calculate the number of rows and columns for the grid (for example 2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots - grid structure based on number of rows and columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Loop through each player and plot the boxplot in the corresponding subplot
        for i, player in enumerate(players):
            # Filter data for the specific player
            player_data = self.df[self.df['Source'] == player]
            player_data = player_data.reset_index(drop=True)

            # Group the data by the more specific clusters (e.g., B1-B10, M1-M10, E1-E10)
            agg_data = player_data.groupby('Cluster').agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std'),
            ).reset_index()

            # Define the proper order for clusters: B1-B10, M1-M10, E1-E10
            cluster_order = ['B1', 'B2', 'B3', 'B4', 'B5',
                             'M1', 'M2', 'M3', 'M4', 'M5',
                             'E1', 'E2', 'E3', 'E4', 'E5']

            # Make sure 'Cluster' column is treated as categorical with this specific order
            agg_data['Cluster'] = pd.Categorical(agg_data['Cluster'], categories=cluster_order, ordered=True)
            agg_data = agg_data.sort_values('Cluster')
            agg_data['Phase'] = agg_data['Cluster'].apply(
                lambda x: 'Beginning' if 'B' in x else 'Middle' if 'M' in x else 'End')

            # Plot the mean 'Attack' line with markers
            axes[i].plot(agg_data['Cluster'], agg_data['mean_metric'], marker='o', linestyle='-', color='b',
                     label=f'Mean {metric}')

            # Add the standard deviation as a shaded area around the mean
            axes[i].fill_between(
                agg_data['Cluster'],
                agg_data['mean_metric'] - agg_data['std_metric'],  # Mean - Std
                agg_data['mean_metric'] + agg_data['std_metric'],  # Mean + Std
                color='b',
                alpha=0.2,  # Transparency level
                label='Standard Deviation'
            )

            axes[i].axvline(x=5, color='red', linestyle='--')  # Line between Beginning and Middle
            axes[i].axvline(x=10, color='red', linestyle='--')

            # axes[i].axvspan(0, 5, color='lightgreen', alpha=0.3, label='Beginning')
            # axes[i].axvspan(5, 10, color='lightblue', alpha=0.3, label='Middle')
            # axes[i].axvspan(10, 15, color='lightcoral', alpha=0.3, label='End')

            # Adjust the x-axis ticks to show only 'Beginning', 'Middle', and 'End'
            axes[i].set_xticks([2.5, 7.5, 12.5])  # Positions for tick marks
            axes[i].set_xticklabels(['Beginning', 'Middle', 'End'])
            # Add labels and title
            axes[i].set_title(f'{player} - {metric}')
            axes[i].set_xlabel('Game Phase')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)

            axes[i].legend()

        # Show the plot
        plt.tight_layout()

        plt.savefig(f"{filename}{metric}_line.png")


    def plots_3d_time_metrics(self,filename):
        players = self.df['Source'].unique()
        n_players = len(players)

        # Calculate the number of rows and columns for the grid (2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots - grid structure based on number of rows and columns
        fig = plt.figure(figsize=(12, 6 * nrows))

        # Loop through each player and plot the 3D surface plot in the corresponding subplot
        for i, player in enumerate(players):
            # Filter data for the specific player
            player_data = self.df[self.df['Source'] == player]
            player_data = player_data.reset_index(drop=True)

            # Group the data by the more specific clusters (e.g., B1-B5, M1-M5, E1-E5)
            agg_data = player_data.groupby('Cluster').agg(
                mean_attack=('Attack', 'mean'),
                mean_defense=('Defense', 'mean'),
                mean_vitality=('Vitality', 'mean'),
            ).reset_index()

            # Define the proper order for clusters: B1-B5, M1-M5, E1-E5
            cluster_order = ['B1', 'B2', 'B3', 'B4', 'B5',
                             'M1', 'M2', 'M3', 'M4', 'M5',
                             'E1', 'E2', 'E3', 'E4', 'E5']

            # Ensure 'Cluster' column is treated as categorical with this specific order
            agg_data['Cluster'] = pd.Categorical(agg_data['Cluster'], categories=cluster_order, ordered=True)
            agg_data = agg_data.sort_values('Cluster')

            # Extract mean values for Attack, Defense, and Vitality
            attack_means = agg_data['mean_attack']
            defense_means = agg_data['mean_defense']
            bin_indices = np.arange(1, len(agg_data) + 1)  # B1-B5, M1-M5, E1-E5 as bin indices
            vitality_means = agg_data['mean_vitality']

            # Create a mesh grid for attack and defense
            bin_grid, attack_grid = np.meshgrid(bin_indices, np.linspace(min(attack_means), max(attack_means), 100))

            # Interpolate defense values over the mesh grid
            defense_grid = griddata((bin_indices, attack_means), defense_means, (bin_grid, attack_grid),
                                    method='linear')

            # Interpolate vitality values over the mesh grid for coloring
            vitality_grid = griddata((bin_indices, attack_means), vitality_means, (bin_grid, attack_grid),
                                     method='linear')

            # Create a 3D subplot for each player
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

            # Plot the surface
            surface = ax.plot_surface(bin_grid, attack_grid, defense_grid, facecolors=plt.cm.viridis(vitality_grid),
                                      rstride=1, cstride=1, linewidth=0, antialiased=False)

            # Set axis labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Mean Defense')
            ax.set_zlabel('Mean Attack')
            ax.set_title(f'{player} - Attack vs Defense')

            ax.set_ylim(0, 1.2)  # Assuming Attack range between 0 and 2
            ax.set_zlim(0, 1.2)

            # Add color bar for vitality
            cbar = fig.colorbar(surface, ax=ax, pad=0.1)
            cbar.set_label('Mean Vitality')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{filename}_3d_surface_time.png")


    def plots_3d_metrics(self, filename):
        players = self.df['Source'].unique()
        n_players = len(players)

        # Calculate the number of rows and columns for the grid (2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots - grid structure based on number of rows and columns
        fig = plt.figure(figsize=(12, 6 * nrows))

        eccentricity_min, eccentricity_max = 0.00, 0.04

        # Create a normalization instance for the color bar
        norm = plt.Normalize(vmin=eccentricity_min, vmax=eccentricity_max)
        # Loop through each player and plot the 3D surface plot in the corresponding subplot
        for i, player in enumerate(players):
            # Filter data for the specific player
            player_data = self.df[self.df['Source'] == player]
            player_data = player_data.reset_index(drop=True)

            # Group the data by the more specific clusters (e.g., B1-B5, M1-M5, E1-E5)
            agg_data = player_data.groupby('Cluster').agg(
                mean_attack=('Attack', 'mean'),
                mean_defense=('Defense', 'mean'),
                mean_vitality=('Vitality', 'mean'),
                mean_eccentricity=('Eccentricity', 'mean'),
            ).reset_index()

            # Define the proper order for clusters: B1-B5, M1-M5, E1-E5
            cluster_order = ['B1', 'B2', 'B3', 'B4', 'B5',
                             'M1', 'M2', 'M3', 'M4', 'M5',
                             'E1', 'E2', 'E3', 'E4', 'E5']

            # Ensure 'Cluster' column is treated as categorical with this specific order
            agg_data['Cluster'] = pd.Categorical(agg_data['Cluster'], categories=cluster_order, ordered=True)
            agg_data = agg_data.sort_values('Cluster')

            # Extract mean values for Attack, Defense, Vitality, and Eccentricity
            attack_means = agg_data['mean_attack']
            defense_means = agg_data['mean_defense']
            vitality_means = agg_data['mean_vitality']
            eccentricity_means = agg_data['mean_eccentricity']

            # Create a mesh grid for attack and defense
            attack_grid, defense_grid = np.meshgrid(np.linspace(0, 2, 100),  # Set x limit between 0 and 2
                                                    np.linspace(0, 2, 100))  # Set y limit between 0 and 2

            # Interpolate vitality and eccentricity values over the grid
            vitality_grid = griddata((attack_means, defense_means), vitality_means,
                                     (attack_grid, defense_grid), method='cubic')

            eccentricity_grid = griddata((attack_means, defense_means), eccentricity_means,
                                         (attack_grid, defense_grid), method='cubic')

            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

            # Plot the surface with vitality as the z-axis and eccentricity as the color
            surface = ax.plot_surface(attack_grid, defense_grid, vitality_grid,
                                      facecolors=cm.viridis(norm(eccentricity_grid)),
                                      edgecolor='none')

            # Set axis limits for x, y, and z
            ax.set_xlim([0, 1.2])
            ax.set_ylim([0, 1.2])
            ax.set_zlim([0, 1.0])

            # Set labels and title
            ax.set_title(f'{player} - Attack, Defense, Vitality')
            ax.set_xlabel('Attack')
            ax.set_ylabel('Defense')
            ax.set_zlabel('Vitality')

        # Adjust layout to prevent overlap
            mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)  # Use the same normalization for the color bar
            mappable.set_array([])  # Set an empty array since we are using facecolors
            cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
            cbar.set_label('Eccentricity')  # Label for color bar
        plt.tight_layout()
        plt.savefig(f"{filename}_3d_surface_eccentricity.png")

    def shadow_mean_line(self,filename, metric):
        players = self.df['Source'].unique()

        n_players = len(players)

        # Calculate the number of rows and columns for the grid (for example 2x2 structure)
        ncols = 2
        nrows = math.ceil(n_players / ncols)

        # Create subplots - grid structure based on number of rows and columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Loop through each player and plot the boxplot in the corresponding subplot
        for i, player in enumerate(players):
            # Filter data for the specific player
            player_data = self.df[self.df['Source'] == player]
            player_data = player_data.reset_index(drop=True)

            # Generate new percentage values (0 to 100%)
            percentages = np.linspace(0, 100, num=100)

            # Interpolate the current metric values
            interpolated_metric = np.interp(percentages, player_data['Round_Percent'], player_data[metric])

            # Group by Round_Percent to calculate mean and standard deviation for the current metric
            agg_data = player_data.groupby('Round_Percent').agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std'),
            ).reset_index()

            # Interpolate the standard deviation for smoother shading
            interpolated_std = np.interp(percentages, agg_data['Round_Percent'], agg_data['std_metric'].fillna(0))

            # Plot the interpolated mean values
            axes[i].plot(percentages, interpolated_metric, marker='o', linestyle='-', color='b',
                    label=f'Interpolated {metric}')

            # Add the standard deviation as a shaded area around the mean
            axes[i].fill_between(
                percentages,
                interpolated_metric - interpolated_std,
                interpolated_metric + interpolated_std,
                color='b',
                alpha=0.2,
                label='Standard Deviation'
            )

            # Set title and labels for each subplot
            axes[i].set_title(f'{player} - {metric}')
            axes[i].set_xlabel('Round Percentage (%)')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
            axes[i].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f"{filename}{metric}_Line.png")

    def chosen_mandatory_pass(self, filename):
        players = self.df['Source'].unique()
        self.df['Mandatory_Pass'] = self.df['Eccentricity'].apply(lambda x: 1 if x == -0.01 else 0)
        self.df['Chosen_Pass'] = self.df['Eccentricity'].apply(lambda x: 1 if x == -0.03 else 0)

        self.df['Phase'] = self.df['Cluster'].apply(
            lambda x: 'Beginning' if 'B' in x else 'Middle' if 'M' in x else 'End')
        phase_order = ['Beginning', 'Middle', 'End']
        self.df['Phase'] = pd.Categorical(self.df['Phase'], categories=phase_order, ordered=True)
        pass_means_phase = self.df.groupby(['Source', 'Phase']).agg(
            mean_mandatory_passes=('Mandatory_Pass', 'mean'),
            mean_chosen_passes=('Chosen_Pass', 'mean')
        ).reset_index()

        # Step 4: Prepare for plotting
        n_players = len(players)
        ncols = 2  # 2 columns in the grid
        nrows = np.ceil(n_players / ncols).astype(int)  # Calculate required rows

        # Create a figure with subplots (2x2 structure)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Step 5: Loop through each player and create a bar plot for each player
        for i, player in enumerate(players):
            # Filter the data for the current player
            player_data = pass_means_phase[pass_means_phase['Source'] == player]

            # Get the mean of mandatory and chosen passes by phase
            mean_mandatory = player_data['mean_mandatory_passes']
            mean_chosen = player_data['mean_chosen_passes']
            phases = player_data['Phase']  # 'Beginning', 'Middle', 'End'

            # Set bar width and positions for plotting
            bar_width = 0.35
            index = np.arange(len(phases))

            # Plot for the current player on the respective subplot
            axes[i].bar(index, mean_mandatory, bar_width, label='Mandatory Passes (-0.01)', color='cornflowerblue')
            axes[i].bar(index + bar_width, mean_chosen, bar_width, label='Chosen Passes (-0.03)', color='limegreen')

            # Set the title, labels, and ticks
            axes[i].set_title(f'{player} - Mean Passes by Phase')
            axes[i].set_xlabel('Game Phase')
            axes[i].set_ylabel('Mean Passes')
            axes[i].set_xticks(index + bar_width / 2)
            axes[i].set_xticklabels(phases)
            axes[i].legend()

        # Step 6: Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig(f"{filename}Ecc_pass.png")

