import CalculateMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

        self._plot_stat(pivoted_attack, 'Attack', rounds, player_types, color_mapping, 'attack.png')
        self._plot_stat(pivoted_defense, 'Defense', rounds, player_types, color_mapping, 'defense.png')
        self._plot_stat(pivoted_vitality, 'Vitality', rounds, player_types, color_mapping, 'vitality.png')




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
        plt.title(f'{stat_name} by Round')
        plt.xlabel('Round')
        plt.ylabel(stat_name)
        plt.legend(loc='upper right')
        plt.savefig(filename)
        save_fig = True
        print("Figure saved")
        return fig, save_fig
