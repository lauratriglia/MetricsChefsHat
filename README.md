#Plot Generation Overview

This script produces two key metrics: Competitiveness and Eccentricity. These metrics are objective and are calculated for each player during a round, offering insights into both individual behavior and the overall dynamics of the game.
Competitiveness Metric Components:

-    Attack: The number of passes after a "Discard" action within a round.
-    Defense: The number of passes before a "Discard" action within a round.
-    Vitality: The total number of discards performed in a round.

Eccentricity Metric:

-    This metric measures the probability of selecting the most common action during a round.

##GeneratorPlots.py

This script utilizes the PlayerAnalysis class to generate visualizations for Attack, Defense, Vitality, and Eccentricity. The available plot types include:

-    radar_chart_tot: Generates a radar chart for the entire game, displaying the mean values of the three Competitiveness metrics.
-    radar_chart: Creates a radar chart for a specific round, showing the total sum of each Competitiveness metric for that round.
-    self_plots_tot: Displays a boxplot of the Eccentricity metric for the entire game.
-    self_plots: Produces a bar plot of the Eccentricity metric, tracking each player’s actions in sequence.
-    stack_plots_sing: Creates individual line plots for Attack, Defense, and Vitality metrics, showcasing each player’s performance.

#Important Notes:

-    The folder contains an example dataset. However, to plot the Competitiveness metrics, use the DataFrame generated by the CalculateMetrics.py script. For plotting the Eccentricity metric, utilize the original dataset obtained from the experiment.
