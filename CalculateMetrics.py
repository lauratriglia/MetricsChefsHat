import pandas as pd

def eccentricity_df(df):
    # Create a df that contains the eccentricity metric
    action_df = pd.read_pickle('random.pkl')
    action_df['Action'] = action_df['Action'].astype(str)
    max_value = action_df['Count'].max()
    visualization_data = []
    action_counts = {}
    df = df[df['Action_Type'] == 'DISCARD']
    for _, row in df.iterrows():
        possible_actions = row['Possible_Actions']
        action_done = row['Action_Description']
        player_type = row['Source']
        round_num = row['Round']
        for action in possible_actions:
            if action == 'pass':
                action_counts[action] = - max_value  # Special handling for 'pass'
            else:
                count = action_df[action_df['Action'] == action]['Count'].values
                action_counts[action] = count[0] if len(count) > 0 else 0

        # Add action_done to action_counts if it's not in possible_actions
        if action_done == 'pass':
            action_counts[action_done] = - max_value

        # Calculate differences
        man_pass = 0
        highest_prob = max(action_counts.values())
        differences = highest_prob - action_counts.get(action_done, 0)
        if highest_prob == - max_value and action_done == 'Pass':
            man_pass = 1
            differences = - 0.01
        if highest_prob != -max_value and action_done == 'Pass':
            man_pass = 2
            differences = - 0.03
        visualization_data.append({
            'Round': round_num,
            'Source': player_type,
            'Action Done': action_done,
            'Differences': differences,
            'Possible Action': possible_actions,
            'No poss': man_pass
        })

    # Convert the collected data into a DataFrame for visualization
    visualization_df = pd.DataFrame(visualization_data)
    visualization_df['Action Count'] = visualization_df.groupby(['Round', 'Source']).cumcount() + 1

    return visualization_df, max_value

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