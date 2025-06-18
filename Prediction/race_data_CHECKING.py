import pandas as pd

df = pd.read_csv("race_predictions_all_thresholds.csv")
df = df.drop_duplicates()
print(df.head(0))
# Create a boolean mask where predicted_winner == 1
winner = df['predicted_winner'] == 0

# Filter the DataFrame using the mask and select the desired columns
print(df[['track_name', 'horse_name', 'win_probability', 'predicted_winner']].sort_values('win_probability', ascending=False).head(60))


print("+" * 70)
# print(df.loc[winner,['horse_name']])
print(df['predicted_winner'].unique())
df1 = pd.read_csv('batch_horses_batch_0001_20250618_211112.csv')
df1 = df1.drop_duplicates()

winner = df1['race_position'] == "1st"

print(df1.loc[winner, ['horse_name','race_position']])


import pandas as pd

# Load your data
df = pd.read_csv('race_predictions_all_thresholds.csv')

# Ensure predicted_probability is numeric
df['win_probability'] = pd.to_numeric(df['win_probability'], errors='coerce')

# Drop NaNs just in case
df = df.dropna(subset=['win_probability', 'track_name'])

# Rank within each track_name group by predicted_probability (highest = 1)
df['probability_rank'] = df.groupby('track_name')['win_probability'] \
                           .rank(method='first', ascending=False) \
                           .astype(int)

# Optional: sort so it looks nice
df = df.sort_values(by=['track_name', 'probability_rank'])

# Preview result
print(df[['track_name', 'horse_name', 'win_probability', 'probability_rank', 'race_position']].head(40))
df = df[['track_name', 'horse_name', 'win_probability', 'probability_rank', 'race_position']]

df.to_csv('ranked_predictions.csv', index=False)


import pandas as pd

# Load and preprocess
df = pd.read_csv('ranked_predictions.csv')
df['win_probability'] = pd.to_numeric(df['win_probability'], errors='coerce')
df = df.dropna(subset=['win_probability', 'track_name', 'race_position'])

# Rank by probability
df['probability_rank'] = df.groupby('track_name')['win_probability'] \
                           .rank(method='first', ascending=False) \
                           .astype(int)

# Keep necessary columns
df = df[['track_name', 'horse_name', 'win_probability', 'probability_rank', 'race_position']]

# --- Convert race_position from string ordinal to integer ---
# Assumes values like "1st", "2nd", "3rd", etc.
df['race_position_int'] = df['race_position'].str.extract(r'(\d+)').astype(float)

# Drop rows where race_position couldn't be converted
df = df.dropna(subset=['race_position_int'])

# --- Calculate error ---
df['rank_error'] = (df['probability_rank'] - df['race_position_int']).abs()

# Print evaluation
print(df[['track_name', 'horse_name', 'probability_rank', 'race_position_int', 'rank_error']].head(40))

# Optional: get average rank error
mean_error = df['rank_error'].mean()
print(f"\nAverage Rank Error: {mean_error:.2f}")

xyz =df['win_probability'] == df['race_position_int'].count()


df['win_probability'] = df['win_probability'].round(3)

print(df[['horse_name', 'win_probability']].sort_values('win_probability', ascending=False))