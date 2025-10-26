import pandas as pd
import numpy as np
import re
from datetime import datetime

# Clean if NOT
# add wilson score
# add elo
# Lock temporally
# Split if Not 20/80 (time aware too)

df = pd.read_csv("Horse_Racing_ML/Pt2/merged_horse_racing_data.csv")

print(df.head(0))


categorical_columns = [
    "track_name_clean",
    "going_clean",
    "distance_clean",
    "horse_name_clean",
    "jockey_clean",
    "trainer_clean",
    "claims_clean"
]
temporal_cols = df[['race_date', 'race_time']]#2 
pre_numeric_cols = df[['distance', 'going', 'claims', 'lost_by_length', 'age', 'weight', 'rating', 'race_position']]#8
drop_cols = df[['batch_number', 'url_index_in_batch', 'first_place_found','first_place_found_clean','source_file']]#3

def extract_date(old):
    new = old.split('|', 1)[-1]
    #drop ordinals
    cleaned_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', new)
    #Match format:
    date_object = datetime.strptime(cleaned_string, "%d %B %Y %H:%M")
    #re-Format
    desired_format = date_object.strftime("%d/%m/%Y")

    return desired_format
# print(df['track_name'].head(4).apply(extract_date))

unique_goings = df['going_clean'].unique()
going_vocab = dict(
    (going, i + 1) for i, going in enumerate(unique_goings)
)
df['going_encoded'] = df['going_clean'].apply(lambda x: going_vocab[x])
# print(df[['going_encoded', 'going_clean']].head(4)) 

print(df['lost_by_length_clean'].head(10))

# top 4 CATS:
# | Feature                 |   P_Value |   Chi2_Stat | Notes                                         |
# |:------------------------|----------:|------------:|:----------------------------------------------|
# | trainer_clean           |      0.00 |   3756.84   | Categories Kept: 602 (incl. 'Other')          |
# | jockey_clean            |      0.00 |   3891.42   | Categories Kept: 407 (incl. 'Other')          |
# | claims_clean            |      0.00 |    310.389  | Categories Kept: 8 (incl. 'Other')            |
# | horse_name_clean        |      0.09 |    236.748  | Categories Kept: 210 (incl. 'Other')          |
# |:------------------------|----------:|------------:|:----------------------------------------------|
