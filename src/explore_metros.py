import pandas as pd
import numpy as np

# Get all unique metros from training data
train = pd.read_csv('data/binaaz_train.csv')
all_locations = train['locations'].dropna().tolist()

metro_set = set()
for loc in all_locations:
    parts = str(loc).split('\n')
    for p in parts:
        p = p.strip()
        if p.endswith(' m.'):
            metro_set.add(p.replace(' m.', ''))

print('=== UNIQUE METRO NAMES IN TRAINING DATA ===')
for m in sorted(metro_set):
    print(f'  "{m}"')
print(f'\nTotal: {len(metro_set)} unique metro stations')

# Get landmarks that look like metro stations
landmarks = pd.read_excel('baku_coordinates.xlsx')
print('\n=== METRO-LIKE LANDMARKS ===')
metro_landmarks = landmarks[landmarks['Title'].str.contains('|'.join([
    'Icheri', 'Sahil', 'May', 'Ganjlik', 'Narimanov', 'Bakmil', 'Ulduz',
    'Koroglu', 'Garayev', 'Neftchi', 'Doslugu', 'Ahmedli', 'Aslanov',
    'Nizami', 'Akademi', 'Inshaatchi', 'Yanvar', 'Ajami', 'Nasimi',
    'Azadlig', 'Darnagul', 'Jabbarli', 'Hatai', 'Avtovagzal', 'Ganjavi'
]), case=False, na=False)]
print(metro_landmarks[['Title', 'Latitude', 'Longitude']].to_string())
