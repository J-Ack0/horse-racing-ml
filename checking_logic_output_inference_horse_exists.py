import pandas as pd

# Load both files
lookup_df = pd.read_csv("V2_engineered_features.csv")

# Process lookup the same way your script does
if 'race_date' in lookup_df.columns:
    lookup_df['race_date'] = pd.to_datetime(lookup_df['race_date'], format='mixed', errors='coerce')
    lookup_df = lookup_df.sort_values('race_date').groupby('horse_name').last().reset_index()

lookup_df.set_index('horse_name', inplace=True)

# Test horse name
test_name = "Morning Mayhem"

print("="*60)
print("DEBUGGING LOOKUP FAILURE")
print("="*60)

print(f"\nTest horse: '{test_name}'")
print(f"Length: {len(test_name)}")
print(f"Repr: {repr(test_name)}")

print(f"\n✓ Horse in lookup index? {test_name in lookup_df.index}")

# Check for similar names
similar = [name for name in lookup_df.index if 'Morning' in name or 'Mayhem' in name]
print(f"\nSimilar names in lookup:")
for name in similar:
    print(f"  - '{name}' | Length: {len(name)} | Repr: {repr(name)}")
    print(f"    Match? {name == test_name}")
    
    # Character-by-character comparison
    if len(name) == len(test_name):
        for i, (c1, c2) in enumerate(zip(name, test_name)):
            if c1 != c2:
                print(f"    Diff at position {i}: '{c1}' vs '{c2}' (ord: {ord(c1)} vs {ord(c2)})")

# Show first few horses in lookup
print(f"\nFirst 10 horses in lookup_df.index:")
for i, name in enumerate(lookup_df.index[:10]):
    print(f"  {i+1}. '{name}'")

print(f"\nTotal horses in lookup: {len(lookup_df)}")