import pandas as pd

df = pd.read_csv("german_credit_data.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Check what columns exist
print("COLUMNS:", df.columns.tolist())

# Check unique values in Checking account
print("\nChecking account values:")
print(df['Checking account'].value_counts(dropna=False))

# Check Credit amount range
print("\nCredit amount stats:")
print(df['Credit amount'].describe())