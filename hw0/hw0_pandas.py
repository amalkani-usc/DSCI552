import pandas as pd

# (a) & (b) Read the CSV file
df = pd.read_csv('Salaries.csv')

# (c) Set playerID as index, first row as header, skip second row
df = pd.read_csv('Salaries.csv', index_col='playerID', header=0, skiprows=[1])

# (d) Players in ATL or HOU with salary > 1,000,000
result = df[(df['teamID'].isin(['ATL', 'HOU'])) & (df['salary'] > 1000000)].index.tolist()
print("Player IDs in ATL or HOU with salary > $1M:")
print(result)

# (e) describe() stats for salary in ATL
atl_salary = df[df['teamID'] == 'ATL']['salary']
stats = atl_salary.describe()

print("\nATL Salary Stats:")
print(f"Mean:    {stats['mean']:.2f}")
print(f"Std Dev: {stats['std']:.2f}")
print(f"Min:     {stats['min']:.2f}")
print(f"Q1:      {stats['25%']:.2f}")
print(f"Median:  {stats['50%']:.2f}")
print(f"Q3:      {stats['75%']:.2f}")
print(f"Max:     {stats['max']:.2f}")

# (f) Create dictionary using iterrows()
data_dict = {col: [] for col in df.columns}

for _, row in df.iterrows():
    for col in df.columns:
        data_dict[col].append(row[col])

print("Dictionary keys:", list(data_dict.keys()))

# (g) Create dataframe from dictionary and rename headers to a, b, c, ...
import string

df2 = pd.DataFrame(data_dict)

# Generate letter headers: a, b, c, ... based on number of columns
new_headers = list(string.ascii_lowercase[:len(df2.columns)])
df2.columns = new_headers

print(df2.head())