import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# (a) & (b) Load Salaries.csv and plot
df = pd.read_csv('Salaries.csv')
print(df.head())

# (b) Basic seaborn plot - salary distribution
sns.histplot(data=df, x='salary')
plt.title('Salary Distribution')
plt.savefig('b_histplot.png')
plt.clf()

# (c) Statistical estimation plots

# lmplot - linear regression of salary over yearID
sns.lmplot(data=df, x='yearID', y='salary')
plt.title('Salary vs Year (lmplot)')
plt.savefig('c_lmplot.png')
plt.clf()

# catplot - salary by team (top 5 teams for readability)
top_teams = df['teamID'].value_counts().head(5).index
df_top = df[df['teamID'].isin(top_teams)]
sns.catplot(data=df_top, x='teamID', y='salary', kind='box')
plt.title('Salary by Team (catplot)')
plt.savefig('c_catplot.png')
plt.clf()

# relplot - salary trend over years
sns.relplot(data=df, x='yearID', y='salary', kind='line')
plt.title('Salary Trend Over Years (relplot)')
plt.savefig('c_relplot.png')
plt.clf()

# (d) Boxplot - salary by league
sns.boxplot(data=df, x='lgID', y='salary')
plt.title('Salary by League (boxplot)')
plt.savefig('d_boxplot.png')
plt.clf()

# (e) Pairplot
sns.pairplot(df[['yearID', 'salary']])
plt.suptitle('Pairplot', y=1.02)
plt.savefig('e_pairplot.png')
plt.clf()

# Jointplot - yearID vs salary
sns.jointplot(data=df, x='yearID', y='salary', kind='reg')
plt.suptitle('Jointplot', y=1.02)
plt.savefig('e_jointplot.png')
plt.clf()

print("All plots saved!")