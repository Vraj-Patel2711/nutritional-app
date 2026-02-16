import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting nutritional analysis...")

# Check if dataset exists
if not os.path.exists('All_Diets.csv'):
    print("ERROR: All_Diets.csv not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    print("Please download the dataset from Kaggle and place it here.")
    exit(1)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('All_Diets.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display column names to verify structure
print(f"Columns in dataset: {df.columns.tolist()}")

# Handle missing values
print("\nHandling missing values...")
numeric_cols = ['Protein(g)', 'Carbs(g)', 'Fat(g)']
for col in numeric_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"Filled {missing_count} missing values in {col}")

# 1. Calculate average macronutrient content for each diet type
print("\n1. Calculating average macronutrients by diet type...")
avg_macros = df.groupby('Diet_type')[numeric_cols].mean().round(2)
print(avg_macros)
avg_macros.to_csv('outputs/avg_macros_by_diet.csv')

# 2. Find top 5 protein-rich recipes for each diet type
print("\n2. Top 5 protein-rich recipes by diet type:")
top_protein = df.sort_values('Protein(g)', ascending=False).groupby('Diet_type').head(5)
print(top_protein[['Diet_type', 'Recipe_name', 'Protein(g)']].head(10))
top_protein.to_csv('outputs/top_protein_recipes.csv')

# 3. Find diet type with highest protein content
highest_protein_diet = df.groupby('Diet_type')['Protein(g)'].mean().idxmax()
highest_protein_value = df.groupby('Diet_type')['Protein(g)'].mean().max()
print(f"\n3. Diet with highest average protein: {highest_protein_diet} ({highest_protein_value:.2f}g)")

# 4. Identify most common cuisines for each diet type
print("\n4. Most common cuisines by diet type:")
common_cuisines = df.groupby('Diet_type')['Cuisine_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
print(common_cuisines)

# 5. Create new metrics (ratios)
print("\n5. Creating new metrics (protein/carbs and carbs/fat ratios)...")
df['Protein_to_Carbs_ratio'] = df['Protein(g)'] / df['Carbs(g)'].replace(0, np.nan)
df['Carbs_to_Fat_ratio'] = df['Carbs(g)'] / df['Fat(g)'].replace(0, np.nan)
df['Protein_to_Fat_ratio'] = df['Protein(g)'] / df['Fat(g)'].replace(0, np.nan)

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Ratios calculated and saved")

# VISUALIZATIONS
print("\n6. Creating visualizations...")

# Set style
sns.set_style("whitegrid")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Nutritional Analysis by Diet Type', fontsize=16, fontweight='bold')

# Bar chart for average protein
ax1 = axes[0, 0]
avg_protein = avg_macros['Protein(g)'].sort_values(ascending=False)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(avg_protein)))
ax1.barh(avg_protein.index, avg_protein.values, color=colors)
ax1.set_title('Average Protein by Diet Type', fontsize=14)
ax1.set_xlabel('Average Protein (g)')

# Bar chart for average carbs
ax2 = axes[0, 1]
avg_carbs = avg_macros['Carbs(g)'].sort_values(ascending=False)
colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(avg_carbs)))
ax2.barh(avg_carbs.index, avg_carbs.values, color=colors)
ax2.set_title('Average Carbs by Diet Type', fontsize=14)
ax2.set_xlabel('Average Carbs (g)')

# Bar chart for average fat
ax3 = axes[1, 0]
avg_fat = avg_macros['Fat(g)'].sort_values(ascending=False)
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(avg_fat)))
ax3.barh(avg_fat.index, avg_fat.values, color=colors)
ax3.set_title('Average Fat by Diet Type', fontsize=14)
ax3.set_xlabel('Average Fat (g)')

# Heatmap for correlations
ax4 = axes[1, 1]
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax4, 
            xticklabels=['Protein', 'Carbs', 'Fat'],
            yticklabels=['Protein', 'Carbs', 'Fat'],
            vmin=-1, vmax=1, center=0)
ax4.set_title('Macronutrient Correlation Heatmap', fontsize=14)

plt.tight_layout()
plt.savefig('outputs/macronutrient_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved: outputs/macronutrient_analysis.png")

# Scatter plot for top protein recipes
plt.figure(figsize=(14, 8))
top_50_protein = df.nlargest(50, 'Protein(g)')
scatter = sns.scatterplot(data=top_50_protein, x='Cuisine_type', y='Protein(g)', 
                          hue='Diet_type', size='Protein(g)', sizes=(50, 500),
                          palette='Set1', alpha=0.7)
plt.title('Top 50 Protein-Rich Recipes by Cuisine and Diet Type', fontsize=16, fontweight='bold')
plt.xlabel('Cuisine Type')
plt.ylabel('Protein (g)')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/top_protein_scatter.png', dpi=300, bbox_inches='tight')
print("Visualization saved: outputs/top_protein_scatter.png")

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analysis complete!")
print("All outputs saved in the 'outputs' folder")