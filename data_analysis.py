import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import sys


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def validate_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        log(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)


def main():
    log("Starting nutritional analysis...")

    dataset_path = "All_Diets.csv"
    output_dir = "outputs"

    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check dataset
    if not os.path.exists(dataset_path):
        log("ERROR: All_Diets.csv not found!")
        log(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    # Load dataset
    log("Loading dataset...")
    df = pd.read_csv(dataset_path)
    log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    log(f"Columns detected: {df.columns.tolist()}")

    # Validate required columns
    required_columns = [
        "Diet_type",
        "Recipe_name",
        "Cuisine_type",
        "Protein(g)",
        "Carbs(g)",
        "Fat(g)"
    ]
    validate_columns(df, required_columns)

    numeric_cols = ["Protein(g)", "Carbs(g)", "Fat(g)"]

    # Handle missing values
    log("Handling missing values...")
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(df[col].mean())
            log(f"Filled {missing_count} missing values in {col}")

    # 1️⃣ Average macronutrients by diet type
    log("Calculating average macronutrients by diet type...")
    avg_macros = df.groupby("Diet_type")[numeric_cols].mean().round(2)
    print("\nAverage Macronutrients:\n", avg_macros)
    avg_macros.to_csv(f"{output_dir}/avg_macros_by_diet.csv")

    # 2️⃣ Top 5 protein recipes per diet
    log("Finding top 5 protein-rich recipes per diet...")
    top_protein = (
        df.sort_values("Protein(g)", ascending=False)
        .groupby("Diet_type")
        .head(5)
    )
    top_protein.to_csv(f"{output_dir}/top_protein_recipes.csv")

    # 3️⃣ Highest protein diet
    protein_means = df.groupby("Diet_type")["Protein(g)"].mean()
    highest_protein_diet = protein_means.idxmax()
    highest_protein_value = protein_means.max()
    log(
        f"Diet with highest average protein: "
        f"{highest_protein_diet} ({highest_protein_value:.2f}g)"
    )

    # 4️⃣ Most common cuisines
    log("Identifying most common cuisine per diet type...")
    common_cuisines = (
        df.groupby("Diet_type")["Cuisine_type"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    )
    print("\nMost Common Cuisines:\n", common_cuisines)

    # 5️⃣ Create ratio metrics safely
    log("Creating ratio metrics...")
    df["Protein_to_Carbs_ratio"] = np.where(
        df["Carbs(g)"] != 0,
        df["Protein(g)"] / df["Carbs(g)"],
        np.nan,
    )
    df["Carbs_to_Fat_ratio"] = np.where(
        df["Fat(g)"] != 0,
        df["Carbs(g)"] / df["Fat(g)"],
        np.nan,
    )
    df["Protein_to_Fat_ratio"] = np.where(
        df["Fat(g)"] != 0,
        df["Protein(g)"] / df["Fat(g)"],
        np.nan,
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 6️⃣ Visualizations
    log("Creating visualizations...")
    sns.set_style("whitegrid")

    # --- Bar Charts + Heatmap ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Nutritional Analysis by Diet Type", fontsize=16)

    # Protein
    avg_macros["Protein(g)"].sort_values().plot.barh(ax=axes[0, 0])
    axes[0, 0].set_title("Average Protein by Diet Type")

    # Carbs
    avg_macros["Carbs(g)"].sort_values().plot.barh(ax=axes[0, 1])
    axes[0, 1].set_title("Average Carbs by Diet Type")

    # Fat
    avg_macros["Fat(g)"].sort_values().plot.barh(ax=axes[1, 0])
    axes[1, 0].set_title("Average Fat by Diet Type")

    # Correlation heatmap
    correlation = df[numeric_cols].corr()
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        ax=axes[1, 1],
        vmin=-1,
        vmax=1,
        center=0,
    )
    axes[1, 1].set_title("Macronutrient Correlation")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/macronutrient_analysis.png", dpi=300)
    plt.close()

    # --- Scatter Plot ---
    plt.figure(figsize=(14, 8))
    top_50 = df.nlargest(50, "Protein(g)")
    sns.scatterplot(
        data=top_50,
        x="Cuisine_type",
        y="Protein(g)",
        hue="Diet_type",
        size="Protein(g)",
        sizes=(50, 400),
        alpha=0.7,
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_protein_scatter.png", dpi=300)
    plt.close()

    log("Analysis complete!")
    log("All outputs saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()
