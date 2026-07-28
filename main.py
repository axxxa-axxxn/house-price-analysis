import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------
# Create images folder
# --------------------------------------------------------

os.makedirs("images", exist_ok=True)

# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------

def load_dataset():
    """
    Load California Housing Dataset
    """

    try:

        df = pd.read_csv("housing.csv")

        print("=" * 60)
        print("CALIFORNIA HOUSING DATASET LOADED")
        print("=" * 60)

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        return df

    except FileNotFoundError:

        print("housing.csv not found!")

        return None


# --------------------------------------------------------
# Dataset Inspection
# --------------------------------------------------------

def inspect_data(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nLast 5 Rows")
    print(df.tail())

    print("\nShape")
    print(df.shape)

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nDataset Information")
    df.info()

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nDescriptive Statistics")
    print(df.describe(include="all"))


# --------------------------------------------------------
# Missing Value Analysis
# --------------------------------------------------------

def missing_value_analysis(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    plt.figure(figsize=(8,5))

    missing.plot(kind="bar")

    plt.title("Missing Values")

    plt.xlabel("Columns")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("images/missing_values.png")

    plt.show()


# --------------------------------------------------------
# Remove Duplicates
# --------------------------------------------------------

def remove_duplicates(df):

    print("\n" + "=" * 60)
    print("REMOVE DUPLICATES")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")

    df = df.drop_duplicates()

    print(f"Rows After Removing Duplicates : {df.shape[0]}")

    return df


# --------------------------------------------------------
# Data Cleaning
# --------------------------------------------------------

def data_cleaning(df):

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print("\nMissing Values Before Cleaning")

    print(df.isnull().sum())

    if "total_bedrooms" in df.columns:

        median_value = df["total_bedrooms"].median()

        df["total_bedrooms"] = df["total_bedrooms"].fillna(median_value)

    print("\nMissing Values After Cleaning")

    print(df.isnull().sum())

    return df

# --------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------

def feature_engineering(df):
    """
    Create new useful features.
    """

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Average Rooms per Household
    df["rooms_per_household"] = (
        df["total_rooms"] / df["households"]
    )

    # Average Bedrooms per Room
    df["bedrooms_per_room"] = (
        df["total_bedrooms"] / df["total_rooms"]
    )

    # Population per Household
    df["population_per_household"] = (
        df["population"] / df["households"]
    )

    print("\nNew Columns Added")

    print(df[
        [
            "rooms_per_household",
            "bedrooms_per_room",
            "population_per_household"
        ]
    ].head())

    return df


# --------------------------------------------------------
# Filtering & Sorting
# --------------------------------------------------------

def filter_and_sort_data(df):

    print("\n" + "=" * 60)
    print("FILTERING & SORTING")
    print("=" * 60)

    print("\nHouses With Median Income > 8")

    print(
        df[df["median_income"] > 8][
            ["median_income", "median_house_value"]
        ].head()
    )

    print("\nHouses Near Ocean")

    print(
        df[df["ocean_proximity"] == "NEAR OCEAN"][
            ["median_house_value", "median_income"]
        ].head()
    )

    print("\nTop 5 Most Expensive Houses")

    print(
        df.sort_values(
            by="median_house_value",
            ascending=False
        )[
            [
                "median_house_value",
                "median_income",
                "ocean_proximity"
            ]
        ].head()
    )

    print("\nYoungest Houses")

    print(
        df.sort_values(
            by="housing_median_age"
        )[
            [
                "housing_median_age",
                "median_house_value"
            ]
        ].head()
    )


# --------------------------------------------------------
# GroupBy Analysis
# --------------------------------------------------------

def groupby_analysis(df):

    print("\n" + "=" * 60)
    print("GROUPBY ANALYSIS")
    print("=" * 60)

    print("\nAverage House Price By Ocean Proximity")

    print(
        df.groupby("ocean_proximity")[
            "median_house_value"
        ].mean()
    )

    print("\nAverage Median Income By Ocean Proximity")

    print(
        df.groupby("ocean_proximity")[
            "median_income"
        ].mean()
    )

    print("\nAverage Population By Ocean Proximity")

    print(
        df.groupby("ocean_proximity")[
            "population"
        ].mean()
    )

    print("\nAverage Rooms Per Household")

    print(
        df.groupby("ocean_proximity")[
            "rooms_per_household"
        ].mean()
    )


# --------------------------------------------------------
# Descriptive Statistics
# --------------------------------------------------------

def descriptive_statistics(df):

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    columns = [
        "median_income",
        "median_house_value",
        "housing_median_age",
        "population"
    ]

    for column in columns:

        print("\n" + "-" * 50)

        print(column.upper())

        print("-" * 50)

        print(f"Mean   : {df[column].mean():.2f}")
        print(f"Median : {df[column].median():.2f}")
        print(f"Mode   : {df[column].mode()[0]:.2f}")
        print(f"Minimum: {df[column].min():.2f}")
        print(f"Maximum: {df[column].max():.2f}")
        print(f"Std Dev: {df[column].std():.2f}")

        print("\nQuartiles")

        print(df[column].quantile([0.25, 0.50, 0.75]))


# --------------------------------------------------------
# Correlation Matrix
# --------------------------------------------------------

def correlation_analysis(df):

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    numerical_df = df.select_dtypes(include="number")

    correlation = numerical_df.corr()

    print(correlation)

    return correlation


# --------------------------------------------------------
# Outlier Detection (IQR)
# --------------------------------------------------------

def detect_outliers(df):

    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)

    columns = [
        "median_income",
        "median_house_value",
        "population"
    ]

    for column in columns:

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)

        upper = Q3 + (1.5 * IQR)

        outliers = df[
            (df[column] < lower) |
            (df[column] > upper)
        ]

        print("\n" + "-" * 50)

        print(f"Column : {column}")

        print(f"Lower Bound : {lower:.2f}")

        print(f"Upper Bound : {upper:.2f}")

        print(f"Total Outliers : {len(outliers)}")

        plt.figure(figsize=(6,5))

        plt.boxplot(df[column])

        plt.title(f"{column} Box Plot")

        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            f"images/{column}_boxplot.png"
        )

        plt.show()

# --------------------------------------------------------
# Data Visualization
# --------------------------------------------------------

def visualization(df):

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # ----------------------------------------------------
    # Price Distribution
    # ----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(
        df["median_house_value"],
        bins=30,
        edgecolor="black"
    )

    plt.title("House Price Distribution")
    plt.xlabel("Median House Value")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("images/price_distribution.png")

    plt.show()


    # ----------------------------------------------------
    # House Age Distribution
    # ----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(
        df["housing_median_age"],
        bins=20,
        edgecolor="black"
    )

    plt.title("House Age Distribution")
    plt.xlabel("House Age")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("images/house_age_distribution.png")

    plt.show()


    # ----------------------------------------------------
    # Population Distribution
    # ----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(
        df["population"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Population Distribution")
    plt.xlabel("Population")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("images/population_distribution.png")

    plt.show()


    # ----------------------------------------------------
    # Income vs House Price
    # ----------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["median_income"],
        df["median_house_value"],
        alpha=0.5
    )

    plt.title("Median Income vs House Price")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")

    plt.tight_layout()

    plt.savefig("images/median_income_vs_price.png")

    plt.show()


    # ----------------------------------------------------
    # Total Rooms vs House Price
    # ----------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["total_rooms"],
        df["median_house_value"],
        alpha=0.5
    )

    plt.title("Rooms vs House Price")
    plt.xlabel("Total Rooms")
    plt.ylabel("Median House Value")

    plt.tight_layout()

    plt.savefig("images/rooms_vs_price.png")

    plt.show()


    # ----------------------------------------------------
    # Ocean Proximity vs Price
    # ----------------------------------------------------

    ocean_price = (
        df.groupby("ocean_proximity")["median_house_value"]
        .mean()
    )

    plt.figure(figsize=(8,5))

    plt.bar(
        ocean_price.index,
        ocean_price.values
    )

    plt.xticks(rotation=20)

    plt.title("Average House Price by Ocean Proximity")
    plt.xlabel("Ocean Proximity")
    plt.ylabel("Average Price")

    plt.tight_layout()

    plt.savefig("images/ocean_proximity_prices.png")

    plt.show()


    # ----------------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------------

    plt.figure(figsize=(10,8))

    correlation = (
        df.select_dtypes(include="number")
        .corr()
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("images/correlation_heatmap.png")

    plt.show()


    # ----------------------------------------------------
    # Pair Plot
    # ----------------------------------------------------

    columns = [
        "median_income",
        "housing_median_age",
        "total_rooms",
        "median_house_value"
    ]

    pairplot = sns.pairplot(df[columns])

    pairplot.savefig("images/pairplot.png")

    plt.show()


# --------------------------------------------------------
# Summary Report
# --------------------------------------------------------

def summary_report(df):

    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    with open("report.txt", "w") as report:

        report.write("CALIFORNIA HOUSING EDA REPORT\n")
        report.write("=" * 50 + "\n\n")

        report.write(
            f"Total Rows : {df.shape[0]}\n"
        )

        report.write(
            f"Total Columns : {df.shape[1]}\n\n"
        )

        report.write("Missing Values\n")
        report.write("-" * 30 + "\n")
        report.write(
            df.isnull().sum().to_string()
        )

        report.write("\n\n")

        report.write("Duplicate Rows\n")
        report.write("-" * 30 + "\n")
        report.write(
            str(df.duplicated().sum())
        )

        report.write("\n\n")

        report.write(
            "Average House Price\n"
        )

        report.write("-" * 30 + "\n")

        report.write(
            str(df["median_house_value"].mean())
        )

        report.write("\n\n")

        report.write(
            "Average Median Income\n"
        )

        report.write("-" * 30 + "\n")

        report.write(
            str(df["median_income"].mean())
        )

        report.write("\n\n")

        report.write(
            "Average Price by Ocean Proximity\n"
        )

        report.write("-" * 30 + "\n")

        report.write(
            df.groupby("ocean_proximity")[
                "median_house_value"
            ].mean().to_string()
        )

        report.write("\n\n")

        report.write("Key Findings\n")
        report.write("-" * 30 + "\n")

        report.write(
            "- Missing values were found mainly in total_bedrooms.\n"
        )

        report.write(
            "- Missing values were replaced using the median.\n"
        )

        report.write(
            "- Several numerical columns contain outliers.\n"
        )

        report.write(
            "- Median income has a strong positive relationship with house prices.\n"
        )

        report.write(
            "- Houses near the ocean generally have higher prices.\n"
        )

        report.write(
            "- Additional features were created for better analysis.\n"
        )

    print("report.txt generated successfully.")


# --------------------------------------------------------
# Main Function
# --------------------------------------------------------

def main():

    df = load_dataset()

    if df is None:
        return

    inspect_data(df)

    missing_value_analysis(df)

    df = remove_duplicates(df)

    df = data_cleaning(df)

    df = feature_engineering(df)

    filter_and_sort_data(df)

    groupby_analysis(df)

    descriptive_statistics(df)

    correlation_analysis(df)

    detect_outliers(df)

    visualization(df)

    summary_report(df)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()