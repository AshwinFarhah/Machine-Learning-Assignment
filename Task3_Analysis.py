# ============================================================
# TASK 3 - CRITICAL ANALYSIS OF DATA QUALITY
# CDC DIABETES HEALTH INDICATORS DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

# CHANGE THIS if your dataset has a different filename
DATA_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

# Folder where all results will be saved
OUTPUT_FOLDER = "Task3_Results"

# Target variable
TARGET = "Diabetes_binary"


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("TASK 3 - DATA QUALITY ANALYSIS")
print("CDC DIABETES HEALTH INDICATORS DATASET")
print("=" * 70)


# ============================================================
# 3. LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print("\nERROR: Dataset file was not found.")
    print(f"Expected file: {DATA_FILE}")
    print("Place the CSV file in the same folder as this Python script.")
    input("\nPress Enter to exit...")
    raise SystemExit

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())


# Save data types
dtype_table = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

dtype_table.to_csv(
    os.path.join(OUTPUT_FOLDER, "01_data_types.csv"),
    index=False
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_table = pd.DataFrame({
    "Feature": df.columns,
    "Missing Count": df.isnull().sum().values,
    "Missing Percentage": (
        df.isnull().mean().values * 100
    )
})

missing_table["Missing Percentage"] = missing_table[
    "Missing Percentage"
].round(4)

print("\nMissing-value summary:")
print(missing_table.to_string(index=False))

missing_only = missing_table[
    missing_table["Missing Count"] > 0
]

if missing_only.empty:
    print("\nRESULT: No missing values were detected.")
else:
    print("\nRESULT: Missing values were detected in:")
    print(missing_only.to_string(index=False))

missing_table.to_csv(
    os.path.join(OUTPUT_FOLDER, "02_missing_values.csv"),
    index=False
)


# ============================================================
# 6. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE RECORD ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()
duplicate_percentage = (duplicate_count / len(df)) * 100

print(f"\nTotal records: {len(df):,}")
print(f"Duplicate records: {duplicate_count:,}")
print(f"Duplicate percentage: {duplicate_percentage:.4f}%")
print(f"Unique records: {len(df) - duplicate_count:,}")

duplicate_summary = pd.DataFrame({
    "Measure": [
        "Total Records",
        "Duplicate Records",
        "Duplicate Percentage",
        "Unique Records"
    ],
    "Value": [
        len(df),
        duplicate_count,
        duplicate_percentage,
        len(df) - duplicate_count
    ]
})

duplicate_summary.to_csv(
    os.path.join(OUTPUT_FOLDER, "03_duplicates.csv"),
    index=False
)


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

descriptive_stats = df[numeric_columns].describe().T

descriptive_stats["median"] = df[numeric_columns].median()

descriptive_stats = descriptive_stats[
    [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ]
]

print("\nDescriptive statistics:")
print(descriptive_stats.round(4).to_string())

descriptive_stats.to_csv(
    os.path.join(OUTPUT_FOLDER, "04_descriptive_statistics.csv")
)


# ============================================================
# 8. SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("5. SKEWNESS ANALYSIS")
print("=" * 70)

skewness = df[numeric_columns].skew()

skewness_table = pd.DataFrame({
    "Feature": skewness.index,
    "Skewness": skewness.values
})

skewness_table["Skewness"] = skewness_table[
    "Skewness"
].round(4)

skewness_table["Distribution"] = skewness_table[
    "Skewness"
].apply(
    lambda x:
        "Approximately symmetric"
        if abs(x) < 0.5
        else
        "Moderately skewed"
        if abs(x) < 1
        else
        "Highly skewed"
)

print("\nSkewness:")
print(skewness_table.to_string(index=False))

skewness_table.to_csv(
    os.path.join(OUTPUT_FOLDER, "05_skewness.csv"),
    index=False
)


# ============================================================
# 9. OUTLIER ANALYSIS USING IQR
# ============================================================

print("\n" + "=" * 70)
print("6. OUTLIER ANALYSIS - IQR METHOD")
print("=" * 70)

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()

    outlier_percentage = (
        outlier_count / len(df)
    ) * 100

    outlier_results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count,
        "Outlier Percentage": outlier_percentage
    })

outlier_table = pd.DataFrame(outlier_results)

for column in [
    "Q1",
    "Q3",
    "IQR",
    "Lower Bound",
    "Upper Bound",
    "Outlier Percentage"
]:
    outlier_table[column] = outlier_table[column].round(4)

print("\nIQR outlier analysis:")
print(outlier_table.to_string(index=False))

outlier_table.to_csv(
    os.path.join(OUTPUT_FOLDER, "06_outliers.csv"),
    index=False
)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("7. CLASS IMBALANCE ANALYSIS")
print("=" * 70)

if TARGET in df.columns:

    class_counts = df[TARGET].value_counts().sort_index()
    class_percentages = (
        df[TARGET].value_counts(
            normalize=True
        ).sort_index() * 100
    )

    class_table = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values,
        "Percentage": class_percentages.values
    })

    class_table["Percentage"] = class_table[
        "Percentage"
    ].round(4)

    print("\nTarget distribution:")
    print(class_table.to_string(index=False))

    majority_count = class_counts.max()
    minority_count = class_counts.min()

    imbalance_ratio = (
        majority_count / minority_count
    )

    print(f"\nMajority class count: {majority_count:,}")
    print(f"Minority class count: {minority_count:,}")
    print(f"Imbalance ratio: {imbalance_ratio:.4f}")

    class_table.to_csv(
        os.path.join(OUTPUT_FOLDER, "07_class_distribution.csv"),
        index=False
    )

else:

    print(
        f"\nWARNING: Target variable '{TARGET}' "
        "was not found."
    )


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. CORRELATION ANALYSIS")
print("=" * 70)

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation.round(3).to_string())

correlation.to_csv(
    os.path.join(OUTPUT_FOLDER, "08_correlation_matrix.csv")
)


# ------------------------------------------------------------
# Strongest correlations
# ------------------------------------------------------------

correlation_pairs = []

for i in range(len(correlation.columns)):

    for j in range(i + 1, len(correlation.columns)):

        feature_1 = correlation.columns[i]
        feature_2 = correlation.columns[j]

        value = correlation.iloc[i, j]

        correlation_pairs.append({
            "Feature 1": feature_1,
            "Feature 2": feature_2,
            "Correlation": value,
            "Absolute Correlation": abs(value)
        })

correlation_pairs = pd.DataFrame(
    correlation_pairs
)

correlation_pairs = correlation_pairs.sort_values(
    "Absolute Correlation",
    ascending=False
)

correlation_pairs["Correlation"] = correlation_pairs[
    "Correlation"
].round(4)

correlation_pairs["Absolute Correlation"] = (
    correlation_pairs["Absolute Correlation"]
    .round(4)
)

print("\nTop 15 strongest correlations:")
print(
    correlation_pairs.head(15).to_string(
        index=False
    )
)

correlation_pairs.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "09_correlation_pairs.csv"
    ),
    index=False
)


# ============================================================
# 12. DATA CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("9. DATA CONSISTENCY ANALYSIS")
print("=" * 70)

consistency_results = []

for column in df.columns:

    unique_count = df[column].nunique(
        dropna=False
    )

    min_value = df[column].min()
    max_value = df[column].max()

    consistency_results.append({
        "Feature": column,
        "Unique Values": unique_count,
        "Minimum": min_value,
        "Maximum": max_value
    })

consistency_table = pd.DataFrame(
    consistency_results
)

print("\nFeature range and uniqueness:")
print(consistency_table.to_string(index=False))

consistency_table.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "10_consistency_summary.csv"
    ),
    index=False
)


# ============================================================
# 13. VALUE COUNTS FOR ALL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("10. UNIQUE VALUE CHECK")
print("=" * 70)

value_count_file = os.path.join(
    OUTPUT_FOLDER,
    "11_value_counts.txt"
)

with open(
    value_count_file,
    "w",
    encoding="utf-8"
) as file:

    for column in df.columns:

        file.write("\n")
        file.write("=" * 60)
        file.write("\n")
        file.write(f"FEATURE: {column}\n")
        file.write("=" * 60)
        file.write("\n")

        counts = df[column].value_counts(
            dropna=False
        ).sort_index()

        file.write(
            counts.to_string()
        )

        file.write("\n\n")

print(
    f"\nFull value counts saved to:\n"
    f"{value_count_file}"
)


# ============================================================
# 14. GENERATE HISTOGRAMS
# ============================================================

print("\n" + "=" * 70)
print("11. GENERATING FEATURE DISTRIBUTIONS")
print("=" * 70)

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column].dropna(),
        bins=30
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = (
        f"distribution_{column}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300
    )

    plt.close()

print(
    f"\nSaved {len(numeric_columns)} "
    "distribution graphs."
)


# ============================================================
# 15. BOXPLOTS
# ============================================================

print("\n" + "=" * 70)
print("12. GENERATING BOXPLOTS")
print("=" * 70)

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df[column].dropna(),
        vert=False
    )

    plt.title(
        f"Boxplot of {column}"
    )

    plt.xlabel(column)

    plt.tight_layout()

    filename = (
        f"boxplot_{column}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300
    )

    plt.close()

print(
    f"\nSaved {len(numeric_columns)} "
    "boxplots."
)


# ============================================================
# 16. CLASS DISTRIBUTION GRAPH
# ============================================================

if TARGET in df.columns:

    plt.figure(figsize=(7, 5))

    class_counts.plot(
        kind="bar"
    )

    plt.title(
        "Distribution of Diabetes_binary"
    )

    plt.xlabel(
        "Diabetes_binary"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "class_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nSaved class distribution graph."
    )


# ============================================================
# 17. CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 70)
print("13. GENERATING CORRELATION HEATMAP")
print("=" * 70)

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Matrix of Dataset Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()

print(
    "\nSaved correlation heatmap."
)


# ============================================================
# 18. AUTOMATIC SUMMARY REPORT
# ============================================================

print("\n" + "=" * 70)
print("14. CREATING SUMMARY REPORT")
print("=" * 70)

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "Task3_Summary_Report.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 3 - CRITICAL ANALYSIS OF DATA QUALITY\n"
    )

    file.write(
        "CDC DIABETES HEALTH INDICATORS DATASET\n"
    )

    file.write("=" * 70 + "\n\n")

    # Dataset
    file.write("DATASET OVERVIEW\n")
    file.write("-" * 70 + "\n")

    file.write(
        f"Number of observations: {len(df):,}\n"
    )

    file.write(
        f"Number of attributes: {len(df.columns):,}\n"
    )

    file.write(
        f"Duplicate records: {duplicate_count:,}\n"
    )

    file.write(
        f"Duplicate percentage: "
        f"{duplicate_percentage:.4f}%\n\n"
    )

    # Missing
    file.write("MISSING VALUES\n")
    file.write("-" * 70 + "\n")

    total_missing = df.isnull().sum().sum()

    file.write(
        f"Total missing values: "
        f"{total_missing:,}\n"
    )

    if total_missing == 0:
        file.write(
            "No missing values detected.\n\n"
        )
    else:
        file.write(
            missing_only.to_string(index=False)
        )
        file.write("\n\n")

    # Class balance
    if TARGET in df.columns:

        file.write("CLASS DISTRIBUTION\n")
        file.write("-" * 70 + "\n")

        file.write(
            class_table.to_string(
                index=False
            )
        )

        file.write(
            f"\n\nImbalance ratio: "
            f"{imbalance_ratio:.4f}\n\n"
        )

    # Skewness
    file.write("SKEWNESS\n")
    file.write("-" * 70 + "\n")

    file.write(
        skewness_table.to_string(
            index=False
        )
    )

    file.write("\n\n")

    # Outliers
    file.write("OUTLIERS\n")
    file.write("-" * 70 + "\n")

    file.write(
        outlier_table.to_string(
            index=False
        )
    )

    file.write("\n\n")

    # Correlation
    file.write(
        "TOP CORRELATIONS\n"
    )

    file.write("-" * 70 + "\n")

    file.write(
        correlation_pairs.head(15).to_string(
            index=False
        )
    )

    file.write("\n\n")

    # Consistency
    file.write(
        "CONSISTENCY SUMMARY\n"
    )

    file.write("-" * 70 + "\n")

    file.write(
        consistency_table.to_string(
            index=False
        )
    )

    file.write("\n")


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nAll results have been saved in:"
)

print(
    f"  {os.path.abspath(OUTPUT_FOLDER)}"
)

print("\nMain files produced:")

print("1. 02_missing_values.csv")
print("2. 03_duplicates.csv")
print("3. 04_descriptive_statistics.csv")
print("4. 05_skewness.csv")
print("5. 06_outliers.csv")
print("6. 07_class_distribution.csv")
print("7. 08_correlation_matrix.csv")
print("8. 09_correlation_pairs.csv")
print("9. 10_consistency_summary.csv")
print("10. 11_value_counts.txt")
print("11. Task3_Summary_Report.txt")
print("12. Distribution graphs")
print("13. Boxplots")
print("14. Class distribution graph")
print("15. Correlation heatmap")

print("\nYou can now use these results to construct Task 3.")
print("=" * 70)

input("\nPress Enter to exit...")