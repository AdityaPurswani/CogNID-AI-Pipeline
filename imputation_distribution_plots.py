import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os # Added import
import regex as re

# --- Configuration ---
# File paths (adjust as necessary)
file_path_before = 'cognid_processed.xlsx'  # Your data before imputation
file_path_after = 'cognid_knn_imputed_with_variation.xlsx' # Your data after imputation

# Column containing the class labels for stratification
class_column = 'Completed Diagnosis'

# List of quantitative columns that were imputed and you want to visualize
# (Make sure these column names exactly match those in your Excel files)
imputed_columns_to_plot = [
    'Total Tau pg/ml (146-595)',
    'Phospho Tau pg/ml (24-68)',
    'A Beta 142 pg/ml (627-1322)',
    'Total', # Assuming this is the total cognitive score
    'Attention',
    'mem', # Make sure this column name is correct
    'fluency', # Make sure this column name is correct
    'language', # Make sure this column name is correct
    'visuospatial', # Make sure this column name is correct
    # Add other imputed cognitive/biomarker scores if needed
]

# --- Load Data ---
try:
    df_before = pd.read_excel(file_path_before)
    print(f"Successfully loaded 'before' data: {file_path_before}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path_before}")
    exit()
except Exception as e:
    print(f"Error loading {file_path_before}: {e}")
    exit()

try:
    df_after = pd.read_excel(file_path_after)
    print(f"Successfully loaded 'after' data: {file_path_after}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path_after}")
    exit()
except Exception as e:
    print(f"Error loading {file_path_after}: {e}")
    exit()

# --- Data Preparation and Plotting ---

# Ensure the class column exists
if class_column not in df_before.columns or class_column not in df_after.columns:
    print(f"Error: Class column '{class_column}' not found in one or both files.")
    exit()

# Create output directory for plots if it doesn't exist
output_plot_dir = 'imputation_comparison_plots'
os.makedirs(output_plot_dir, exist_ok=True)
print(f"Plots will be saved in: {output_plot_dir}")

# Process and plot each specified column
for column_name in imputed_columns_to_plot:
    print(f"\nProcessing column: {column_name}")

    # Check if the column exists in both dataframes
    if column_name not in df_before.columns or column_name not in df_after.columns:
        print(f"  Warning: Column '{column_name}' not found in one or both files. Skipping.")
        continue

    # Convert column to numeric, coercing errors
    df_before_col = df_before[[class_column, column_name]].copy()
    df_after_col = df_after[[class_column, column_name]].copy()
    df_before_col[column_name] = pd.to_numeric(df_before_col[column_name], errors='coerce')
    df_after_col[column_name] = pd.to_numeric(df_after_col[column_name], errors='coerce')

    # Add a 'Dataset' column
    df_before_col['Dataset'] = 'Before Imputation'
    df_after_col['Dataset'] = 'After Imputation'

    # Combine the data
    combined_df = pd.concat([df_before_col, df_after_col], ignore_index=True)

    # *** NEW STEP: Filter out rows where class is 'No diagnosis' ***
    original_rows = len(combined_df)
    combined_df_filtered = combined_df[combined_df[class_column] != 'No diagnosis'].copy()
    rows_removed = original_rows - len(combined_df_filtered)
    if rows_removed > 0:
         print(f"  Removed {rows_removed} rows where '{class_column}' was 'No diagnosis'.")

    # Remove rows where the *value* column is NaN or class label is missing (after filtering 'No diagnosis')
    plot_df = combined_df_filtered.dropna(subset=[column_name, class_column])

    if plot_df.empty:
        print(f"  Warning: No valid data to plot for column '{column_name}' after filtering. Skipping.")
        continue

    # --- Create Plot ---
    plt.figure(figsize=(14, 8)) # Adjust figure size as needed

    # Get sorted unique class labels for consistent x-axis order
    sorted_classes = sorted(plot_df[class_column].unique())

    sns.boxplot(
        data=plot_df,
        x=class_column,
        y=column_name,
        hue='Dataset', # Group by 'Before' vs 'After'
        order=sorted_classes, # Ensure consistent order on x-axis
        palette='viridis', # Choose a color palette
        showfliers=True,
        fliersize=3
    )

    # Improve plot aesthetics
    plt.title(f'Distribution of {column_name}\nBefore vs. After Imputation (Classed Based Imputation)")', fontsize=14)
    plt.xlabel('Diagnosis Class', fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Status', loc='upper right') # Move legend outside plot
    plt.tight_layout() # Adjust layout to make space for legend
    plt.show()

    # Close the plot figure to free memory
    plt.close()

print("\nFinished generating plots.")