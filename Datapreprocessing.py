import pandas as pd
import openpyxl
import re
from sklearn.impute import KNNImputer

def get_original_column_names(file_path):
    """
    Get the original column names from the Excel file without auto-renaming.
    """
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    return [cell.value for cell in sheet[1]]

def remove_duplicate_columns(df, original_column_names):
    """
    Remove duplicate columns based on column names.
    """
    renamed_columns = df.columns.tolist()
    
    # Identify duplicates in original column names
    duplicate_columns = []
    seen_columns = set()
    
    for orig_name in original_column_names:
        if orig_name in seen_columns:
            duplicate_columns.append(orig_name)
        else:
            seen_columns.add(orig_name)
    
    # Get only non-duplicate columns from pandas DataFrame
    columns_to_keep = []
    seen_columns = set()
    
    for i, col_name in enumerate(renamed_columns):
        # Extract original name from this column
        orig_name = original_column_names[i]
        
        # If we haven't seen this column name before, keep it
        if orig_name not in seen_columns:
            columns_to_keep.append(col_name)
            seen_columns.add(orig_name)
    
    # Create DataFrame with only first instance of each column name
    df_no_dupes = df[columns_to_keep]
    
    return df_no_dupes, duplicate_columns

def remove_empty_columns(df):
    """
    Remove columns where all cells are empty.
    """
    empty_columns = []
    
    for col in df.columns:
        if df[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).all():
            empty_columns.append(col)
    
    # Remove the empty columns
    df_no_empty = df.drop(columns=empty_columns)
    
    return df_no_empty, empty_columns

def remove_sparse_columns(df, min_values=25):
    """
    Remove columns with fewer than specified number of non-empty values.
    """
    sparse_columns = []
    
    for col in df.columns:
        # Count non-empty values in the column
        non_empty_count = df[col].apply(lambda x: not (pd.isna(x) or (isinstance(x, str) and x.strip() == ''))).sum()
        
        # If fewer than min_values, mark for removal
        if non_empty_count < min_values:
            sparse_columns.append(col)
    
    # Remove the sparse columns
    df_no_sparse = df.drop(columns=sparse_columns)
    
    return df_no_sparse, sparse_columns

def fill_yes_no_columns(df, columns_to_fill):
    """
    Fill specified columns with 'No' for empty cells.
    """
    df_filled = df.copy()
    
    for column in columns_to_fill:
        if column in df_filled.columns:
            # Fill NaN, None, empty string, or whitespace-only string with 'No'
            df_filled[column] = df_filled[column].apply(
                lambda x: 'No' if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x
            )
            print(f"Processed column: {column}")
        else:
            print(f"Warning: Column '{column}' not found in the file")
    
    return df_filled

def remove_rows_with_empty_diagnosis(df, diagnosis_column="Completed Diagnosis"):
    """
    Remove rows where the diagnosis column is empty and return their Study No.
    
    Parameters:
    df (DataFrame): pandas DataFrame
    diagnosis_column (str): Name of the diagnosis column
    
    Returns:
    tuple: (DataFrame with rows removed, DataFrame of removed rows with Study No.)
    """
    # Check if diagnosis column exists
    if diagnosis_column not in df.columns:
        print(f"Warning: '{diagnosis_column}' column not found in the file")
        return df, pd.DataFrame()
    
    # Identify rows with empty diagnosis
    empty_diagnosis_mask = df[diagnosis_column].apply(
        lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')
    )
    
    # Get Study No. of rows with empty diagnosis
    if "Study No." in df.columns:
        removed_rows = df[empty_diagnosis_mask][["Study No."]]
    else:
        print("Warning: 'Study No.' column not found")
        removed_rows = pd.DataFrame()
    
    # Remove rows with empty diagnosis
    df_filtered = df[~empty_diagnosis_mask]
    
    return df_filtered, removed_rows

def process_excel_file(file_path, output_path=None):
    """
    Process Excel file with all requested modifications.
    """
    # Get original column names
    original_column_names = get_original_column_names(file_path)
    
    # Read the Excel file with all data types preserved
    df = pd.read_excel(file_path, dtype=object)
    
    # Step 1: Remove duplicate columns
    df_no_dupes, duplicate_columns = remove_duplicate_columns(df, original_column_names)
    
    # Step 2: Remove empty columns
    df_no_empty, empty_columns = remove_empty_columns(df_no_dupes)
    
    # Step 3: Remove rows with empty diagnosis and get their Study No.
    df_filtered, removed_rows = remove_rows_with_empty_diagnosis(df_no_empty)
    
    # Step 4: Fill Yes/No columns
    yes_no_columns = [
        'history of head trauma y/n, domestic? ',
        'T2DM',
        'hypertension',
        'history of stroke/TIA',
        'radiologicalevidence of stroke',
        'depression/anxiety (either has score or from letters or on meds',
        'other medical co-morbididties (IHD, asthma, etc)',
        'cardiovascular disease'
    ]
    
    df_processed = fill_yes_no_columns(df_filtered, yes_no_columns)
    # Step 5: Remove columns with fewer than 25 values
    df_no_sparse, sparse_columns = remove_sparse_columns(df_processed, min_values=25)
    
    # Save to a new Excel file
    if output_path is None:
        import os
        file_name = os.path.basename(file_path)
        output_path = os.path.join(os.path.dirname(file_path), 'processed_' + file_name)
    
    df_no_sparse.to_excel(output_path, index=False)
    
    # Print summary
    print(f"\nOriginal column count: {len(original_column_names)}")
    print(f"Duplicate columns removed: {len(duplicate_columns)}")
    print(f"Empty columns removed: {len(empty_columns)}")
    print(f"Sparse columns removed (< 25 values): {len(sparse_columns)}")
    print(sparse_columns)
    print(f"Rows with empty diagnosis removed: {len(removed_rows)}")
    print(f"New column count: {len(df_no_sparse.columns)}")
    print(f"New row count: {len(df_no_sparse)}")
    print(f"Saved processed file to: {output_path}")
    
    # Print Study No. of removed rows
    if not removed_rows.empty:
        print("\nStudy No. of rows with empty diagnosis:")
        for study_no in removed_rows["Study No."]:
            print(f"- {study_no}")
    
    return df_processed, removed_rows

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = 'cognid.xlsx'
    output_path = 'cognid_processed.xlsx'
    
    # Process the Excel file
    df_processed, removed_rows = process_excel_file(file_path, output_path)