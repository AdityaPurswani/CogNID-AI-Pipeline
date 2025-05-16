import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def knn_impute_by_class_with_variation(df, class_column, impute_columns, n_neighbors=5, 
                                       min_samples_required=2, variation_factor=0.1, 
                                       apply_bounds=True, random_state=42):
    """
    Apply KNN imputation separately within each class, with fallback to similar class borrowing
    and added variation for small classes to prevent identical values.
    
    Parameters:
    df (DataFrame): Input dataframe with missing values
    class_column (str): Column name containing class labels
    impute_columns (list): List of column names to impute
    n_neighbors (int): Number of neighbors for KNN imputation
    min_samples_required (int): Minimum non-missing values required for class-based imputation
    variation_factor (float): Factor to control amount of variation (0.1 = 10% variation)
    apply_bounds (bool): Whether to apply reasonable bounds to imputed values
    random_state (int): Random seed for reproducibility
    
    Returns:
    DataFrame: Dataframe with imputed values
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Create a copy of the original dataframe to hold results
    result_df = df.copy()
    
    # Get unique class values (excluding NaN)
    classes = df[class_column].dropna().unique()
    print(f"Found {len(classes)} classes for imputation: {classes}")
    
    # Track imputation statistics
    total_imputed = 0
    imputed_by_class = {}
    
    # First, verify all impute_columns exist
    missing_cols = [col for col in impute_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in the dataframe")
        # Filter to only include columns that exist
        impute_columns = [col for col in impute_columns if col in df.columns]
    
    # Create a set to track columns that need fallback
    failed_imputation_columns = {}
    
    # Calculate global statistics for bounds
    global_stats = {}
    if apply_bounds:
        for col in impute_columns:
            if col in df.columns:
                col_values = pd.to_numeric(df[col], errors='coerce')
                global_stats[col] = {
                    'mean': col_values.mean(),
                    'std': col_values.std(),
                    'min': col_values.min(),
                    'max': col_values.max(),
                    'q1': col_values.quantile(0.25) if col_values.count() > 0 else None,
                    'q3': col_values.quantile(0.75) if col_values.count() > 0 else None
                }
                # Calculate reasonable min/max bounds using IQR
                if global_stats[col]['q1'] is not None and global_stats[col]['q3'] is not None:
                    iqr = global_stats[col]['q3'] - global_stats[col]['q1']
                    global_stats[col]['lower_bound'] = max(
                        global_stats[col]['min'], 
                        global_stats[col]['q1'] - 1.5 * iqr
                    )
                    global_stats[col]['upper_bound'] = min(
                        global_stats[col]['max'], 
                        global_stats[col]['q3'] + 1.5 * iqr
                    )
                else:
                    global_stats[col]['lower_bound'] = global_stats[col]['min']
                    global_stats[col]['upper_bound'] = global_stats[col]['max']
    
    # Calculate class similarities for fallback strategy
    print("\nCalculating class similarities for fallback imputation...")
    class_similarities = calculate_class_similarities(df, class_column, impute_columns)
    
    # Print similarity matrix
    print("\nClass similarity matrix:")
    for cls1 in classes:
        similar_classes = [(cls2, sim) for cls2, sim in class_similarities[cls1].items() if cls2 != cls1]
        similar_classes = sorted(similar_classes, key=lambda x: x[1], reverse=True)
        if similar_classes:
            top_similars = similar_classes[:2]  # Show top 2 most similar classes
            print(f"  Class '{cls1}' is most similar to: " + 
                  ", ".join([f"'{cls}' (sim={sim:.3f})" for cls, sim in top_similars]))
    
    # Before imputation, analyze data availability for each class
    print("\nAnalyzing data availability per class:")
    for cls in classes:
        class_mask = df[class_column] == cls
        subset = df[class_mask]
        print(f"\nClass '{cls}' has {len(subset)} samples")
        
        # Count non-NA values per column
        available_per_column = {}
        for col in impute_columns:
            # Convert to numeric first
            numeric_values = pd.to_numeric(subset[col], errors='coerce')
            non_missing = numeric_values.count()
            available_per_column[col] = non_missing
            print(f"  - {col}: {non_missing}/{len(subset)} values available " + 
                  f"({100*non_missing/len(subset):.1f}%)")
    
    # Process each class separately
    for cls in classes:
        # Get rows belonging to this class
        class_mask = df[class_column] == cls
        subset = df[class_mask].copy()
        
        # Skip if no samples
        if len(subset) == 0:
            print(f"No samples for class '{cls}', skipping")
            continue
            
        # Convert columns to numeric for imputation
        subset_numeric = subset[impute_columns].copy()
        for col in impute_columns:
            subset_numeric[col] = pd.to_numeric(subset_numeric[col], errors='coerce')
        
        # Check for missing values before imputation
        missing_before = subset_numeric.isna().sum().sum()
        
        if missing_before == 0:
            print(f"No missing values to impute for class '{cls}'")
            continue
        
        print(f"\nImputing class '{cls}' with {len(subset)} samples")
        
        # Identify columns with enough data for imputation
        valid_columns = []
        not_valid_columns = []
        for col in impute_columns:
            non_missing = subset_numeric[col].count()
            if non_missing >= min_samples_required:  # Need at least min_samples_required values for KNN to function
                valid_columns.append(col)
            else:
                not_valid_columns.append(col)
                print(f"  - Warning: Column '{col}' has only {non_missing} values and can't be imputed within this class")
                # Track columns that need fallback
                if cls not in failed_imputation_columns:
                    failed_imputation_columns[cls] = []
                failed_imputation_columns[cls].append(col)
        
        # Check if we have enough samples for the requested n_neighbors
        effective_k = min(n_neighbors, max(1, len(subset) - 1))
        if effective_k < n_neighbors:
            print(f"  - Warning: Using {effective_k} neighbors instead of {n_neighbors} due to limited samples")
        
        if not valid_columns:
            print(f"  - No columns have enough data for imputation in class '{cls}'")
            continue
            
        # Apply imputation column by column
        imputer = KNNImputer(n_neighbors=effective_k, weights='distance')
        
        try:
            # Impute columns with data
            imputed_values = imputer.fit_transform(subset_numeric[valid_columns])
            
            # Convert back to dataframe
            imputed_df = pd.DataFrame(imputed_values, 
                                    index=subset.index, 
                                    columns=valid_columns)
            
            # Update missing values in the result dataframe
            num_imputed = 0
            for col in valid_columns:
                missing_mask = subset_numeric[col].isna()
                if missing_mask.any():
                    # Get indices of missing values
                    missing_indices = subset[missing_mask].index
                    
                    # Check if this is a small class that needs variation
                    # (defined as having fewer than 2*min_samples_required samples)
                    needs_variation = len(subset) < 3 * min_samples_required
                    
                    # Get the imputed values
                    imputed_col_values = imputed_df.loc[missing_indices, col].values
                    
                    # Check if all imputed values are identical
                    unique_values = np.unique(imputed_col_values)
                    if len(unique_values) == 1 and len(imputed_col_values) > 1:
                        # If we got identical values, definitely need variation
                        needs_variation = True
                    
                    # Add variation if needed
                    if needs_variation:
                        # Get column statistics for variation
                        col_values = subset_numeric[col].dropna()
                        if len(col_values) > 1:
                            col_std = col_values.std()
                        elif col in global_stats:
                            # Use global stats if class-specific not available
                            col_std = global_stats[col]['std']
                        else:
                            # Default small variation
                            col_std = abs(np.mean(imputed_col_values)) * 0.1 if np.mean(imputed_col_values) != 0 else 1.0
                        
                        # Ensure non-zero std
                        if pd.isna(col_std) or col_std == 0:
                            col_std = abs(np.mean(imputed_col_values)) * 0.1 if np.mean(imputed_col_values) != 0 else 1.0
                        
                        # Generate random variation (smaller for small classes to avoid extremes)
                        variation_scale = col_std * variation_factor * 0.8
                        noise = np.random.normal(0, variation_scale, size=len(imputed_col_values))
                        
                        # Add noise
                        varied_values = imputed_col_values + noise
                        
                        # Apply bounds if requested
                        if apply_bounds and col in global_stats:
                            # Handle non-negative columns
                            should_be_non_negative = global_stats[col]['min'] >= 0 or any(
                                kw in col.lower() for kw in ['tau', 'beta', 'total', 'score']
                            )
                            
                            if should_be_non_negative:
                                # Fix negative values by reflecting them
                                negative_mask = varied_values < 0
                                if np.any(negative_mask):
                                    # Use reflection with small random adjustment
                                    varied_values[negative_mask] = abs(varied_values[negative_mask]) * np.random.uniform(0.95, 1.05, size=np.sum(negative_mask))
                            
                            # Apply general bounds
                            lower_bound = global_stats[col]['lower_bound']
                            upper_bound = global_stats[col]['upper_bound']
                            varied_values = np.clip(varied_values, lower_bound, upper_bound)
                        
                        # Update with varied values
                        for i, idx in enumerate(missing_indices):
                            result_df.loc[idx, col] = varied_values[i]
                    else:
                        # No variation needed, use original imputed values
                        for idx in missing_indices:
                            result_df.loc[idx, col] = imputed_df.loc[idx, col]
                    
                    num_imputed += len(missing_indices)
            
            imputed_by_class[cls] = num_imputed
            total_imputed += num_imputed
            
            print(f"  - Successfully imputed {num_imputed} values for class '{cls}'")
            
        except Exception as e:
            print(f"  - Error during imputation for class '{cls}': {e}")
            
            # Add all columns from this class to failed_imputation_columns
            if cls not in failed_imputation_columns:
                failed_imputation_columns[cls] = []
            for col in valid_columns:
                if col not in failed_imputation_columns[cls]:
                    failed_imputation_columns[cls].append(col)
    
    # Handle rows with missing class values
    na_class_mask = df[class_column].isna()
    if na_class_mask.any():
        no_class_count = na_class_mask.sum()
        print(f"\nImputing {no_class_count} rows with missing class values")
        
        # Get rows with missing class
        na_class_rows = df[na_class_mask].copy()
        
        # Convert to numeric
        na_numeric = na_class_rows[impute_columns].copy()
        for col in impute_columns:
            na_numeric[col] = pd.to_numeric(na_numeric[col], errors='coerce')
        
        # Check for missing values
        missing_before = na_numeric.isna().sum().sum()
        
        if missing_before > 0:
            # Identify columns with enough data for imputation across the full dataset
            all_data_numeric = df[impute_columns].apply(pd.to_numeric, errors='coerce')
            full_valid_columns = []
            
            for col in impute_columns:
                non_missing = all_data_numeric[col].count()
                if non_missing >= 2:
                    full_valid_columns.append(col)
            
            if not full_valid_columns:
                print("  - No columns have enough data for imputation")
            else:
                try:
                    # Fit imputer on the full dataset
                    full_imputer = KNNImputer(n_neighbors=min(n_neighbors, len(df) - 1))
                    full_imputer.fit(all_data_numeric[full_valid_columns])
                    
                    # Transform only the rows with missing class
                    na_imputed = full_imputer.transform(na_numeric[full_valid_columns])
                    
                    # Convert back to dataframe
                    na_imputed_df = pd.DataFrame(na_imputed, 
                                              index=na_class_rows.index, 
                                              columns=full_valid_columns)
                    
                    # Update only the missing values
                    num_imputed = 0
                    for col in full_valid_columns:
                        missing_mask = na_numeric[col].isna()
                        if missing_mask.any():
                            # Get indices of missing values
                            missing_indices = na_class_rows[missing_mask].index
                            
                            # Always add variation for no-class rows
                            imputed_col_values = na_imputed_df.loc[missing_indices, col].values
                            
                            # Add variation based on global stats
                            if col in global_stats and not pd.isna(global_stats[col]['std']):
                                variation_scale = global_stats[col]['std'] * variation_factor
                            else:
                                # Default small variation
                                variation_scale = abs(np.mean(imputed_col_values)) * 0.1 * variation_factor
                            
                            # Generate and add noise
                            noise = np.random.normal(0, variation_scale, size=len(imputed_col_values))
                            varied_values = imputed_col_values + noise
                            
                            # Apply bounds if requested
                            if apply_bounds and col in global_stats:
                                # Handle non-negative columns
                                should_be_non_negative = global_stats[col]['min'] >= 0 or any(
                                    kw in col.lower() for kw in ['tau', 'beta', 'total', 'score']
                                )
                                
                                if should_be_non_negative:
                                    # Fix negative values by reflecting them
                                    negative_mask = varied_values < 0
                                    if np.any(negative_mask):
                                        varied_values[negative_mask] = abs(varied_values[negative_mask]) * np.random.uniform(0.95, 1.05, size=np.sum(negative_mask))
                                
                                # Apply general bounds
                                lower_bound = global_stats[col]['lower_bound']
                                upper_bound = global_stats[col]['upper_bound']
                                varied_values = np.clip(varied_values, lower_bound, upper_bound)
                            
                            # Update with varied values
                            for i, idx in enumerate(missing_indices):
                                result_df.loc[idx, col] = varied_values[i]
                                num_imputed += 1
                    
                    imputed_by_class['No Class'] = num_imputed
                    total_imputed += num_imputed
                    
                    print(f"  - Successfully imputed {num_imputed} values for rows with missing class")
                    
                except Exception as e:
                    print(f"  - Error during imputation for rows with missing class: {e}")
        else:
            print("  - No missing values in rows with missing class")
    
    # Now apply similar class borrowing with variation for columns that couldn't be imputed within their class
    if failed_imputation_columns:
        print("\n=== Applying similar class borrowing imputation with variation ===")
        
        fallback_imputed_total = 0
        
        # Process each class with failed columns
        for cls, cols in failed_imputation_columns.items():
            if not cols:
                continue
                
            print(f"  - Attempting fallback imputation for class '{cls}' on columns: {cols}")
            
            # Get rows for this class
            class_mask = df[class_column] == cls
            subset = df[class_mask].copy()
            
            # Get similar classes for this class
            if cls not in class_similarities:
                print(f"    - No similarity data available for class '{cls}'")
                continue
                
            similar_classes = [(other_cls, sim) for other_cls, sim in class_similarities[cls].items() 
                             if other_cls != cls]
            similar_classes = sorted(similar_classes, key=lambda x: x[1], reverse=True)
            
            if not similar_classes:
                print(f"    - No similar classes found for '{cls}'")
                continue
                
            # Process each column that needs fallback
            fallback_count = 0
            
            for col in cols:
                # Skip if column doesn't exist
                if col not in df.columns:
                    continue
                    
                # Convert to numeric
                subset[col] = pd.to_numeric(subset[col], errors='coerce')
                
                # Check for missing values in this column
                missing_mask = subset[col].isna()
                if not missing_mask.any():
                    continue
                    
                # Get indices of missing values
                missing_indices = subset[missing_mask].index
                
                # Try to find a similar class with enough data for this column
                found_fallback = False
                
                for similar_cls, similarity in similar_classes:
                    # Skip if similarity is too low
                    if similarity < 0.4:  # Minimum similarity threshold
                        continue
                        
                    # Get data for the similar class
                    similar_mask = df[class_column] == similar_cls
                    similar_subset = df[similar_mask].copy()
                    
                    # Check if similar class has enough data for this column
                    similar_subset[col] = pd.to_numeric(similar_subset[col], errors='coerce')
                    non_missing = similar_subset[col].count()
                    
                    if non_missing >= min_samples_required:
                        print(f"    - Using class '{similar_cls}' (similarity={similarity:.3f}) for column '{col}'")
                        
                        # Check if we have enough samples for KNN
                        effective_k = min(n_neighbors, max(1, len(similar_subset) - 1))
                        
                        # Option 1: If the similar class has enough samples, try KNN imputation
                        if len(similar_subset) >= 5:  # Minimum samples for reliable KNN
                            try:
                                # Create a temporary dataframe with both classes
                                combined_df = pd.concat([similar_subset, subset])
                                
                                # Apply KNN imputation
                                combined_numeric = combined_df[[col]].copy()
                                imputer = KNNImputer(n_neighbors=effective_k, weights='distance')
                                imputed_values = imputer.fit_transform(combined_numeric)
                                
                                # Extract only our class's imputed values
                                imputed_df = pd.DataFrame(imputed_values, 
                                                        index=combined_df.index, 
                                                        columns=[col])
                                
                                # Add variation to imputed values
                                for idx in missing_indices:
                                    imputed_value = imputed_df.loc[idx, col]
                                    
                                    # Calculate variation based on similar class statistics
                                    similar_std = similar_subset[col].std()
                                    if pd.isna(similar_std) or similar_std == 0:
                                        if col in global_stats and not pd.isna(global_stats[col]['std']):
                                            similar_std = global_stats[col]['std']
                                        else:
                                            similar_std = abs(imputed_value) * 0.1 if imputed_value != 0 else 1.0
                                    
                                    # Add variation
                                    variation_scale = similar_std * variation_factor * 0.7
                                    noise = np.random.normal(0, variation_scale)
                                    varied_value = imputed_value + noise
                                    
                                    # Handle bounds
                                    if apply_bounds and col in global_stats:
                                        # Handle non-negative columns
                                        should_be_non_negative = global_stats[col]['min'] >= 0 or any(
                                            kw in col.lower() for kw in ['tau', 'beta', 'total', 'score']
                                        )
                                        
                                        if should_be_non_negative and varied_value < 0:
                                            varied_value = abs(varied_value) * np.random.uniform(0.95, 1.05)
                                        
                                        # Apply bounds
                                        lower_bound = global_stats[col]['lower_bound']
                                        upper_bound = global_stats[col]['upper_bound']
                                        varied_value = max(lower_bound, min(varied_value, upper_bound))
                                    
                                    # Update the result dataframe
                                    result_df.loc[idx, col] = varied_value
                                    fallback_count += 1
                                
                                found_fallback = True
                                break
                                
                            except Exception as e:
                                print(f"      - Error using KNN from similar class: {e}")
                        
                        # Option 2: If KNN fails or not enough samples, use statistical distribution
                        if not found_fallback:
                            try:
                                # Get mean and std from similar class
                                similar_mean = similar_subset[col].mean()
                                similar_std = similar_subset[col].std()
                                
                                if not pd.isna(similar_mean):
                                    # Either use mean directly (if std is unavailable) or generate from distribution
                                    if pd.isna(similar_std) or similar_std == 0:
                                        # If no std available, get global std or use a small fraction of mean
                                        if col in global_stats and not pd.isna(global_stats[col]['std']):
                                            similar_std = global_stats[col]['std'] * 0.5  # Reduced variation
                                        else:
                                            similar_std = abs(similar_mean) * 0.1 if similar_mean != 0 else 1.0
                                    
                                    # Generate values from distribution with variation
                                    for idx in missing_indices:
                                        # Generate with appropriate variation
                                        value = np.random.normal(similar_mean, similar_std * 0.8)
                                        
                                        # Handle bounds
                                        if apply_bounds and col in global_stats:
                                            # Handle non-negative columns
                                            should_be_non_negative = global_stats[col]['min'] >= 0 or any(
                                                kw in col.lower() for kw in ['tau', 'beta', 'total', 'score']
                                            )
                                            
                                            if should_be_non_negative and value < 0:
                                                value = abs(value) * np.random.uniform(0.95, 1.05)
                                            
                                            # Apply bounds
                                            lower_bound = global_stats[col]['lower_bound']
                                            upper_bound = global_stats[col]['upper_bound']
                                            value = max(lower_bound, min(value, upper_bound))
                                        
                                        # Update result
                                        result_df.loc[idx, col] = value
                                        fallback_count += 1
                                    
                                    found_fallback = True
                                    break
                            except Exception as e:
                                print(f"      - Error using statistical distribution from similar class: {e}")
                
                if not found_fallback:
                    print(f"    - Could not find suitable similar class for column '{col}'")
                    
                    # Last resort: use global statistics with higher variation
                    if apply_bounds and col in global_stats:
                        print(f"    - Using global statistics as last resort for column '{col}'")
                        
                        global_mean = global_stats[col]['mean']
                        global_std = global_stats[col]['std']
                        
                        if not pd.isna(global_mean) and not pd.isna(global_std) and global_std > 0:
                            # Generate values from global distribution with higher variation
                            for idx in missing_indices:
                                # Use higher variation for this fallback approach
                                value = np.random.normal(global_mean, global_std * 0.9)
                                
                                # Handle bounds
                                should_be_non_negative = global_stats[col]['min'] >= 0 or any(
                                    kw in col.lower() for kw in ['tau', 'beta', 'total', 'score']
                                )
                                
                                if should_be_non_negative and value < 0:
                                    value = abs(value) * np.random.uniform(0.95, 1.05)
                                
                                # Apply bounds
                                lower_bound = global_stats[col]['lower_bound']
                                upper_bound = global_stats[col]['upper_bound']
                                value = max(lower_bound, min(value, upper_bound))
                                
                                # Update result
                                result_df.loc[idx, col] = value
                                fallback_count += 1
                            
                            print(f"      - Successfully imputed {len(missing_indices)} values using global statistics")
            
            # Update statistics
            if fallback_count > 0:
                fallback_name = f"{cls} (Similar Class Borrowing with Variation)"
                if fallback_name not in imputed_by_class:
                    imputed_by_class[fallback_name] = 0
                imputed_by_class[fallback_name] += fallback_count
                total_imputed += fallback_count
                fallback_imputed_total += fallback_count
        
        print(f"\nTotal similar class borrowing imputed values with variation: {fallback_imputed_total}")
    
    # Print summary
    print("\n======= Imputation Summary =======")
    print(f"Total values imputed: {total_imputed}")
    for cls, count in sorted(imputed_by_class.items()):
        print(f"  - {cls}: {count} values")
    
    return result_df


def calculate_class_similarities(df, class_column, impute_columns):
    """
    Calculate similarity between different classes based on their feature distributions
    """
    classes = df[class_column].dropna().unique()
    similarities = {cls: {} for cls in classes}
    
    # Convert all data to numeric
    numeric_df = df[impute_columns].copy()
    for col in impute_columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    # Calculate class statistics
    class_stats = {}
    for cls in classes:
        class_mask = df[class_column] == cls
        class_data = numeric_df[class_mask]
        
        # Calculate mean and std for each column
        means = {}
        stds = {}
        non_missing = {}
        
        for col in impute_columns:
            means[col] = class_data[col].mean()
            stds[col] = class_data[col].std()
            non_missing[col] = class_data[col].count()
        
        class_stats[cls] = {
            'means': means,
            'stds': stds,
            'counts': non_missing
        }
    
    # Calculate similarity between each pair of classes
    for cls1 in classes:
        for cls2 in classes:
            if cls1 == cls2:
                similarities[cls1][cls2] = 1.0  # Perfect similarity with self
                continue
            
            # Get statistics for both classes
            stats1 = class_stats[cls1]
            stats2 = class_stats[cls2]
            
            # Find columns that have data in both classes
            common_cols = []
            for col in impute_columns:
                if (not pd.isna(stats1['means'][col]) and not pd.isna(stats2['means'][col]) and
                    stats1['counts'][col] >= 2 and stats2['counts'][col] >= 2):
                    common_cols.append(col)
            
            if not common_cols:
                similarities[cls1][cls2] = 0.0  # No common data
                continue
            
            # Calculate similarity based on normalized Euclidean distance between means
            distances = []
            
            for col in common_cols:
                # Get means
                mean1 = stats1['means'][col]
                mean2 = stats2['means'][col]
                
                # Get column range across all classes to normalize
                all_values = numeric_df[col].dropna()
                col_range = all_values.max() - all_values.min() if len(all_values) > 0 else 1.0
                
                if col_range == 0:  # Avoid division by zero
                    col_range = 1.0
                
                # Calculate normalized squared difference
                norm_diff = ((mean1 - mean2) / col_range) ** 2
                distances.append(norm_diff)
            
            # Calculate overall distance and convert to similarity
            if distances:
                distance = np.sqrt(np.mean(distances))
                similarity = 1.0 / (1.0 + distance)
                similarities[cls1][cls2] = similarity
            else:
                similarities[cls1][cls2] = 0.0
    
    return similarities


# Example usage
if __name__ == "__main__":
    # Load the dataset
    file_path = 'cognid_processed.xlsx'
    df = pd.read_excel(file_path)
    
    # Define columns to impute
    impute_columns = [
        'Total Tau pg/ml (146-595)', 
        'Phospho Tau pg/ml (24-68)', 
        'A Beta 142 pg/ml (627-1322)', 
        'Total', 
        'Attention', 
        'mem', 
        'fluency', 
        'language', 
        'visuospatial', 
        'phonemic fluency (letters)', 
        'semantic fluency (animal)'
    ]
    
    # Apply enhanced KNN imputation with variation for small classes
    df_imputed = knn_impute_by_class_with_variation(
        df, 
        class_column='Completed Diagnosis', 
        impute_columns=impute_columns, 
        n_neighbors=5,
        min_samples_required=2,  # Minimum non-missing values required for class-based imputation
        variation_factor=0.3,    # 30% variation based on standard deviation
        apply_bounds=True,       # Apply reasonable bounds to prevent extreme values
        random_state=42          # For reproducibility
    )
    
    # Save the imputed data
    output_path = 'cognid_knn_imputed_with_variation.xlsx'
    df_imputed.to_excel(output_path, index=False)
    print(f"\nSaved imputed data to: {output_path}")
    
    # Print before/after missing values count
    original_missing = df[impute_columns].apply(pd.to_numeric, errors='coerce').isna().sum().sum()
    imputed_missing = df_imputed[impute_columns].apply(pd.to_numeric, errors='coerce').isna().sum().sum()
    print(f"Original missing values: {original_missing}")
    print(f"Remaining missing values: {imputed_missing}")
    if original_missing > 0:
        print(f"Imputation success rate: {100 * (1 - imputed_missing/original_missing):.2f}%")