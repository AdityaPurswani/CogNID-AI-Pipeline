import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, Matern
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split



def load_and_prepare_data(data_path, target_column, feature_columns=None):
    """
    Load and prepare the dataset for modeling.
    
    Parameters:
    -----------
    data_path : str
        Path to the imputed dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    
    Returns:
    --------
    tuple
        (X, y, feature_columns, class_names)
    """
    # Load data
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    # Handle missing values in the target column
    if df[target_column].isna().sum() > 0:
        print(f"Warning: Found {df[target_column].isna().sum()} missing values in target column. Dropping these rows.")
        df = df.dropna(subset=[target_column])
    
    # Automatically select features if not specified
    if feature_columns is None:
        # Select numeric columns and exclude the target
        feature_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        print(f"Automatically selected {len(feature_columns)} numeric features")
    
    # Validate feature columns
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Features {missing_features} not found in the dataset")
    
    # Prepare features and target
    X = df[feature_columns].copy()
    
    # Encode target if it's categorical
    if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
        print(f"Encoding categorical target: {target_column}")
        le = LabelEncoder()
        y = le.fit_transform(df[target_column])
        class_names = le.classes_
        print(f"Classes: {class_names}")
    else:
        y = df[target_column].values
        class_names = np.unique(y)
    
    # Convert X to numeric, handling any remaining missing values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Check for any remaining missing values
    missing_count = X.isna().sum().sum()
    if missing_count > 0: # After performing KNN class based data imputation if still there are missing values impute it with 0
        print(f"Warning: Found {missing_count} missing values in feature columns. impute it with value 0")
        X = X.fillna(0)
    
    return X, y, feature_columns, class_names


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets and scale features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : array-like
        Target variable
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_adasyn(X_train, y_train, use_resampling=True, n_neighbors=5, random_state=42):
    """
    Apply oversampling to handle class imbalance, trying ADASYN first with SMOTE as fallback.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training target variable
    use_resampling : bool, default=True
        Whether to use oversampling for imbalanced classes
    n_neighbors : int, default=5
        Number of neighbors to use in resampling algorithms
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train_resampled, y_train_resampled, resampling_method)
    """
    from imblearn.over_sampling import ADASYN, SMOTE
    import numpy as np
    
    # Check for class imbalance
    train_class_counts = np.bincount(y_train)
    print("Class distribution in training set:", train_class_counts)
    
    # Apply resampling if there's significant class imbalance
    imbalance_ratio = max(train_class_counts) / min(train_class_counts)
    
    if not use_resampling:
        print("Resampling is disabled. Using original class distribution.")
        return X_train, y_train, "none"
    
    if imbalance_ratio <= 1.5:
        print(f"Class imbalance ratio ({imbalance_ratio:.2f}) is acceptable. Skipping resampling.")
        return X_train, y_train, "none"
    
    if min(train_class_counts) < 3:
        print(f"Minority class has too few samples ({min(train_class_counts)}) for reliable resampling.")
        return X_train, y_train, "none"
    
    # Create a conservative sampling strategy - don't fully balance classes
    # Target for minority classes is 70% of majority class size
    sampling_strategy = {
        cls: min(int(max(train_class_counts) * 0.7), int(max(train_class_counts)))  # At most double, up to 70% of majority
        for cls, count in enumerate(train_class_counts) 
        if count < max(train_class_counts) / 1.5  # Only oversample if significantly imbalanced
    }
    
    # Ensure effective neighbors is appropriate
    min_samples = min(train_class_counts)
    effective_neighbors = min(n_neighbors, min_samples - 1)
    effective_neighbors = max(1, effective_neighbors)  # Ensure at least 1
    
    # Try ADASYN first
    try:
        print(f"Detected class imbalance (ratio: {imbalance_ratio:.2f}). Trying ADASYN.")
        
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=effective_neighbors,
            random_state=random_state
        )
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        
        print(f"After ADASYN - Training set size: {X_train_resampled.shape[0]}")
        print("Class distribution after ADASYN:", np.bincount(y_train_resampled))
        return X_train_resampled, y_train_resampled, "adasyn"
        
    except Exception as e:
        print(f"ADASYN failed: {e}")
        print("Trying SMOTE as fallback...")
        
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=effective_neighbors,
                random_state=random_state
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"After SMOTE - Training set size: {X_train_resampled.shape[0]}")
            print("Class distribution after SMOTE:", np.bincount(y_train_resampled))
            return X_train_resampled, y_train_resampled, "smote"
            
        except Exception as smote_e:
            print(f"SMOTE also failed: {smote_e}")
            print("Proceeding with original imbalanced data.")
            return X_train, y_train, "none"

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X_test : array-like
        Test feature matrix
    y_test : array-like
        Test target variable
    class_names : array-like
        Names of the target classes
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For probability-based metrics, check if model has predict_proba
    has_predict_proba = hasattr(model, "predict_proba")
    if has_predict_proba:
        y_pred_proba = model.predict_proba(X_test)
    else:
        # For models without predict_proba, use a placeholder
        y_pred_proba = None
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # For multi-class, calculate macro average for ROC AUC if probabilities available
    if has_predict_proba:
        if len(class_names) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = None
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, target_names=[str(name) for name in class_names])
    print(class_report)
    
    # Return evaluation metrics
    test_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return test_metrics


def get_feature_importance(model, feature_columns, model_type):
    """
    Extract feature importance from the trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_columns : list
        Names of feature columns
    model_type : str
        Type of the model ('xgboost', 'rf', 'dt', 'svm', 'dnn')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame of feature importances
    """
    # Initialize empty dataframe
    feature_importance = pd.DataFrame({'Feature': feature_columns})
    
    # Get feature importances based on model type
    if model_type.lower() in ['xgboost', 'rf', 'dt']:
        # Models with built-in feature importance
        importance_values = model.feature_importances_
        feature_importance['Importance'] = importance_values
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
    else:
        # Models without direct feature importance
        feature_importance['Importance'] = np.nan
        print("\nFeature importance not available for this model type.")
    
    return feature_importance

def plot_results(X_test, y_test, test_metrics, feature_importance, class_names, 
                feature_names, model_name, used_adasyn=False):
    """
    Create visualizations of model results and display them in the terminal
    
    Parameters:
    -----------
    X_test : array-like
        Test feature matrix
    y_test : array-like
        Test target variable
    test_metrics : dict
        Dictionary of evaluation metrics
    feature_importance : pandas.DataFrame
        DataFrame of feature importances
    class_names : array-like
        Names of the target classes
    feature_names : list
        Names of feature columns
    model_name : str
        Name of the model for plot titles
    used_adasyn : bool, default=False
        Whether ADASYN was used in training
    """
    y_pred = test_metrics['y_pred']
    y_pred_proba = test_metrics['y_pred_proba']
    
    # Set style
    sns.set(style="whitegrid")
    
    # Feature importance plot if available
    if not feature_importance['Importance'].isna().all():
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(min(15, len(feature_importance)))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'{model_name} Top Feature Importance')
        plt.tight_layout()
        plt.show()  # Show in terminal instead of saving
    
    
    # Confusion Matrix heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred).astype(int)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 10, "color": "black"})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title_suffix = " (with ADASYN)" if used_adasyn else ""
    plt.title(f'{model_name} Confusion Matrix{title_suffix}')
    plt.tight_layout()
    plt.show()  # Show in terminal instead of saving
    
    # ROC curve and Precision-Recall curve if probabilities are available
    if y_pred_proba is not None:
        # ROC curve
        plt.figure(figsize=(8, 6))
        
        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            title_suffix = " (with ADASYN)" if used_adasyn else ""
            plt.title(f'{model_name} ROC Curve{title_suffix}')
            plt.legend(loc='lower right')
        else:
            # Multi-class classification - one-vs-rest ROC curves
            for i, class_name in enumerate(class_names):
                # For multiclass, we consider each class as positive and rest as negative
                y_test_bin = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba[:, i])
                
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            title_suffix = " (with ADASYN)" if used_adasyn else ""
            plt.title(f'{model_name} ROC Curves (One-vs-Rest{title_suffix})')
            plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()  # Show in terminal instead of saving
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        
        if len(class_names) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            pr_score = average_precision_score(y_test, y_pred_proba[:, 1])
            
            plt.plot(recall, precision, label=f'PR = {pr_score:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            title_suffix = " (with ADASYN)" if used_adasyn else ""
            plt.title(f'{model_name} Precision-Recall Curve{title_suffix}')
            plt.legend(loc='lower left')
        else:
            # Multi-class - one vs rest precision-recall curves
            for i, class_name in enumerate(class_names):
                y_test_bin = (y_test == i).astype(int)
                precision, recall, _ = precision_recall_curve(y_test_bin, y_pred_proba[:, i])
                ap_score = average_precision_score(y_test_bin, y_pred_proba[:, i])
                
                plt.plot(recall, precision, label=f'{class_name} (AP = {ap_score:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            title_suffix = " (with ADASYN)" if used_adasyn else ""
            plt.title(f'{model_name} Precision-Recall Curves (One-vs-Rest{title_suffix})')
            plt.legend(loc='lower left')
        
        plt.tight_layout()
        plt.show()  # Show in terminal instead of saving
        
def compare_models(models_results, metric='accuracy'):
    """
    Compare multiple models based on specified metric.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary of model results with model names as keys
    metric : str, default='accuracy'
        Metric to use for comparison ('accuracy', 'roc_auc')
    
    Returns:
    --------
    pandas.DataFrame
        Comparison of models
    """
    results = []
    for model_name, model_data in models_results.items():
        _, _, test_metrics, _, _ = model_data
        
        # For ROC AUC, check if it's available (some models might not have predict_proba)
        if metric.lower() == 'roc_auc' and test_metrics['roc_auc'] is None:
            print(f"Warning: ROC AUC not available for {model_name}. Using accuracy instead.")
            metric_value = test_metrics['accuracy']
        else:
            metric_value = test_metrics[metric.lower()]
        
        results.append({
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'ROC_AUC': test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else np.nan
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Convert metric name to match DataFrame column format
    metric_col = 'Accuracy' if metric.lower() == 'accuracy' else 'ROC_AUC'
    comparison_df = comparison_df.sort_values(metric_col, ascending=False)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric_col, data=comparison_df)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.tight_layout()
    plt.savefig(f'model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df

def build_xgboost_model(data_path, target_column, feature_columns=None, 
                       test_size=0.2, use_adasyn=True, adasyn_neighbors=5, 
                       param_grid=None, model_name="XGBoost", random_state=42):
    """
    Build and evaluate an XGBoost model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    param_grid : dict, optional
        Grid of parameters to search during hyperparameter tuning
    model_name : str, default="XGBoost"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3]
        }
    
    # Create base model
    base_model = XGBClassifier(
        random_state=random_state,
        eval_metric='logloss'
    )
    
    # Cross-validation to assess baseline performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(base_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy', verbose=3)
    print(f"Cross-validation accuracy (default params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model on the full training set
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # Get feature importance
    feature_importance = get_feature_importance(best_model, feature_columns, 'xgboost')
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return best_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_random_forest_model(data_path, target_column, feature_columns=None, 
                            test_size=0.2, use_adasyn=True, adasyn_neighbors=5, 
                            param_grid=None, model_name="RandomForest", random_state=42):
    """
    Build and evaluate a Random Forest model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    param_grid : dict, optional
        Grid of parameters to search during hyperparameter tuning
    model_name : str, default="RandomForest"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=random_state)
    
    # Cross-validation to assess baseline performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(base_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy (default params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model on the full training set
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # Get feature importance
    feature_importance = get_feature_importance(best_model, feature_columns, 'rf')
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return best_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_svm_model(data_path, target_column, feature_columns=None, 
                      test_size=0.2, use_adasyn=True, adasyn_neighbors=5, 
                      param_grid=None, model_name="SVM", random_state=42):
    """
    Build and evaluate an SVM model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    param_grid : dict, optional
        Grid of parameters to search during hyperparameter tuning
    model_name : str, default="SVM"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly'],
            'probability': [True]  # Needed for ROC curves
        }
    
    # Create base model
    base_model = SVC(random_state=random_state, probability=True)
    
    # Cross-validation to assess baseline performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(base_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy (default params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model on the full training set
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # Get feature importance - SVMs don't have built-in feature importance
    feature_importance = get_feature_importance(best_model, feature_columns, 'svm')
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return best_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_neural_network_model(data_path, target_column, feature_columns=None, 
                             test_size=0.2, use_adasyn=True, adasyn_neighbors=5, 
                             param_grid=None, model_name="NeuralNetwork", random_state=42):
    """
    Build and evaluate a Neural Network model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    param_grid : dict, optional
        Grid of parameters to search during hyperparameter tuning
    model_name : str, default="NeuralNetwork"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'solver': ['adam', 'sgd']
        }
    
    # Create base model
    base_model = MLPClassifier(
        random_state=random_state,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1# Increasing max iterations for convergence
    )
    
    # Cross-validation to assess baseline performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(base_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy (default params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model on the full training set
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # Get feature importance - Neural Networks don't have built-in feature importance
    feature_importance = get_feature_importance(best_model, feature_columns, 'dnn')
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return best_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_knn_model(data_path, target_column, feature_columns=None, 
                  test_size=0.2, use_adasyn=True, adasyn_neighbors=5, 
                  param_grid=None, model_name="KNN", random_state=42):
    """
    Build and evaluate a K-Nearest Neighbors model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    param_grid : dict, optional
        Grid of parameters to search during hyperparameter tuning
    model_name : str, default="KNN"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
        }
    
    # Create base model - need to import KNeighborsClassifier at the top of the file
    base_model = KNeighborsClassifier()
    
    # Cross-validation to assess baseline performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(base_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy (default params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model on the full training set
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # KNN doesn't have feature importance, so create empty DataFrame
    feature_importance = get_feature_importance(best_model, feature_columns, 'knn')
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return best_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_stacking_ensemble(data_path, target_column, feature_columns=None, 
                          base_models=None, meta_model=None,
                          test_size=0.2, use_adasyn=True, adasyn_neighbors=5,
                          model_name="StackingEnsemble", random_state=42):
    """
    Build and evaluate a stacking ensemble model for classification.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    base_models : list, optional
        List of (name, model) tuples for base models
    meta_model : estimator, optional
        Meta-learner model that combines base model predictions
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    model_name : str, default="StackingEnsemble"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    print(f"\n{'='*50}")
    print(f"Building {model_name} Model")
    print(f"{'='*50}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Define default base models if not provided
    if base_models is None:
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=random_state)),
            ('svm', SVC(probability=True, random_state=random_state)),
        ]
    
    # Define default meta-model if not provided
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=100, multi_class='auto')
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=1
    )
    
    # Cross-validation to assess performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(stacking_model, X_train_resampled, y_train_resampled, cv=cv, n_jobs=1, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model on the full training set
    print("\nTraining final stacking ensemble model...")
    stacking_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    test_metrics = evaluate_model(stacking_model, X_test_scaled, y_test, class_names)
    
    # Stacking models don't have direct feature importance
    feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': np.nan})
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, model_name, used_adasyn
    )
    
    print(f"\n{model_name} model training and evaluation complete.")
    
    return stacking_model, feature_importance, test_metrics, X_test_scaled, y_test

def build_advanced_neural_network(data_path, target_column, feature_columns=None, 
                                nn_type='tabnet', batch_size=64, epochs=200,
                                test_size=0.2, use_adasyn=True, adasyn_neighbors=5,
                                model_name="AdvancedNN", random_state=42):
    """
    Build and evaluate an advanced neural network model for classification using PyTorch.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    target_column : str
        Column name of the target variable
    feature_columns : list, optional
        List of column names to use as features, if None will use all numeric columns
    nn_type : str, default='tabnet'
        Type of neural network: 'tabnet', 'snn' (Self-Normalizing NN), or 'att' (Attention-based)
    batch_size : int, default=64
        Batch size for training
    epochs : int, default=200
        Maximum number of epochs for training
    test_size : float, default=0.2
        Proportion of data to use for testing
    use_adasyn : bool, default=True
        Whether to use ADASYN oversampling for imbalanced classes
    adasyn_neighbors : int, default=5
        Number of neighbors to use in ADASYN algorithm
    model_name : str, default="AdvancedNN"
        Name of the model for plot titles and filenames
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance, test_metrics, X_test, y_test)
    """
    print(f"\n{'='*50}")
    print(f"Building {model_name} ({nn_type.upper()}) Model with PyTorch")
    print(f"{'='*50}")
    
    # Try to import PyTorch
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Check for CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
    except ImportError:
        print("PyTorch is not installed. Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch"])
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Check for CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Load and prepare data
    X, y, feature_columns, class_names = load_and_prepare_data(
        data_path, target_column, feature_columns
    )
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    # Apply ADASYN if needed
    X_train_resampled, y_train_resampled, used_adasyn = apply_adasyn(
        X_train_scaled, y_train, use_adasyn, adasyn_neighbors, random_state
    )
    
    # Get dimensions
    n_features = X_train_resampled.shape[1]
    n_classes = len(np.unique(y))
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_resampled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    if n_classes > 2:
        y_train_tensor = torch.LongTensor(y_train_resampled)
        y_test_tensor = torch.LongTensor(y_test)
    else:
        y_train_tensor = torch.FloatTensor(y_train_resampled).view(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Define model architectures
    
    # Self-Normalizing Neural Network
    class SNNModel(nn.Module):
        def __init__(self, n_features, n_classes):
            super(SNNModel, self).__init__()
            self.n_classes = n_classes
            
            # Using SELU activation for self-normalization
            self.layer1 = nn.Linear(n_features, 128)
            self.selu1 = nn.SELU()
            self.dropout1 = nn.AlphaDropout(p=0.1)
            
            self.layer2 = nn.Linear(128, 64)
            self.selu2 = nn.SELU()
            self.dropout2 = nn.AlphaDropout(p=0.1)
            
            self.layer3 = nn.Linear(64, 32)
            self.selu3 = nn.SELU()
            
            if n_classes > 2:
                self.output = nn.Linear(32, n_classes)
            else:
                self.output = nn.Linear(32, 1)
                
        def forward(self, x):
            x = self.layer1(x)
            x = self.selu1(x)
            x = self.dropout1(x)
            
            x = self.layer2(x)
            x = self.selu2(x)
            x = self.dropout2(x)
            
            x = self.layer3(x)
            x = self.selu3(x)
            
            x = self.output(x)
            
            if self.n_classes == 2:
                x = torch.sigmoid(x)
                
            return x
    
    # Basic model with attention
    class AttentionModel(nn.Module):
        def __init__(self, n_features, n_classes):
            super(AttentionModel, self).__init__()
            self.n_classes = n_classes
            
            # Embedding layer
            self.embedding = nn.Linear(n_features, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu1 = nn.ReLU()
            
            # Self-attention (simplified)
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            
            # Output layers
            self.fc1 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
            if n_classes > 2:
                self.output = nn.Linear(32, n_classes)
            else:
                self.output = nn.Linear(32, 1)
                
        def forward(self, x):
            # Initial embedding
            x = self.embedding(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            # Self-attention
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            
            # Dot-product attention (simplified)
            attention_scores = torch.mm(q, k.t()) / (64 ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=1)
            x = torch.mm(attention_weights, v)
            
            # Output
            x = self.fc1(x)
            x = self.relu2(x)
            x = self.dropout(x)
            x = self.output(x)
            
            if self.n_classes == 2:
                x = torch.sigmoid(x)
                
            return x
    
    if nn_type.lower() == 'snn':
        model = SNNModel(n_features, n_classes)
    elif nn_type.lower() == 'att':
        model = AttentionModel(n_features, n_classes)
    else:
        raise ValueError(f"Unknown nn_type: {nn_type}. Use 'tabnet', 'snn', or 'att'.")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    if n_classes > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define early stopping parameters
    early_stopping = True
    patience = 15  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as improvement
    best_loss = float('inf')
    wait_count = 0
    best_model_state = None
    
    # Split data for validation
    val_size = int(0.1 * len(X_train_tensor))  # 10% for validation
    train_size = len(X_train_tensor) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    print("\nTraining PyTorch model with early stopping...")
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        valid_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping logic
        if early_stopping:
            if avg_val_loss < best_loss - min_delta:  # Improvement
                best_loss = avg_val_loss
                wait_count = 0
                # Save the best model
                best_model_state = model.state_dict().copy()
            else:  # No improvement
                wait_count += 1
                if wait_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
    
    # If we completed all epochs without early stopping
    if epoch == epochs - 1:
        print(f"Completed all {epochs} epochs.")
    else:
        print(f"Training stopped early at epoch {epoch+1}/{epochs}")
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        outputs = model(X_test_tensor)
        
        if n_classes > 2:
            # Multi-class
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        else:
            # Binary
            y_pred_proba_raw = outputs.cpu().numpy().flatten()
            y_pred = (y_pred_proba_raw > 0.5).astype(int)
            y_pred_proba = np.column_stack((1 - y_pred_proba_raw, y_pred_proba_raw))
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    if n_classes > 2:
        # One-hot encode y_test for multi-class ROC AUC
        y_test_one_hot = np.zeros((len(y_test), n_classes))
        for i, val in enumerate(y_test):
            y_test_one_hot[i, val] = 1
        roc_auc = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovr', average='macro')
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, target_names=[str(name) for name in class_names])
    print(class_report)
    
    # Create test_metrics dictionary
    test_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Extract feature importance for TabNet
    if hasattr(model, 'attention_weights'):
        # Average attention weights across batches
        if model.attention_weights is not None:
            avg_weights = np.mean(model.attention_weights, axis=0)
            
            # Normalize
            if np.sum(avg_weights) > 0:
                avg_weights = avg_weights / np.sum(avg_weights)
                
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': avg_weights
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance (from attention weights):")
            print(feature_importance.head(10))
        else:
            feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': np.nan})
    else:
        # No direct feature importance for other models
        feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': np.nan})
    
    # Plot results
    plot_results(
        X_test_scaled, y_test, test_metrics, feature_importance, 
        class_names, feature_columns, f"{model_name}_{nn_type}_pytorch", used_adasyn
    )
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    
    # Mark early stopping point if applicable
    if early_stopping and epoch < epochs - 1:
        stop_epoch = epoch - patience + 1
        plt.axvline(x=stop_epoch, color='r', linestyle='--', 
                   label=f'Early Stopping Point (epoch {stop_epoch+1})')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{model_name} ({nn_type}) PyTorch model training and evaluation complete.")
    
    # Create a wrapper class for sklearn compatibility
    class PyTorchModelWrapper:
        def __init__(self, pytorch_model, is_multiclass=False, device='cpu'):
            self.model = pytorch_model.to('cpu')  # Move to CPU for inference
            self.model.eval()  # Set to evaluation mode
            self.is_multiclass = is_multiclass
            
        def predict(self, X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                
                if self.is_multiclass:
                    _, predicted = torch.max(outputs, 1)
                    return predicted.numpy()
                else:
                    return (outputs.numpy() > 0.5).astype(int).flatten()
            
        def predict_proba(self, X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                
                if self.is_multiclass:
                    return torch.softmax(outputs, dim=1).numpy()
                else:
                    probs = outputs.numpy().flatten()
                    return np.column_stack((1 - probs, probs))
    
    # Create the wrapper model
    wrapped_model = PyTorchModelWrapper(model, is_multiclass=(n_classes > 2))
    
    return wrapped_model, feature_importance, test_metrics, X_test_scaled, y_test


