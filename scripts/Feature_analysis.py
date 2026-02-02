#TODO: Clean up interim files and add comments
#TODO: Deal iwth edge cases (incorrect column names, missing columns, etc.)
#TODO: Add comments that specify the equations used for feature importance calculations in sklearn
#TODO: add arguments where the user can specify which are the target columns for ML training
#TODO: add arguments that specify an output directory for files and plots

#importing libraries
import pandas as pd
import numpy as np
import os
import argparse

#----------------block for parsing arguments---------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Script to calculate ML feature importances from ATAC-seq peaks and features."
        " Excludes features specified in the command line arguments."
        " Requires bed files in ../data/bed/ "
    )
    
    p.add_argument(
        "-x", "--exclude_features",
        type=str,
        nargs='*',
        default=[],
        help="List of features to exclude from the feature matrix (e.g., ['feature1', 'feature2'])."
        " Features should be specified as the name of the bed file without the .bed extension."
    )

    p.add_argument(
        "-i", "--input_matrix",
        type=str,
        required=True,
        help="Input peak matrix"
    )

    p.add_argument(
        "-o", "--output_dir",
        type=str,
        default="../data/output/",
        help="Output directory for the feature matrix and plots."
    )

    p.add_argument(
        "-fc", "--fold_change",
        type=str,
        default="log2",
        help="Fold change column name to find in input matrix"
    )

    p.add_argument(
        "-d", "--diff_peaks",
        type=str,
        default="change",
        help="Differential peak accessibility column name to find in input matrix"
    )

    p.add_argument(
        "-p",
        action="store_true",
        help="If set, will calculate permutation importances for the Random Forest model."
    )

    p.add_argument(
        "-j", "--threads",
        type=int,
        default=1,
        help="Number of parallel threads to use for the pipeline. Default is 1."
    )

    args = p.parse_args()

    return args
#--------------------------end of block--------------------------
#--------------block for making feature matrix-------------------
def make_feature_matrix(bed_list, feature_matrix_input, fc_column, diff_peaks_column):
    """
    Overlaps the ATAC-seq peaks with the features in the bed_list and returns a binary feature matrix.
    Parameters:
    bed_list: list of bed files containing features
    feature_matrix: pandas DataFrame with features as columns and ATAC-seq peaks as rows with target as log2 fold change
    """
    #calculate overlap between ATAC-seq peaks and features using bedtools overlap

    #generate bed file from feature matrix input
    #keeping first three columns (names are not known)
    matrix_input_df = pd.read_csv(feature_matrix_input, sep="\t")
    bed_out_df = matrix_input_df.iloc[:, :3].copy()
    #adding name column with peak names --> Peak_i
    bed_out_df['name'] = ['Peak_' + str(i) for i in range(len(bed_out_df))]
    bed_out_df[fc_column] = matrix_input_df[fc_column]
    bed_out_df[diff_peaks_column] = matrix_input_df[diff_peaks_column]

    #if output directory does not exist, create it
    if not os.path.exists("../data/output"):
        os.makedirs("../data/output", exist_ok=True)

    #save as bed file
    bed_out_df.to_csv("../data/output/final_merged_peaks.bed", sep="\t", header=False, index=False)

    #if ./data/overlap_beds exists, delete it and create a new one
    if os.path.exists("../data/output/overlap_beds"):
        os.system("rm -r ../data/output/overlap_beds")
    os.makedirs("../data/output/overlap_beds", exist_ok=True)

    for bed_file in bed_list:
        #get the name of the bed file without the path
        bed_name = os.path.basename(bed_file).split('.')[0]
        #use bedtools intersect to get the overlap between ATAC-seq peaks and features
        os.system(f"bedtools intersect -a ../data/output/final_merged_peaks.bed -b {bed_file} -wa -wb > ../data/output/overlap_beds/{bed_name}_overlap.bed")

    #read the overlap files and create a binary feature matrix
    feature_matrix = pd.DataFrame(columns=['peak_name', fc_column, diff_peaks_column] + [os.path.basename(f).split('.')[0] for f in bed_list])
    #initialize the feature matrix with the ATAC-seq peaks
    feature_matrix['peak_name'] = bed_out_df['name']
    feature_matrix[fc_column] = bed_out_df[fc_column]
    feature_matrix[diff_peaks_column] = bed_out_df[diff_peaks_column]

    #set the index to the peak names
    feature_matrix.set_index('peak_name', inplace=True)

    for bed_file in bed_list:
        #get the name of the bed file without the path
        bed_name = os.path.basename(bed_file).split('.')[0]
        #read the overlap file
        overlap_file = f"../data/output/overlap_beds/{bed_name}_overlap.bed"
        if os.path.exists(overlap_file):
            # Always create a binary column for the feature, initialized to 0
            feature_matrix[bed_name] = 0
            # If overlap file is not empty, set the value to 1 for the peaks that overlap with the feature
            if os.path.getsize(overlap_file) != 0:
                overlap_df = pd.read_csv(overlap_file, sep="\t", header=None)
                # overlap_df[3] contains the peak names
                overlapping_peaks = overlap_df[3].unique()
                feature_matrix.loc[feature_matrix.index.isin(overlapping_peaks), bed_name] = 1

    #fill NaN values with 0
    feature_matrix.fillna(0, inplace=True)
    #save the feature matrix to a csv file
    feature_matrix.to_csv("../data/output/feature_matrix.csv")

    return feature_matrix

#--------------------------end of block--------------------------

#-------------------block for machine learning-------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid

def random_forest(feature_matrix, target_column, fold_change, threads=1):
    """
    Train a Random Forest classifier on the feature matrix.
    Parameters:
    feature_matrix: pandas DataFrame with features as columns and ATAC-seq peaks as rows with target as log2 fold change
    target_column: column name for the target variable (default is 'target')
    """
    # Drop logFC column if it exists
    if fold_change in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=[fold_change])

    # Balance the dataset by undersampling
    class_counts = feature_matrix[target_column].value_counts()
    min_class_count = class_counts.min()
    balanced_data = pd.concat([
        feature_matrix[feature_matrix[target_column] == cls].sample(min_class_count, random_state=3)
        for cls in class_counts.index
    ])
    # Shuffle the balanced data
    balanced_data = balanced_data.sample(frac=1, random_state=3).reset_index(drop=True)
    feature_matrix = balanced_data

    # Split the data into features and target
    X = feature_matrix.drop(columns=[target_column])
    y = feature_matrix[target_column]

    # 70:15:15 split for train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3, stratify=y_temp)

    # Hyperparameter tuning using validation set

    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        clf = RandomForestClassifier(random_state=3, n_jobs=threads, **params)
        clf.fit(X_train, y_train)
        val_score = clf.score(X_val, y_val)
        if val_score > best_score:
            best_score = val_score
            best_params = params

    print(f"Best hyperparameters: {best_params}")
    print(f"Validation Accuracy (best): {best_score:.2f}")

    # Train final model with best hyperparameters on train+val, test on test set
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    rf_classifier = RandomForestClassifier(random_state=3, n_jobs=threads, **best_params)
    rf_classifier.fit(X_trainval, y_trainval)

    # Evaluate on test set
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    #---------------block for plotting performance metrics-------------------
    # Plot ROC and Precision-Recall curves on the same figuer side by side
    fpr, tpr, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
    pr_auc = auc(recall, precision)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(['ROC curve (area = {:.2f})'.format(roc_auc), 'Chance Line (y=x)'], loc='lower right')
    ax2.plot(recall, precision, color='green', label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    ax2.plot([0, 1], [0.5, 0.5], color='grey', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(['Precision-Recall curve (area = {:.2f})'.format(pr_auc), 'Chance Line (y=0.5)'], loc='upper right')
    plt.tight_layout()
    plt.savefig("../data/output/roc_pr_curves.pdf", format='pdf')
    #--------------------------end of block--------------------------------
    
    return rf_classifier, X_val, y_val

def random_forest_feature_importance(rf_classifier, X_eval, y_eval, permutation, threads=1):
    # MDI importances
    mdi_imps = rf_classifier.feature_importances_
    fi_df = pd.DataFrame({
        'feature': rf_classifier.feature_names_in_,
        'importance': mdi_imps
    }).sort_values('importance', ascending=False)

    mean_imp = fi_df['importance'].mean()
    top30 = fi_df.head(30).copy()
    top30.loc[len(top30)] = ['mean_value', mean_imp]

    plt.figure(figsize=(6,8))
    sns.barplot(x='importance', y='feature', data=top30, color=sns.xkcd_rgb['dark peach'])
    plt.title('MDI Feature Importances')
    plt.tight_layout()
    plt.savefig("../data/output/feature_importances_rf.pdf")

    if permutation:
        # If permutation is True, calculate permutation importances
        # Permutation importances
        from joblib import parallel_backend

        with parallel_backend("threading", n_jobs=threads):
            perm = permutation_importance(
                rf_classifier, X_eval, y_eval,
                n_repeats=10, random_state=3, n_jobs=threads
        )

        perm_df = pd.DataFrame({
            'feature': rf_classifier.feature_names_in_,
            'MDI_importance': fi_df['importance'],
            'permutation_importance': perm.importances_mean,
            'permutation_std': perm.importances_std
        }).sort_values('permutation_importance', ascending=False)

        plt.figure(figsize=(6, 8))
        top30_perm = perm_df.head(30)
        sns.boxplot(
            x='permutation_importance',
            y='feature',
            data=top30_perm,
            orient='h',
            color=sns.xkcd_rgb['dark peach']
        )
        plt.title('Permutation Importances (Boxplot)')
        plt.tight_layout()
        plt.savefig("../data/output/feature_importances_rf_permutation.pdf")

        #make return df
        rf_feature_importance_df = pd.DataFrame({
            'feature': perm_df['feature'],
            'MDI_importance': perm_df['MDI_importance'],
            'permutation_importance': perm_df['permutation_importance'],
            'permutation_std': perm_df['permutation_std']
        })
    else:
        # If permutation is False, return only MDI importances
        rf_feature_importance_df = pd.DataFrame({
            'feature': fi_df['feature'],
            'MDI_importance': fi_df['importance']
        })

    return rf_feature_importance_df

def ridge_regression(feature_matrix, target_column, change_column):
    """
    Train a Ridge regularized linear regression model on the feature matrix.
    Parameters:
    feature_matrix: pandas DataFrame with features as columns and ATAC-seq peaks as rows with target as log2 fold change
    target_column: column name for the target variable (default is 'logFC')
    """

    #remove inf and nan values from the feature matrix and column BAFdep
    if change_column in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=[change_column])
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).dropna()

    # Split the data into features and target
    X_df = feature_matrix.drop(columns=[target_column])
    y = feature_matrix[target_column]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    #subsample 70% of the data and run ridge regression 10 times with random seed --> save the average coefficients with standard deviation
    indices = [feature_matrix.sample(frac=0.7, random_state=3 + i).index for i in range(10)]

    coefficients = {col: [] for col in X_df.columns}

    # Loop through each set of sampled indices
    for index_list in indices:
        # Convert index_list to positional indices for numpy array X
        pos_indices = feature_matrix.index.get_indexer(index_list)
        temp_X = X[pos_indices, :]
        temp_y = y.iloc[pos_indices]
        
        # Fit ridge regression (adjust alpha as needed)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(temp_X, temp_y)
        
        # Store the coefficient for each feature
        for j, col in enumerate(X_df.columns):
            coefficients[col].append(ridge_model.coef_[j])

    #calculate the mean and standard deviation of the coefficients for each feature
    mean_coefficients = {col: np.mean(coefs) for col, coefs in coefficients.items()}
    std_coefficients  = {col: np.std(coefs)  for col, coefs in coefficients.items()}

    #sort the features by the mean coefficient in descending order
    sorted_features = sorted(mean_coefficients.items(), key=lambda x: x[1], reverse=True)

    # Print coefficients
    feature_importance_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
    feature_importance_df['std'] = feature_importance_df['feature'].map(std_coefficients)

    return feature_importance_df

def plot_ridge(feature_importance_df):
    """
    Calculate feature importance from Ridge regression coefficient df
    """
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_importance_df['feature'],
        'importance': feature_importance_df['importance'],
        'std': feature_importance_df['std']
    })

    #---------------block for plotting Ridge feature importances-------------------
    plot_df = feature_importance_df[['feature', 'importance']]
    plt.figure(figsize=(6, 20))
    # plot all values in a heatmap with a color gradient (zero as white in coolwarm)
    sns.heatmap(
        plot_df.set_index('feature'),
        annot=True,
        cmap='coolwarm',
        center=0,
        yticklabels=plot_df['feature']
    )
    plt.title('Feature Importances from Ridge Regression')
    plt.ylabel('Feature')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("../data/output/feature_importances_ridge.pdf", format='pdf')
    #--------------------------end of block--------------------------------

    return feature_importance_df
#--------------------------end of block--------------------------

def remove_features(feature_matrix, exclude_features):
    """
    Remove rows with 1 in specified columns from the feature matrix.
    Parameters:
    feature_matrix: pandas DataFrame with features as columns and ATAC-seq peaks as rows
    exclude_features: list of features to exclude from the feature matrix
    """
    #drop rows where any of the specified features are 1
    for feature in exclude_features:
        if feature in feature_matrix.columns:
            feature_matrix = feature_matrix[feature_matrix[feature] != 1]
        else:
            print(f"Feature '{feature}' not found in the feature matrix. Skipping.")

    #remove columns that need to be excluded
    feature_matrix = feature_matrix.drop(columns=exclude_features, errors='ignore')

    #reset index
    feature_matrix.reset_index(drop=True, inplace=True)
    
    return feature_matrix

def plot(feature_importance_df):
    """function for plotting"""

    #---------- block for plotting correlation between MDI importance and Ridge coefficient-------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=feature_importance_df['MDI_importance'],
        y=feature_importance_df['Ridge_coef'].abs(),
        color=sns.xkcd_rgb['dark peach']
    )

    # calculate correlation coefficient
    correlation = feature_importance_df['MDI_importance'].corr(feature_importance_df['Ridge_coef'].abs())
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.title('Correlation between MDI Importance and Ridge Coefficient')
    plt.xlabel('MDI Importance')
    plt.ylabel('Absolute Ridge Coefficient')
    plt.tight_layout()
    plt.savefig("../data/output/correlation_mdi_ridge.pdf", format='pdf')
    #--------------------------end of block--------------------------------
    return


def main():
    # Define the list of bed files containing features
    bed_path = '../data/bed/'
    bed_list = [os.path.join(bed_path, f) for f in os.listdir(bed_path) if f.endswith('.bed') or f.endswith('.bed.gz')]

    # Parse command line arguments
    args = parse_args()
    input_matrix = args.input_matrix
    diff_peaks_column = args.diff_peaks
    fc_column = args.fold_change
    exclude_features = args.exclude_features
    permutation = args.p
    threads = args.threads
    print(f"Excluding features: {exclude_features}")

    # Make feature matrix 
    feature_matrix = make_feature_matrix(bed_list, input_matrix, fc_column, diff_peaks_column)

    if exclude_features:
        feature_matrix = remove_features(feature_matrix, exclude_features)

    #random forest classifier
    rf_classifier, X_val, y_val = random_forest(feature_matrix, diff_peaks_column, fc_column, threads)
    rf_feature_importance_df = random_forest_feature_importance(rf_classifier, X_val, y_val, permutation, threads)

    #ridge regularized linear regression
    ridge_df = ridge_regression(feature_matrix , fc_column, diff_peaks_column)
    ridge_feature_importance_df = plot_ridge(ridge_df)

    #concat feature importances --> column names are 'feature', 'MDI_importance', 'Ridge_coef' and map to same feature names
    #make dictionary for ridge coef
    ridge_coef_dict = dict(zip(ridge_feature_importance_df['feature'], ridge_feature_importance_df['importance']))
    #make dictionary for Ridge standard deviations
    ridge_std_dict = dict(zip(ridge_feature_importance_df['feature'], ridge_feature_importance_df['std']))

    # map ridge coefficients and standard deviations to the feature importance df
    rf_feature_importance_df['Ridge_coef'] = rf_feature_importance_df['feature'].map(ridge_coef_dict)
    rf_feature_importance_df['Ridge_std'] = rf_feature_importance_df['feature'].map(ridge_std_dict)
    #rename importance column to MDI_importance
    rf_feature_importance_df.rename(columns={'importance': 'MDI_importance'}, inplace=True)

    # call plotting method
    plot(rf_feature_importance_df)

    #save feature importance df to csv
    rf_feature_importance_df.to_csv("../data/output/feature_importance.csv", index=False)
    
    return 


if __name__ == "__main__":
    main()