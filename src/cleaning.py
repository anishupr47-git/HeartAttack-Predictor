import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#First we clean the data so that the dataset we have is safe to use and optimized
#check missing value
#change text data into numbers
#scale numbers

def generate_eda_stats(filepath):
    """
    Generates EDA statistics directly from raw data state
    """
    #step 1 load data
    print("Starting EDA process")
    print(f"Reading from file: {filepath}")
    df = pd.read_excel(filepath, engine='openpyxl')

    #lets count how many rows we have
    total_rows = df.shape[0]
    print(f"Total number of rows:{total_rows}")

    #lets count how many rows we have
    total_cols = df.shape[1]
    print(f"Total number of columns:{total_cols}")

    #Step 2 check for missing value
    missing_summary = df.isnull().sum()
    print("Here are the missing value of all columns")
    print(missing_summary)

    #we check all columns individually
    missing_cols = missing_summary[missing_summary > 0].to_dict()
    print(f"Columns with at least one missing value: {missing_cols}")

    #Step 3 descriptive statistics
    #.describe() gives us the mean,min,max and percentile for numbers
    desc_stats = df.describe().to_dict()
    
    target_col = 'Heart_Disease_Diagnosis'
    print(f"Checking if the target column '{target_col}' exists")
    if target_col in df.columns:
        target_dist = df[target_col].value_counts(normalize=True).to_dict()
        print(f"Target distribution is: {target_dist}")
    else:
        print("Target Column Not Found!")
        target_dist = {}

    print("EDA stats generation complete. Returning dictionary of stats")
    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_columns': missing_cols,
        'descriptive_stats': desc_stats,
        'target_distribution': target_dist,
        'raw_df': df    
    }

def absolute_preprocessing_pipeline(filepath):
    """
    Reads the excel file and starts step by step preprocessing
    """

    #Phase 1 loading
    print("Starting Absolute Preprocessing Pipeline")
    print("Step 1: Load data into pandas")
    df = pd.read_excel(filepath, engine='openpyxl')

    #Phase 2 drop unnecessary colums
    #patient id is not so necessary we drop that
    print("Check for Patient ID")
    if 'Patient_ID' in df.columns:
        print("Found Patient_ID dropping it for useful learning")
        df = df.drop(columns=['Patient_ID'])
    else:
        print("'Patient_ID' not found")

    #Phase 3 checking each column type and fill in missing value
    print("Step 3 looping though columns to check types and fill missing value")

    for col in df.columns:
        print(f" Examining Column: {col}")
        col_type=df[col].dtype
        print(f" Type is: {col_type}")

        #check for missing value
        missing_count = df[col].isnull().sum()
        print(f"Number of missing values: {missing_count}")

        if col_type in ['int64','float64']:
            print("This is numerical column")
            #calculate median
            median_val = df[col].median()
            print(f"Calculated Median: {median_val}")

            if missing_count > 0:
                print("Filling missing value with median")
                df[col] = df[col].fillna(median_val)
            else:
                print("No missing values")
        else:
            print("This is categorical column")
            #calculate mode.mode()
            mode_val=df[col].mode()[0]
            print(f"Calculated Mode: {mode_val}")

            if missing_count > 0:
                print("Filling missing value with mode")
                df[col] = df[col].fillna(mode_val)
            else:
                print("No missing Value")

    #Phase 4: Seperating features (x) from (y) for training data
    print("Step 4 Seperating the target variable from the features")
    target_col = 'Heart_Disease_Diagnosis'

    #We must make sure that target column exists
    if target_col not in df.columns:
        print("ERROR! Target column not found")
        raise ValueError(f"Target columns '{target_col}' not found in dataset")
    
    print(f"Extracting '{target_col}' as our 'y' variable")
    y=df[target_col].values

    print("Dropping target from Dataframe to create our x feature")
    X_df = df.drop(columns=[target_col])

    #Phase 5 Identifying column types for ui
    print("Step 5 Identifying categorical and numerical columns for the UI")
    cat_cols = X_df.select_dtypes(include=['object','category']).columns.to_list()
    num_cols = X_df.select_dtypes(include=['int64','float64']).columns.to_list()

    print(f"Categorical columns found: {len(cat_cols)}")
    print(cat_cols)
    print(f"Numerical columns found: {len(num_cols)}")
    print(num_cols)
    #build dictionary ui config
    print("Building Dictonary")
    ui_config = {}

    #first configure numerical columns
    print("Configuring numerical columns")
    for col in num_cols:
        print(f"{col}")

        #we need the min and max value
        # a num too high or low
        min_val = float(X_df[col].min())
        max_val = float(X_df[col].max())
        mean_val = float(X_df[col].mean())

        #User requirement
        if col == 'Age':
            print("Number should fall under min and max value")
            min_val = 1.0
            max_val = 150.0
        else:
            min_val = 0.0
            max_val = max_val + 50.0

        ui_config[col] = {
            'type': 'continious',
            'min': min_val,
            'max': max_val,
            'mean': mean_val
        }

    #second, configure categorical columns
    print("Configuring categorical columns")
    for col in cat_cols:
        print(f"{col}")

        #we need the list of all unique options
        option_list= X_df[col].unique().tolist()
        mode_val= X_df[col].mode()[0]

        ui_config[col] = {
            'type': 'categorical',
            'options': option_list,
            'mode': mode_val
        }

    #Phase 6 one hot encoding
    #computer cannot understand characters 
    #so we change them into computer understanding code
    print("Step 6 Performing One hot encoding on categorical columns")
    X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    
    #We save the names of the final columns after encoding
    feature_names = X_encoded.columns.to_list()
    print(f"Total features after encoding: {len(feature_names)}")

    #Phase 7: Scaling
    print("Step 7 Applying StandardScaler to normalize all numerical values")
    scaler = StandardScaler()

    #.fit_transform does
    #learns the mean and does SD
    #changes the number ACTUALLY
    X_scaled = scaler.fit_transform(X_encoded.values)

    print("Absolute Preprocessing Complete!")
    
    #return all now
    return X_scaled, y, scaler, feature_names, ui_config, cat_cols





