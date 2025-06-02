import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import List, Tuple, Any
import mlflow
import mlflow.sklearn

def data_preprocessing(data_dir: str) -> pd.DataFrame:
    """Preprocess the raw data for analysis and modeling.

    This function performs several preprocessing steps:
    1. Reads the CSV data
    2. Fixes data type issues in box_orders
    3. Converts week to datetime
    4. Converts boolean columns to integers
    5. Creates dummy variables for box_type

    Args:
        data_dir (str): Directory path containing the data.csv file

    Returns:
        pd.DataFrame: Preprocessed DataFrame with all necessary transformations
    """
    df = pd.read_csv(data_dir + '/data.csv')

    # replace '1O0' with '100'
    df['box_orders'] = df['box_orders'].str.replace('1O0', '100')

    # box_orders to float
    df['box_orders'] = df['box_orders'].astype(float)

    # convert week to datetime
    df['week'] = pd.to_datetime(df['week'])

    # is_marketing_week to int
    df['is_marketing_week'] = df['is_marketing_week'].astype(int)

    # holiday_week to int
    df['holiday_week'] = df['holiday_week'].astype(int)

    # get dummy variables for box_type
    box_type_dummies = pd.get_dummies(df['box_type'], prefix='boxType')

    # as int
    box_type_dummies = box_type_dummies.astype(int)

    # add box_type_dummies to data
    df = pd.concat([df, box_type_dummies], axis=1)

    return df 




def RF_forecast(
    df: pd.DataFrame,
    box_types: List[str],
    time_horizon: int = 1,
    res_dir: str = '../res'
) -> Tuple[pd.DataFrame, pd.DataFrame, float, RandomForestRegressor, List[str], List[str], List[str], List[str]]:
    """Train and evaluate a Random Forest model for box order forecasting.

    This function:
    1. Creates rolling mean features for different window sizes
    2. Splits data into train/test sets
    3. Trains a global Random Forest model
    4. Generates predictions and various evaluation plots
    5. Calculates feature importances

    Args:
        df (pd.DataFrame): Preprocessed DataFrame from data_preprocessing
        box_types (List[str]): List of box types to forecast
        time_horizon (int, optional): Number of weeks to forecast ahead. Defaults to 1.
        res_dir (str, optional): Directory to save results and plots. Defaults to '../res'.

    Returns:
        Tuple containing:
            - pd.DataFrame: Test DataFrame with predictions
            - pd.DataFrame: Train DataFrame
            - float: MAE score for global model
            - RandomForestRegressor: Trained global model
            - List[str]: List of feature names
            - List[str]: List of box type feature names
            - List[str]: List of rolling feature names
            - List[str]: List of leaky feature names
    """
    # features = all columns except columns with boxType_ in the name and week
    features = [col for col in df.columns if not col.startswith('boxType_') and col != 'week']

    # leaky features
    leaky_features = ['weekly_subscribers', 'fortnightly_subscribers', 'box_orders', 'total_box_orders']

    # target
    target = 'box_orders'


    test_df_list = [] # list to store test dataframes
    train_df_list = [] # list to store train dataframes
    mae_list = [] # list to store MAEs


    # iterate across box_types
    for box_type in box_types:
        # keep only boxType_ == box_type
        df_i = df[df['boxType_' + box_type] == 1]

        # sort by week
        df_i = df_i.sort_values(by='week')
        
        for window_size in [1, 4]:
            df_i[f'{target}_rolling_mean_{window_size}w'] = df_i[target].shift(time_horizon).rolling(window=window_size, min_periods=1).mean()
            for feature in features:
                # Ensure the feature column exists before trying to calculate rolling mean
                if feature in df_i.columns:
                    df_i[f'{feature}_rolling_mean_{window_size}w'] = df_i[feature].shift(time_horizon).rolling(window=window_size, min_periods=1).mean()

        # drop rows with NaN
        df_i = df_i.dropna()

        # train test split at 80/20
        train_df = df_i.iloc[:int(len(df_i) * 0.8)]
        test_df = df_i.iloc[int(len(df_i) * 0.8):]

        # add a dummy regressor equal to the shifted target
        test_df['dummy_regressor'] = test_df[target].shift(time_horizon)

        # drop na
        test_df = test_df.dropna()

        # store test dataframe
        test_df_list.append(test_df)

       # store train dataframe
        train_df_list.append(train_df)

    # concat train dataframes
    train_df_concat = pd.concat(train_df_list)

    # concat test dataframes
    test_df_concat = pd.concat(test_df_list)

    # box_type features
    box_type_features = train_df_concat.columns[train_df_concat.columns.str.contains('boxType_')].tolist()

    # rolling features
    rolling_features = train_df_concat.columns[train_df_concat.columns.str.contains('rolling_mean_')].tolist()

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("time_horizon", time_horizon)
        mlflow.log_param("n_estimators", 1000) # Assuming n_estimators is fixed
        mlflow.log_param("max_depth", 10) # Assuming max_depth is fixed


        # train global model
        rf_regressor_global = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=10)
        rf_regressor_global.fit(train_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features), train_df_concat[target])

        # Predict on the concatenated test dataframes
        test_df_concat['predicted_box_orders_global'] = rf_regressor_global.predict(test_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features))

        # calculate MAE for global model
        mae_global = mean_absolute_error(test_df_concat[target], test_df_concat['predicted_box_orders_global'])
        print(f'MAE for global model: {mae_global}')
        mlflow.log_metric("mae_global", mae_global)

        # plot actual vs predicted
        plt.figure()
        sns.lineplot(data=test_df_concat, x=test_df_concat.index, y=target, label='Actual')
        sns.lineplot(data=test_df_concat, x=test_df_concat.index, y='predicted_box_orders_global', label='Predicted')
        sns.lineplot(data=test_df_concat, x=test_df_concat.index, y='dummy_regressor', label='Dummy Regressor')
        plt.title('Actual vs Predicted for Global Model')
        plt.tight_layout()
        actual_vs_predicted_path = f'{res_dir}/actual_vs_predicted_{time_horizon}w.png'
        plt.savefig(actual_vs_predicted_path)
        mlflow.log_artifact(actual_vs_predicted_path)
        plt.close()

        # plot actual vs predicted for each box type separately
        for box_type in box_types:
            plt.figure()
            test_data_i = test_df_concat[test_df_concat['boxType_' + box_type] == 1]
            sns.lineplot(data=test_data_i, x=test_data_i.index, y=target, label='Actual')
            sns.lineplot(data=test_data_i, x=test_data_i.index, y='predicted_box_orders_global', label='Predicted')
            plt.title('Actual vs Predicted for ' + box_type)
            plt.tight_layout()
            actual_vs_predicted_box_type_path = f'{res_dir}/actual_vs_predicted_{time_horizon}w_{box_type}.png'
            plt.savefig(actual_vs_predicted_box_type_path)
            mlflow.log_artifact(actual_vs_predicted_box_type_path)
            plt.close()


        # feature importance
        plt.figure()
        importances = rf_regressor_global.feature_importances_
        feature_names = train_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features).columns.tolist()
        feature_importances = pd.Series(importances, index=feature_names)
        feature_importances = feature_importances.sort_values(ascending=False)
        sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
        plt.title('Feature Importances for Global Model')
        plt.tight_layout()
        feature_importances_path = f'{res_dir}/feature_importances_{time_horizon}w.png'
        plt.savefig(feature_importances_path)
        mlflow.log_artifact(feature_importances_path)
        plt.close()


        # also sns regplot of actual vs predicted global model
        plt.figure()
        sns.regplot(data=test_df_concat, x = target, y = 'predicted_box_orders_global')
        r_squared = test_df_concat[target].corr(test_df_concat["predicted_box_orders_global"])**2
        mlflow.log_metric("r_squared_global", r_squared)
        print(f'R^2 for global model: {r_squared}')
        plt.title('Actual vs Predicted for Global Model')
        plt.tight_layout()
        actual_vs_predicted_regplot_path = f'{res_dir}/actual_vs_predicted_regplot_{time_horizon}w.png'
        plt.savefig(actual_vs_predicted_regplot_path)
        mlflow.log_artifact(actual_vs_predicted_regplot_path)
        plt.close()

        # same for the dummy regressor
        plt.figure()
        sns.regplot(data=test_df_concat, x = target, y = 'dummy_regressor')
        plt.title('Actual vs Dummy Regressor for Global Model') 
        plt.tight_layout()
        actual_vs_dummy_regressor_regplot_path = f'{res_dir}/actual_vs_dummy_regressor_regplot_{time_horizon}w.png'
        plt.savefig(actual_vs_dummy_regressor_regplot_path)
        mlflow.log_artifact(actual_vs_dummy_regressor_regplot_path)
        plt.close()

        # plot distribution of predictions in histogram
        plt.figure()
        sns.histplot(data=test_df_concat, x='predicted_box_orders_global')
        plt.title('Distribution of Predictions for Global Model')
        plt.tight_layout()
        predictions_distribution_path = f'{res_dir}/predictions_distribution_{time_horizon}w.png'
        plt.savefig(predictions_distribution_path)
        mlflow.log_artifact(predictions_distribution_path)
        plt.close()

        # plot distribution of actual in histogram
        plt.figure()
        sns.histplot(data=test_df_concat, x=target)
        plt.title('Distribution of Actual for Global Model')
        plt.tight_layout()
        actual_distribution_path = f'{res_dir}/actual_distribution_{time_horizon}w.png'
        plt.savefig(actual_distribution_path)
        mlflow.log_artifact(actual_distribution_path)
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(rf_regressor_global, "random-forest-model")

    return test_df_concat, train_df_concat, mae_global, rf_regressor_global, features, box_type_features, rolling_features, leaky_features


