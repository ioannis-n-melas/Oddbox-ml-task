import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def RF_forecast(df):

    # box_types
    box_types = df['box_type'].unique()

    # drop box_type
    df.drop(columns=['box_type'], inplace=True)

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
            df_i[f'{target}_rolling_mean_{window_size}w'] = df_i[target].shift(1).rolling(window=window_size, min_periods=1).mean()
            for feature in features:
                # Ensure the feature column exists before trying to calculate rolling mean
                if feature in df_i.columns:
                    df_i[f'{feature}_rolling_mean_{window_size}w'] = df_i[feature].shift(1).rolling(window=window_size, min_periods=1).mean()

        # drop rows with NaN
        df_i = df_i.dropna()

        # train test split at 80/20
        train_df = df_i.iloc[:int(len(df_i) * 0.8)]
        test_df = df_i.iloc[int(len(df_i) * 0.8):]

        # add a dummy regressor equal to the shifted target
        test_df['dummy_regressor'] = test_df[target].shift(1)

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

    # train global model
    rf_regressor_global = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=10)
    rf_regressor_global.fit(train_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features), train_df_concat[target])

    # Predict on the concatenated test dataframes
    test_df_concat['predicted_box_orders_global'] = rf_regressor_global.predict(test_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features))

    # calculate MAE for global model
    mae_global = mean_absolute_error(test_df_concat[target], test_df_concat['predicted_box_orders_global'])
    print(f'MAE for global model: {mae_global}')

    # plot actual vs predicted
    sns.lineplot(data=test_df_concat, x=test_df_concat.index, y=target, label='Actual')
    sns.lineplot(data=test_df_concat, x=test_df_concat.index, y='predicted_box_orders_global', label='Predicted')
    sns.lineplot(data=test_df_concat, x=test_df_concat.index, y='dummy_regressor', label='Dummy Regressor')
    plt.title('Actual vs Predicted for Global Model')
    plt.tight_layout()
    plt.savefig('../res/actual_vs_predicted.png')
    plt.close()

    # plot actual vs predicted for each box type separately
    for box_type in box_types:
        test_data_i = test_df_concat[test_df_concat['boxType_' + box_type] == 1]
        sns.lineplot(data=test_data_i, x=test_data_i.index, y=target, label='Actual')
        sns.lineplot(data=test_data_i, x=test_data_i.index, y='predicted_box_orders_global', label='Predicted')
        plt.title('Actual vs Predicted for ' + box_type)
        plt.tight_layout()
        plt.savefig('../res/actual_vs_predicted_' + box_type + '.png')
        plt.close()


    # feature importance
    importances = rf_regressor_global.feature_importances_
    feature_names = train_df_concat[rolling_features + box_type_features + features].drop(columns=leaky_features).columns.tolist()
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)
    sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
    plt.title('Feature Importances for Global Model')
    plt.tight_layout()
    plt.savefig('../res/feature_importances.png')
    plt.close()


    # also sns regplot of actual vs predicted global model
    sns.regplot(data=test_df_concat, x = target, y = 'predicted_box_orders_global')
    # print r^2
    print(f'R^2 for global model: {test_df_concat[target].corr(test_df_concat["predicted_box_orders_global"])}')
    plt.title('Actual vs Predicted for Global Model')
    plt.tight_layout()
    plt.savefig('../res/actual_vs_predicted_regplot.png')
    plt.close()

    # same for the dummy regressor
    sns.regplot(data=test_df_concat, x = target, y = 'dummy_regressor')
    plt.title('Actual vs Dummy Regressor for Global Model') 
    plt.tight_layout()
    plt.savefig('../res/actual_vs_dummy_regressor_regplot.png')
    plt.close()

    # plot distribution of predictions in histogram
    sns.histplot(data=test_df_concat, x='predicted_box_orders_global')
    plt.title('Distribution of Predictions for Global Model')
    plt.tight_layout()
    plt.savefig('../res/predictions_distribution.png')
    plt.close()

    # plot distribution of actual in histogram
    sns.histplot(data=test_df_concat, x=target)
    plt.title('Distribution of Actual for Global Model')
    plt.tight_layout()
    plt.savefig('../res/actual_distribution.png')
    plt.close()

    return test_df_concat, train_df_concat, mae_global, rf_regressor_global, features, box_type_features, rolling_features, leaky_features


