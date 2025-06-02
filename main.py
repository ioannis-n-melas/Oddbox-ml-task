import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prophet_forecast(df_fb, target, features):
    # Prepare data for Prophet
    # We'll use df_fb which is already filtered for boxType_FB
    df_prophet = df_fb[['week', 'box_orders']].copy()
    df_prophet.rename(columns={'week': 'ds', 'box_orders': 'y'}, inplace=True)

    # Identify regressors to add to Prophet model
    # These are columns from the 'features' list that are present in df_fb (excluding 'week' which is 'ds')
    prophet_regressors = []
    if 'features' in globals() and isinstance(features, list):
        for col_name in features:
            if col_name in df_fb.columns and col_name != 'week': # 'week' is already ds
                prophet_regressors.append(col_name)
                # Add regressor column to df_prophet from df_fb, aligning by 'ds' (which was 'week')
                # Ensure correct alignment if df_fb isn't sorted or has missing weeks for some reason.
                # A merge is safer.
                temp_df = df_fb[['week', col_name]].copy()
                temp_df.rename(columns={'week': 'ds'}, inplace=True)
                df_prophet = pd.merge(df_prophet, temp_df, on='ds', how='left')

    # --- Train-Test Split and Evaluation ---
    print("\n--- Train-Test Split Evaluation ---")
    if not df_prophet.empty and len(df_prophet) > 10: # Ensure enough data for a split (e.g., >10 points)
        # Define split point (e.g., 80% train, 20% test)
        n_total = len(df_prophet)
        n_train = int(n_total * 0.8)
        
        train_df = df_prophet.iloc[:n_train].copy()
        test_df = df_prophet.iloc[n_train:].copy()

        print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples.")

        # Clean training data (handle NaNs in regressors)
        if prophet_regressors: # Make sure prophet_regressors is defined
            train_df.dropna(subset=prophet_regressors, inplace=True)
        
        if not train_df.empty and not train_df[['ds', 'y']].isnull().values.any():
            model_eval = Prophet()
            for regressor in prophet_regressors:
                model_eval.add_regressor(regressor)
            
            try:
                model_eval.fit(train_df[['ds', 'y'] + prophet_regressors])

                future_test = test_df[['ds'] + prophet_regressors].copy()
                
                for regressor in prophet_regressors:
                    if future_test[regressor].isnull().any():
                        print(f"Warning: NaN found in regressor '{regressor}' in test set. Filling with 0 for prediction.")
                        future_test[regressor] = future_test[regressor].fillna(0) # Addressed FutureWarning

                forecast_test = model_eval.predict(future_test)
                eval_df = pd.merge(test_df[['ds', 'y']], forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

                if not eval_df.empty:
                    mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
                    rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))
                    print(f"Test Set MAE: {mae:.2f}")
                    print(f"Test Set RMSE: {rmse:.2f}")

                    plt.figure(figsize=(12, 6))
                    plt.plot(eval_df['ds'], eval_df['y'], label='Actual', marker='.')
                    plt.plot(eval_df['ds'], eval_df['yhat'], label='Predicted', marker='.')
                    plt.fill_between(eval_df['ds'], eval_df['yhat_lower'], eval_df['yhat_upper'], alpha=0.2, label='Confidence Interval')
                    plt.title('Prophet Forecast vs Actuals on Test Set')
                    plt.xlabel('Date')
                    plt.ylabel('Box Orders')
                    plt.legend()
                    plt.savefig('test_set_actuals_vs_predicted.png')
                    plt.close()

                    fig_comp_eval = model_eval.plot_components(forecast_test)
                    fig_comp_eval.savefig('test_set_components_plot.png')
                    plt.close(fig_comp_eval)
                else:
                    print("Evaluation dataframe is empty after merging predictions. Check test data and forecast.")
            except Exception as e:
                print(f"Error during model evaluation fitting or prediction: {e}")
        else:
            print("Training data is empty or contains NaNs in 'ds' or 'y' after processing. Skipping train-test evaluation.")
    else:
        print("df_prophet is empty or too small for a meaningful train-test split. Skipping evaluation.")
    # --- END: Train-Test Split and Evaluation ---


    # --- Full Data Training and Future Forecasting ---
    print("\n--- Full Data Training and Future Forecasting ---")
    df_prophet_full_train = df_prophet.copy()

    if prophet_regressors:
        df_prophet_full_train.dropna(subset=prophet_regressors, inplace=True)
    df_prophet_full_train.dropna(subset=['ds', 'y'], inplace=True)

    if not df_prophet_full_train.empty:
        model_future_forecast = Prophet()
        for regressor in prophet_regressors:
            model_future_forecast.add_regressor(regressor)
        
        try:
            model_future_forecast.fit(df_prophet_full_train[['ds', 'y'] + prophet_regressors])
            future_actual_forecast = model_future_forecast.make_future_dataframe(periods=52, freq='W')

            if prophet_regressors:
                future_actual_forecast = pd.merge(future_actual_forecast, df_prophet_full_train[['ds'] + prophet_regressors], on='ds', how='left')
                for regressor in prophet_regressors:
                    if regressor in ['is_marketing_week', 'holiday_week']:
                        future_actual_forecast[regressor] = future_actual_forecast[regressor].fillna(0) # Addressed FutureWarning
                    else:
                        future_actual_forecast[regressor] = future_actual_forecast[regressor].ffill() # Addressed FutureWarning
                        future_actual_forecast[regressor] = future_actual_forecast[regressor].fillna(0) # Addressed FutureWarning

            forecast_actual_future = model_future_forecast.predict(future_actual_forecast)

            fig1_future = model_future_forecast.plot(forecast_actual_future)
            plt.title('Forecast of Box Orders (FB) with Prophet (Full Data + Future)')
            plt.xlabel('Date')
            plt.ylabel('Box Orders')
            fig1_future.savefig('full_data_forecast_plot.png')
            plt.close(fig1_future)

            fig2_future = model_future_forecast.plot_components(forecast_actual_future)
            fig2_future.savefig('full_data_components_plot.png')
            plt.close(fig2_future)

            print("Forecast head (historical fit on full data):")
            print(forecast_actual_future[forecast_actual_future['ds'].isin(df_prophet_full_train['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

            print("\nForecast tail (future predictions - 52 weeks):")
            print(forecast_actual_future[forecast_actual_future['ds'] > df_prophet_full_train['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
            print("\nLast few future predictions (52 weeks):")
            print(forecast_actual_future[forecast_actual_future['ds'] > df_prophet_full_train['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        except Exception as e:
            print(f"Error during full data model fitting or future forecasting: {e}")
    else:
        print("Full dataset for Prophet (df_prophet_full_train) is empty after NaN removal. Skipping future forecast.")

# Load the data
df = pd.read_csv('data.csv')

# Display the first few rows of the data
print(df.head())

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

# nunique of box_type
print(df['box_type'].nunique())

# assign to var
box_types = df['box_type'].unique()

# get dummy variables for box_type
box_type_dummies = pd.get_dummies(df['box_type'], prefix='boxType')

# as int
box_type_dummies = box_type_dummies.astype(int)

# add box_type_dummies to data
df = pd.concat([df, box_type_dummies], axis=1)

# plot box orders by box type
sns.lineplot(data=df, x=df.index, y='box_orders', hue='box_type')
plt.savefig('box_orders_by_type.png')
plt.close()

# same without hue
sns.lineplot(data=df, x=df.index, y='box_orders')
plt.savefig('box_orders_overall.png')
plt.close()

# drop box_type
df.drop(columns=['box_type'], inplace=True)

# calculate total number of box orders each week
total_box_orders = df.groupby('week')['box_orders'].sum().reset_index()
# rename box_orders to total_box_orders
total_box_orders.rename(columns={'box_orders': 'total_box_orders'}, inplace=True)

# merge total_box_orders with data
df = pd.merge(df, total_box_orders, on='week', how='left')

# plot total_box_orders
sns.lineplot(data=df, x=df.index, y='total_box_orders')
plt.savefig('total_box_orders_overall.png')
plt.close()

# # iterate across box_types
# for box_type in box_types:
#     # keep only boxType_FB == 1
#     df_fb = df[df['boxType_FB'] == 1]

#     # features = all columns except box_orders, total_box_orders, and columns with boxType_ in the name
#     features = [col for col in df.columns if col not in ['box_orders', 'total_box_orders'] and not col.startswith('boxType_')]


# keep only boxType_FB == 1
df_fb = df[df['boxType_FB'] == 1]

# features = all columns except box_orders, total_box_orders, and columns with boxType_ in the name
features = [col for col in df.columns if col not in ['box_orders', 'total_box_orders'] and not col.startswith('boxType_')]

# target = box_orders
target = 'box_orders'

# time series forecasting
# sort by week
df_fb = df_fb.sort_values(by='week')

# calculate rolling average of all features and target 1, 2, 4, 6, 8, 12 weeks ago without the current week
# window_size = 2
for window_size in [1, 2, 4, 6, 8, 12]:
    df_fb[f'{target}_rolling_mean_{window_size}w'] = df_fb[target].shift(1).rolling(window=window_size, min_periods=1).mean()
    for feature in [x for x in features if x != 'week']:
        # Ensure the feature column exists before trying to calculate rolling mean
        if feature in df_fb.columns:
            df_fb[f'{feature}_rolling_mean_{window_size}w'] = df_fb[feature].shift(1).rolling(window=window_size, min_periods=1).mean()

# drop rows with NaN
df_fb = df_fb.dropna()

# leaky features
leaky_features = ['weekly_subscribers', 'fortnightly_subscribers']

# rolling features
rolling_features = df_fb.columns[df_fb.columns.str.contains('rolling_mean_')].tolist()

# train test split at 80/20
train_df = df_fb.iloc[:int(len(df_fb) * 0.8)]
test_df = df_fb.iloc[int(len(df_fb) * 0.8):]

# train Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=5)
rf_regressor.fit(train_df[rolling_features + features].drop(columns=leaky_features).drop(columns='week'), train_df[target])

# predict on test set
test_df['predicted_box_orders'] = rf_regressor.predict(test_df[rolling_features + features].drop(columns=leaky_features).drop(columns='week'))

# calculate MAE and RMSE
import sklearn.metrics as metrics
mae = metrics.mean_absolute_error(test_df[target], test_df['predicted_box_orders'])
rmse = np.sqrt(metrics.mean_squared_error(test_df[target], test_df['predicted_box_orders']))
print(f'MAE: {mae}, RMSE: {rmse}')

# plot feature importance
importances = rf_regressor.feature_importances_
feature_names = train_df[rolling_features + features].drop(columns=leaky_features).drop(columns='week').columns.tolist()
feature_importances = pd.Series(importances, index=feature_names)

# sort by importance
feature_importances = feature_importances.sort_values(ascending=False)

# sns barplot of feature importance
sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature Name')
plt.savefig('feature_importances.png')

# plot actual vs predicted
sns.lineplot(data=test_df, x=test_df.index, y=target, label='Actual')
sns.lineplot(data=test_df, x=test_df.index, y='predicted_box_orders', label='Predicted')
plt.savefig('actual_vs_predicted.png')
