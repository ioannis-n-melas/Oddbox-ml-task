import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lib
import importlib

importlib.reload(lib)

pd.set_option('display.max_columns', 40)


if __name__ == '__main__':

    res_dir = '../res'
    data_dir = '../data'

    # Load the data
    df = lib.data_preprocessing(data_dir)

    # get box_types
    box_types = df['box_type'].unique()

    # plot box orders by box type
    sns.lineplot(data=df, x=df.index, y='box_orders', hue='box_type')
    plt.savefig('../res/box_orders_by_type.png')
    plt.close()

    # drop box_type
    df.drop(columns=['box_type'], inplace=True)

    # same without hue
    sns.lineplot(data=df, x=df.index, y='box_orders')
    plt.savefig('../res/box_orders_overall.png')
    plt.close()

    # calculate total number of box orders each week
    total_box_orders = df.groupby('week')['box_orders'].sum().reset_index()
    # rename box_orders to total_box_orders
    total_box_orders.rename(columns={'box_orders': 'total_box_orders'}, inplace=True)

    # merge total_box_orders with data
    df = pd.merge(df, total_box_orders, on='week', how='left')

    # plot total_box_orders
    sns.lineplot(data=df, x=df.index, y='total_box_orders')
    plt.savefig('../res/total_box_orders_overall.png')
    plt.close()

    test_df_concat, train_df_concat, mae_global, rf_regressor_global, features, box_type_features, rolling_features, leaky_features = lib.RF_forecast(df, box_types, time_horizon=1, res_dir=res_dir)
    test_df_concat, train_df_concat, mae_global, rf_regressor_global, features, box_type_features, rolling_features, leaky_features = lib.RF_forecast(df, box_types, time_horizon=4, res_dir=res_dir)

