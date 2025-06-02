import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# get dummy variables for box_type
box_type_dummies = pd.get_dummies(df['box_type'], prefix='boxType')

# as int
box_type_dummies = box_type_dummies.astype(int)

# add box_type_dummies to data
df = pd.concat([df, box_type_dummies], axis=1)

# plot box orders by box type
sns.lineplot(data=df, x=df.index, y='box_orders', hue='box_type')
plt.show()

# same without hue
sns.lineplot(data=df, x=df.index, y='box_orders')
plt.show()

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
plt.show()

# keep only boxType_FB == 1
df_fb = df[df['boxType_FB'] == 1]

# features = all columns except box_orders, total_box_orders, and columns with boxType_ in the name
features = [col for col in df.columns if col not in ['box_orders', 'total_box_orders'] and not col.startswith('boxType_')]

# target = box_orders
target = 'box_orders'

# time series forecasting