# Box Orders Forecasting Model

This project implements a machine learning model to forecast box orders for different box types. The model uses a Random Forest Regressor to predict future box orders based on historical data and various features, with support for multiple prediction horizons (1-week and 4-week forecasts).

## Project Overview

### Prediction Task
The model aims to forecast the number of orders for different types of produce boxes (e.g., Large Veg, Medium Fruit & Veg, etc.) for both 1-week and 4-week horizons. This forecasting helps in inventory management, supply chain planning, and business operations optimization.

### Data Characteristics
The dataset contains historical order data with the following key components:
- Weekly order volumes for different box types (coded as LV, LFV, MV, MFV, SV, SFV, XSFV, FB)
- Number of subscribers (weekly, fortnightly)
- Temporal information (dates, weeks)
- Business context (marketing weeks, holidays)

#### Data visualization

![Box Orders by Type](res/box_orders_by_type.png)
![Box Orders Overall](res/box_orders_overall.png)
![Total Box Orders](res/total_box_orders_overall.png)


### Overview of forecasting methods

There are broadly two approaches to forecasting:

1.  **Time Series Modeling**: This approach uses models like Prophet to capture the temporal dynamics of time series data. These models excel in problems with strong seasonal components, such as forecasting flight demand and pricing.

2.  **Machine Learning (ML) Based Approaches**: This involves using algorithms like Random Forest (RF) or XGBoost. Time series characteristics are engineered as features, such as rolling means over different time windows. ML methods are generally better at leveraging diverse features but can be weaker in modeling pure time dynamics. For instance, to predict multiple steps into the future, separate models often need to be trained for each prediction horizon, whereas models like Prophet can be simulated autoregressively.

In this project, we initially experimented with Prophet, but it did not perform well. This is likely due to the significant stochastic component inherent in box order forecasting.

When employing an ML model for problems with multiple categories (like different box types), there are two main strategies:

*   **Separate Models**: Train a distinct model for each category.
*   **Global Model**: Train a single model across all categories, using the category itself as a feature.

A global model can be advantageous when data is scarce for individual categories, as it allows the model to learn from shared patterns across categories, akin to transfer learning.

We tested both approaches, and the global model yielded superior performance for this task.

### Approach
Our approach combines traditional time series analysis with machine learning:
1. Data preprocessing and feature engineering to capture temporal patterns and business context
2. Training separate models for 1-week and 4-week predictions
3. Box-specific analysis to understand individual box type behaviors
4. Comprehensive evaluation using multiple metrics and visualizations

## Project Structure

```
.
├── data/
│   └── data.csv
├── src/
│   ├── main.py      # Main execution script
│   └── lib.py       # Core functionality and utilities
├── res/
│   ├── box_orders_by_type.png
│   ├── box_orders_overall.png
│   ├── total_box_orders_overall.png
│   ├── [1-week horizon results]
│   │   ├── actual_vs_predicted_1w.png
│   │   ├── actual_vs_predicted_1w_*.png
│   │   ├── feature_importances_1w.png
│   │   └── ...
│   └── [4-week horizon results]
│       ├── actual_vs_predicted_4w.png
│       ├── actual_vs_predicted_4w_*.png
│       ├── feature_importances_4w.png
│       └── ...
└── README.md
```

## Getting Started

### Environment Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Model

To run the forecasting model:
```bash
python src/main.py
```

## Model Details

### Data Preprocessing
- Converts box orders from string to float
- Transforms week column to datetime

### Feature Engineering
- Creates dummy variables for categorical features (i.e. box type)
- Generates time-based features:
  - Rolling means (1-week and 4-week windows)
  - Lag features for historical patterns
- Adds business context features:
  - Marketing week indicators
  - Holiday week indicators

### Model Architecture
The model uses a Random Forest Regressor with the following specifications:
- Number of estimators: 1000
- Maximum depth: 10
- Random state: 42

### Features
The model uses several types of features:
1. Box type indicators (one-hot encoded)
2. Rolling mean features (1-week and 4-week windows)
3. Marketing and holiday indicators
4. Historical order data (i.e. previous week's orders)

### Training Process
- Data is split into training (80%) and testing (20%) sets
- A global model is trained on all box types. The box type is used as a feature.
- Performance is evaluated using Mean Absolute Error (MAE)
- Models are trained for both 1-week and 4-week prediction horizons

### Model Assumptions

The Random Forest-based forecasting model operates under several key assumptions:

1.  **Stability of Relationships**: The model assumes that the relationships between the engineered features (like rolling means, marketing indicators, and historical orders) and the box order volumes observed in the training data will remain relatively stable and continue to hold true for the prediction period. Significant shifts in underlying customer behavior or market dynamics not captured by the features could reduce accuracy.
2.  **Feature Sufficiency**: It's assumed that the features engineered (e.g., rolling means, lag features, holiday indicators, box types) are comprehensive enough to capture the primary drivers of box order fluctuations. If there are other significant external factors not included as features (e.g., competitor activities, sudden economic changes), the model might not account for their impact.
3.  **Effectiveness of Global Modeling**: By using a single global model with 'box type' as a feature, the approach assumes that there are learnable common patterns and relationships across different box types. It also assumes that the model can effectively differentiate and adapt its predictions for individual box types based on this feature.

## Model Evaluation

The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Correlation between actuals and predicted values
- Comparison with a dummy regressor (previous week's orders)

### Performance Monitoring and Drift Detection

Once deployed, continuous monitoring is crucial to ensure the model's predictive accuracy remains high and to detect any degradation in performance over time. This degradation can occur due to several reasons, broadly categorized as model drift.

**Types of Drift:**

1.  **Concept Drift**: This occurs when the statistical properties of the target variable (box orders) change over time, or the underlying relationship between input features and the target variable changes. For example, shifts in customer preferences, new market trends, or long-term effects of business strategies might lead to concept drift. The model, trained on historical data, may no longer accurately reflect these new relationships.
2.  **Data Drift**: This refers to changes in the statistical properties of the input features themselves. For instance, a significant change in the average number of weekly subscribers, the introduction of new marketing campaigns not seen during training, or changes in how box types are categorized could constitute data drift. If the live data significantly deviates from the training data distribution, the model's performance can suffer.

**Monitoring Strategies:**

To detect and address these types of drift, the following strategies should be implemented:

*   **Track Key Performance Metrics**: Regularly re-evaluate the model on new, incoming data using the same metrics established during the evaluation phase (e.g., Mean Absolute Error, correlation). A significant decline in these metrics compared to the baseline is a strong indicator of drift.
*   **Monitor Feature Distributions**: Continuously track the statistical distributions (e.g., mean, median, variance, presence of categories) of the key input features used by the model. Compare these live distributions against the distributions observed in the training dataset. Statistical tests like the Kolmogorov-Smirnov test (for numerical features) or Chi-Squared test (for categorical features) can help quantify the extent of drift.
*   **Monitor Target Variable Distribution**: Observe the distribution of the actual box orders over time. Unexpected shifts in this distribution can indicate that the underlying process generating the orders is changing.
*   **Monitor Prediction Distribution**: Analyze the distribution of the model's predictions on new data. If the model starts predicting values that are systematically different (e.g., consistently higher or lower, or with a different spread) than observed during training and validation, it might be a sign of drift or that the model is encountering out-of-distribution data.
*   **Set Up Alerting Mechanisms**: Implement automated alerts that trigger when performance metrics drop below predefined thresholds or when significant statistical drift is detected in features or predictions.

**Addressing Drift:**

If significant drift is detected, it typically necessitates retraining the model with more recent data that reflects the current state of the environment. The frequency of retraining might be periodic (e.g., every quarter) or event-triggered (i.e., when drift is detected).

## Results

The model generates several visualizations to evaluate its performance for both 1-week and 4-week prediction horizons:


### 1-Week Horizon Predictions
#### Actuals vs Predicted vs Dummy Regressor
![1-Week Actuals vs Predicted](res/actual_vs_predicted_1w.png)

*This plot compares the actual box orders (ground truth) with the predictions made by the Random Forest model and a simple dummy regressor (which predicts the previous week's orders) for the 1-week forecast horizon. It helps visualize the model's accuracy against a baseline.*

#### Regression Plot: Actual vs. Predicted
![1-Week Regplot](res/actual_vs_predicted_regplot_1w.png)

*This regression plot visualizes the relationship between the actual values and the values predicted by the ML model for the 1-week horizon. A tighter clustering of points around the diagonal line (indicating perfect prediction) signifies better model performance. When compared to the dummy regressor (which simply predicts the previous week's orders), the ML model's predictions align more closely with the actuals, demonstrating significantly lower error and higher accuracy.*

#### Regression Plot: Actual vs. Dummy Regressor
![1-Week Regplot](res/actual_vs_dummy_regressor_regplot_1w.png)



#### Feature Importances
![1-Week Feature Importances](res/feature_importances_1w.png)

*This chart displays the relative importance of each feature used by the Random Forest model for the 1-week predictions. It helps understand which factors (e.g., rolling means, specific box types, marketing weeks) have the most influence on the forecast.*

#### Predictions Distribution
![1-Week Predictions Distribution](res/predictions_distribution_1w.png)

*This histogram shows the distribution of the predicted box order values from the ML model for the 1-week horizon. It helps to understand the range and frequency of the model's forecasts.*

#### Actuals Distribution
![1-Week Actuals Distribution](res/actual_distribution_1w.png)

*This histogram displays the distribution of the actual box order values in the test set for the 1-week horizon. Comparing this to the predictions distribution can reveal if the model captures the general characteristics of the true data distribution.*

### 4-Week Horizon Predictions
#### Actuals vs Predicted vs Dummy Regressor
![4-Week Actuals vs Predicted](res/actual_vs_predicted_4w.png)

*Similar to the 1-week horizon, this plot compares actual box orders with predictions from the Random Forest model and the dummy regressor, but for the 4-week forecast horizon. It allows for an assessment of the model's performance over a longer prediction window.*

#### Regression Plot: Actual vs. Predicted
![4-Week Regplot](res/actual_vs_predicted_regplot_4w.png)

*Similarly, for the 4-week horizon, this regression plot compares the ML model's predictions against the actual values. The ML model again shows a stronger correlation and less scatter compared to what would be observed with the dummy regressor, indicating its superior ability to forecast orders even further into the future. The dummy regressor's performance would typically degrade more sharply over a longer horizon, making the ML model's advantage even more pronounced.*

#### Regression Plot: Actual vs. Dummy Regressor
![4-Week Regplot](res/actual_vs_dummy_regressor_regplot_4w.png)

#### Feature Importances
![4-Week Feature Importances](res/feature_importances_4w.png)

*This chart shows the feature importances for the Random Forest model trained for the 4-week prediction horizon. It can be compared with the 1-week feature importances to see if the influential factors change with the forecast duration.*

#### Predictions Distribution
![4-Week Predictions Distribution](res/predictions_distribution_4w.png)

*This histogram illustrates the distribution of the predicted box order values from the ML model for the 4-week horizon, providing insight into the model's forecast characteristics at this longer range.*

#### Actuals Distribution
![4-Week Actuals Distribution](res/actual_distribution_4w.png)

*This histogram shows the distribution of actual box order values in the test set for the 4-week horizon, serving as a reference to evaluate the realism of the model's prediction distribution.*


## Future Work

- Analyze box composition: Investigate the ingredients of each box to understand if similarities in composition influence customer choices or switching patterns. For example, do customers switch between boxes that share common ingredients or have similar sizes?
- Individual subscriber data: Incorporate data on individual subscriber behavior to model how users transition between different box types. This could reveal patterns related to preferences, seasonality, or responses to promotions, leading to more accurate and personalized forecasts.

## Stretch Goals

- At Oddbox, customers can swap produce items in their box with one of N alternatives. How would you approach incorporating this customisation logic into your demand forecast?

I would engineer features to capture the swap activity. These features would then be used in similar fashion to all other features of the model. Additionally, I would consider using the box composition alongside the box type and swap activity to capture the effect of the swap on the box demand.



## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## MLflow Integration for Experiment Tracking

This project uses [MLflow](https://mlflow.org/) to log and track machine learning experiments. This allows for versioning of models, parameters, metrics, and artifacts.

### Setup

1.  **Install MLflow**:
    If you haven't already, install MLflow. It's included in the `requirements.txt` file, so if you've set up your environment using that, you should be good to go. Otherwise, you can install it via pip:
    ```bash
    pip install mlflow
    ```

### Running Experiments

1.  **Execute the main script**:
    Run the main training script located in the `src` directory:
    ```bash
    python src/main.py
    ```
    The script is configured to automatically log parameters (like `time_horizon`), metrics (like MAE and R-squared), the trained RandomForest model, and generated plots as artifacts to MLflow.

2.  **MLflow Data Storage**:
    MLflow will store all experiment data locally within an `mlruns` directory. In this project, this directory will be created at `src/mlruns/` when you first run the script.
    This `src/mlruns/` directory is included in the `.gitignore` file, so it will not be committed to the Git repository.

### Viewing Results with MLflow UI

1.  **Navigate to the correct directory**:
    Open your terminal and change to the `src` directory, as this is where the `mlruns` folder is located:
    ```bash
    cd src
    ```

2.  **Start the MLflow UI**:
    From within the `src` directory, run the following command:
    ```bash
    mlflow ui
    ```
    This will start a local web server (usually at `http://localhost:5000`).

3.  **Open in Browser**:
    Open the URL provided by the `mlflow ui` command in your web browser to view the experiment runs, compare parameters and metrics, and examine the logged artifacts (models, plots, etc.).

Each execution of `src/main.py` will create a new run within MLflow, allowing you to track changes and performance over time.

