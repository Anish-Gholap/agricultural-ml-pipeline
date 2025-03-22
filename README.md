# AgroTech Innovations - Farm Environment Prediction

## Anish Gholap (AIIP5 Technical Assessment)

Email: anishgholap@gmail.com

## Project Overview

This project develops an end-to-end machine learning pipeline for AgroTech Innovations to improve crop management in controlled farming environments. The pipeline addresses two key challenges:

1. **Temperature Prediction**: Developing regression models to predict temperature conditions within the farm environment
2. **Plant Type-Stage Classification**: Creating classification models to categorize plant types and growth stages based on sensor data

The models help AgroTech Innovations optimize resource allocation, improve crop management, and increase yield predictability.

## Folder Structure

```
.
├── src/
│   ├── base_model.py
│   ├── config.yaml
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── gb_model.py
│   ├── label_mappings.py
│   ├── rf_model.py
│   ├── rf_reg_model.py
│   ├── svc_model.py
│   ├── svr_model.py
│   ├── train.py
│   ├── xgb_reg_model.py_
├── .gitignore
├── eda.ipynb
├── README.md
├── requirements.txt
└── run.sh
```

## Instructions for Executing the Pipeline

1. Clone the repository
2. Place the `agri.db` file in the `data/` folder (not included in repository)
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```
   ./run.sh
   ```

The pipeline will:
1. Load data from the SQLite database
2. Preprocess the data
3. Train and evaluate both regression and classification models
4. Save the best models to the `models/` directory

## Pipeline Design Overview

The pipeline follows an object-oriented approach with a focus on modularity and reusability:

1. **Base Model Class**: Provides common functionality for all models
2. **Feature Engineering**: Custom transformers for feature creation and preprocessing
3. **Model-Specific Classes**: Specialized implementations for Random Forest and XGBoost algorithms
4. **Configuration**: YAML-based configuration for hyperparameters and experimental settings
5. **Training & Evaluation**: Separate modules for model training and performance assessment

This design enables easy experimentation with various algorithms and parameters through the configuration file.

## Key Findings from EDA

The exploratory data analysis revealed several important insights that informed the pipeline design:

### Data Quality Issues
- Nutrient sensor readings of wrong dtype
- Negative values in temperature, Light intensity and Electrical Conductivity sensors which is not possible
- Inconsistent formatting in plant type and stage labels
- Missing values in many features. Humidity Sensor was predominantly null

### Feature Relationships
- Strong correlation between light intensity and temperature
- pH levels showing distinct clusters for different plant types
- CO2 and O2 levels exhibiting inverse relationships

### Feature Engineering Opportunities
- NPK ratios provide valuable information about nutrient balance
- Light-to-CO2 ratio captures important growing conditions
- pH category clustering reveals optimal ranges for different plant types
- Light-to-Temp ratio to capture the griowing conditions for different plant types

The detailed findings can be found in `eda.ipynb`.

## Feature Processing Summary

| Feature | Transformation |
|---------|----------------|
| Temperature Sensor (°C) | Filtered negative values |
| Light Sensor | Removed negative values |
| Conductivity (EC) | Removed negative values |
| Nutrients (N, P, K) | Removed negative values, fixed data type inconsistencies |
| Humidity | Removed from dataset |
| Light & CO2 | Created Light:CO2 ratio feature |
| Light & Temperature | Created light_temp feature |
| Nutrients (N, P, K) | Created NPK ratio features |
| pH Sensor | Clustered into categorical ranges |
| Plant Type & Plant Stage | Fixed label inconsistencies |
| System Location & Plant Type | One-hot encoded |
| Nutrient & Water Sensors | Imputed missing values |
| Numerical Features | Standard scaling |

## Model Selection and Evaluation

### Regression Task (Temperature Prediction)

**Why I chose these models:**

I selected Random Forest and XGBoost regressors for temperature prediction because:

- **Random Forest** works great for this data because it handles the non-linear relationships between sensors without needing complicated feature engineering. It's also not heavily affected by outliers from faulty sensors, which is common in farm environments. The built-in feature importance helps understand which sensors matter most for temperature prediction.

- **XGBoost** tends to perform really well on tabular data like our sensor readings. It uses gradient boosting to sequentially improve predictions, which helps capture the complex patterns in temperature variations. The regularization options also help prevent overfitting to noise in the sensor data.

- I tried **SVR** too, but it was way slower to train and didn't perform as well as the tree-based models. The computational efficiency matters since we need to make predictions in near real-time for the farm environment.

### Classification Task (Plant Type-Stage)

For classifying plant types and stages, I went with Random Forest and Gradient Boosting classifiers:

- **Random Forest** handles the class imbalance in our dataset pretty naturally. Some plant stages have fewer examples than others, and RF deals with this well. It also gives us feature importance, so we can tell which environmental factors best indicate different plant types and stages.

- **Gradient Boosting** is really good at finding subtle patterns that distinguish between similar plant types or stages. It focuses on correcting misclassifications from previous decision trees, which helps it perform well on the harder examples in our dataset.

- **SVM** was tested but performed significantly worse. The multi-class nature of our problem (multiple plant types each with multiple stages) created complexity that the ensemble methods handled better.

## Model Evaluation

For evaluating the models, I used standard performance metrics appropriate for each task:

### Regression Task (Temperature Prediction)
For evaluating the temperature prediction models, I focused on multiple error metrics:

- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of errors
- **MAE (Mean Absolute Error)**: Measures the average absolute errors 
- **R² Score**: Indicates how well the model explains the variance in the data

Results:
- **Random Forest**: 
  - RMSE = 0.9073
  - MAE = 0.6717
  - R² = 0.6809

- **XGBoost**: 
  - RMSE = 0.9438
  - MAE = 0.7319
  - R² = 0.6547

The Random Forest regressor performed slightly better across all metrics, showing lower error rates and better explanatory power for temperature prediction.

### Classification Task (Plant Type-Stage)
I used the classification report which provides precision, recall, and F1-score for each class, with overall accuracy as the primary metric. Both models performed equally well:

- **Random Forest**: 86% accuracy
- **Gradient Boosting**: 86% accuracy

Looking at the classification reports, both models showed similar patterns:
- Perfect performance (1.00 precision and recall) on certain plant type-stages (classes 1, 4, 7, and 10)
- Strong performance (0.85-0.94 precision/recall) on several classes (0, 2)
- Moderate performance (0.65-0.80 precision/recall) on the more challenging classes (3, 5, 6, 8, 9, 11)

While both models achieved identical overall accuracy, their performance on individual classes was remarkably similar, with slight variations in precision and recall for specific plant type-stage combinations.

## Deployment Considerations

For deploying these models in AgroTech's production environment:

1. **Model Retraining**: Schedule periodic retraining as new sensor data accumulates
2. **Real-time Predictions**: Implement a service for real-time temperature predictions
3. **Monitoring**: Track model performance and drift to maintain accuracy
4. **Feedback Loop**: Capture corrections from domain experts to improve future model versions
5. **Scaling**: Ensure the solution scales to multiple farm environments and sensor configurations