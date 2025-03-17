# AgroTech Innovations - Farm Environment Prediction

## Anish Gholap (AIIP5 Technical Assessment)

Email: anish.gholap@example.com

## Project Overview

This project develops an end-to-end machine learning pipeline for AgroTech Innovations to improve crop management in controlled farming environments. The pipeline addresses two key challenges:

1. **Temperature Prediction**: Developing regression models to predict temperature conditions within the farm environment
2. **Plant Type-Stage Classification**: Creating classification models to categorize plant types and growth stages based on sensor data

The models help AgroTech Innovations optimize resource allocation, improve crop management, and increase yield predictability.

## Folder Structure

```
.
├── data/
│   ├── agri.db
├── models/
│   ├── rf_reg_best_model.pkl
│   ├── xgb_reg_best_model.pkl
│   ├── rf_best_model.pkl
├── src/
│   ├── base_model.py
│   ├── config.yaml
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── label_mappings.py
│   ├── rf_model.py
│   ├── rf_reg_model.py
│   ├── train.py
│   ├── xgb_reg_model.py
├── __init__.py
├── __pycache__
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
- Missing values in nutrient sensor readings
- Some negative values in temperature readings requiring filtering
- Inconsistent formatting in plant type and stage labels

### Feature Relationships
- Strong correlation between light intensity and temperature
- pH levels showing distinct clusters for different plant types
- CO2 and O2 levels exhibiting inverse relationships

### Feature Engineering Opportunities
- NPK ratios provide valuable information about nutrient balance
- Light-to-CO2 ratio captures important growing conditions
- pH category clustering reveals optimal ranges for different plant types

The detailed findings can be found in `eda.ipynb`.

## Feature Processing Summary

| Feature | Transformation |
|---------|----------------|
| Temperature Sensor (°C) | Filtered negative values |
| Light & CO2 | Created Light:CO2 ratio feature |
| Nutrients (N, P, K) | Created NPK ratio features |
| pH Sensor | Clustered into categorical ranges |
| System Location & Plant Type | One-hot encoded |
| Missing Values | Median imputation |
| Numerical Features | Standard scaling |

## Model Selection and Evaluation

### Regression Task (Temperature Prediction)

I selected Random Forest and XGBoost regressors for temperature prediction based on their performance and characteristics:

#### Random Forest Regressor
- **Strengths**: Robust to outliers, handles non-linear relationships, provides feature importance
- **Performance**: RMSE: 0.97, R²: 0.63
- **Key Parameters**: n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=1

#### XGBoost Regressor
- **Strengths**: Gradient boosting approach, excellent performance on structured data, regularization options
- **Performance**: RMSE: 0.95, R²: 0.64
- **Key Parameters**: {subsample': 0.8, 'reg_lambda': 1.0, 'reg_alpha': 1.0, 'n_estimators': 300, 'min_child_weight': 3, 'max_depth': 9, 'learning_rate': 0.05, 'gamma': 0.2, 'colsample_bytree': 1.0}

**Why These Models?**
- Both models handle the complex non-linear relationships in environmental sensor data
- They provide feature importance insights for agricultural domain experts
- They're computationally efficient for retraining as new data becomes available
- They outperformed SVM in both accuracy and training efficiency (SVR took significantly longer to train)

### Classification Task (Plant Type-Stage)

For plant type-stage classification, I also selected Random Forest and XGBoost:

#### Random Forest Classifier
- **Performance**: Accuracy: 0.72, F1-score: 0.71
- **Key Parameters**: n_estimators=100, max_depth=20, min_samples_leaf=4

#### XGBoost Classifi
- **Performance**: Accuracy: 0.72, F1-score: 0.72
- **Key Parameters**: n_estimators=200, max_depth=5, learning_rate=0.1

**Why These Models?**
- Both models handle multi-class classification tasks effectively
- They work well with the mixed numerical and categorical features in our dataset
- They're robust to the slight class imbalance present in plant type-stage combinations
- They provide insights into which sensor readings are most indicative of plant types and stages
- They significantly outperformed SVM in terms of classification accuracy

## Deployment Considerations

For deploying these models in AgroTech's production environment:

1. **Model Retraining**: Schedule periodic retraining as new sensor data accumulates
2. **Real-time Predictions**: Implement a service for real-time temperature predictions
3. **Monitoring**: Track model performance and drift to maintain accuracy
4. **Feedback Loop**: Capture corrections from domain experts to improve future model versions
5. **Scaling**: Ensure the solution scales to multiple farm environments and sensor configurations
