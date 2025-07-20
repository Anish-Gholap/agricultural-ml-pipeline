# 🌱 Farm Environment ML Pipeline

*An end-to-end machine learning solution for smart agriculture and precision farming*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.0-green.svg)](https://xgboost.readthedocs.io/)

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline for agricultural IoT data, addressing real-world challenges in precision farming. It showcases advanced feature engineering, model selection, and deployment-ready ML systems for smart agriculture applications.

### 🔍 Problem Statement
Modern agriculture faces critical challenges in optimizing crop growth conditions and monitoring plant development. Traditional manual monitoring is inefficient and error-prone, while IoT sensor data requires sophisticated analysis to extract actionable insights.

**Key Challenges Addressed:**
- **Temperature Optimization**: Predict optimal temperature conditions for crop growth using environmental sensor data
- **Automated Plant Monitoring**: Classify plant types and growth stages automatically, eliminating manual inspection
- **Data Quality Issues**: Handle noisy, inconsistent IoT sensor data with missing values and anomalies
- **Real-time Decision Making**: Enable data-driven agricultural decisions for resource optimization

**Technical Impact**: Enable precision farming through automated analysis of multi-sensor agricultural data

### 🏆 Key Achievements
- **86% accuracy** on multi-class plant classification (12 different plant type-stage combinations)
- **R² = 0.68** for temperature prediction with RMSE of 0.91°C
- Robust feature engineering pipeline handling real-world sensor data anomalies
- Production-ready, modular architecture with comprehensive evaluation metrics

## 🚀 Technical Highlights

### Machine Learning Techniques
- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost
- **Feature Engineering**: Custom transformers for agricultural domain knowledge
- **Data Pipeline**: Robust preprocessing for noisy IoT sensor data
- **Model Selection**: Systematic evaluation across regression and classification tasks

### Engineering Best Practices
- Object-oriented design with inheritance and composition
- Configuration-driven experimentation (YAML-based)
- Automated pipeline execution with shell scripting
- Comprehensive data validation and cleaning

## 🛠️ Tech Stack

```
Core ML: scikit-learn, XGBoost, NumPy, Pandas
Visualization: Matplotlib, Seaborn
Data Processing: SQLite, PyYAML
Environment: Python 3.9+, Jupyter Notebooks
```

## 📊 Model Performance

### Temperature Prediction (Regression)
| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Random Forest** | **0.907** | **0.672** | **0.681** |
| XGBoost | 0.944 | 0.732 | 0.655 |
| SVR | 1.124 | 0.891 | 0.523 |

### Plant Classification (Multi-class)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **86%** | **0.86** | **0.86** | **0.86** |
| **Gradient Boosting** | **86%** | **0.86** | **0.86** | **0.86** |
| SVM | 78% | 0.77 | 0.78 | 0.77 |

## 🔧 Advanced Feature Engineering

### Domain-Specific Features
```python
# NPK Nutrient Balance Ratio
npk_ratio = (N + P + K) / 3

# Light-to-CO2 Growing Conditions
light_co2_ratio = Light_Intensity / CO2_Concentration

# Temperature-Light Optimization
light_temp_ratio = Light_Intensity / Temperature

# pH Categorical Clustering
pH_categories = ["Acidic", "Slightly_Acidic", "Neutral", "Slightly_Basic", "Basic"]
```

### Data Quality Solutions
- **Negative Value Filtering**: Removed physically impossible sensor readings
- **Smart Imputation**: Strategy-based missing value handling per sensor type
- **Data Type Correction**: Fixed inconsistent nutrient sensor formats
- **Label Standardization**: Unified plant type and stage naming conventions

## 🏗️ Architecture & Design Patterns

### Object-Oriented ML Pipeline
```
BaseModel (Abstract Class)
├── RegressionModels
│   ├── RandomForestRegressor
│   ├── XGBoostRegressor
│   └── SVRModel
└── ClassificationModels
    ├── RandomForestClassifier
    ├── GradientBoostingClassifier
    └── SVMClassifier
```

### Configuration-Driven Development
- YAML-based hyperparameter management
- Modular feature selection
- Environment-specific settings
- Easy experimentation and reproducibility

## 📈 Key Insights from EDA

### Sensor Data Patterns
- **Strong Correlation**: Light intensity ↔ Temperature (r=0.73)
- **Inverse Relationship**: CO2 ↔ O2 levels (r=-0.65)
- **pH Clustering**: Distinct optimal ranges per plant type
- **Seasonal Patterns**: Clear growth stage transitions in sensor readings

### Feature Importance (Top 5)
1. **Light Intensity** (0.24) - Primary growth driver
2. **CO2 Concentration** (0.18) - Photosynthesis indicator  
3. **NPK Ratio** (0.16) - Nutrient balance metric
4. **pH Level** (0.14) - Soil condition proxy
5. **Temperature** (0.13) - Environmental stability

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# Make executable
chmod +x run.sh

# Execute pipeline
./run.sh

# Or run specific tasks
python src/train.py --config src/config.yaml --task regression
python src/evaluate.py --config src/config.yaml --task classification
```

### Project Structure
```
src/
├── base_model.py          # Abstract base class for all models
├── feature_engineering.py # Custom transformers and feature creation
├── {model_name}_model.py  # Individual model implementations
├── train.py              # Training pipeline
├── evaluate.py           # Model evaluation and metrics
├── config.yaml           # Configuration management
└── label_mappings.py     # Data standardization utilities

eda.ipynb                 # Comprehensive exploratory data analysis
requirements.txt          # Dependency management
run.sh                   # Automated pipeline execution
```

## 🔬 Experimentation & Validation

### Cross-Validation Strategy
- **5-Fold CV** for robust performance estimation
- **Stratified sampling** to handle class imbalance
- **Time-series aware** splits for temporal data

### Hyperparameter Optimization
- Grid search for Random Forest parameters
- Bayesian optimization for XGBoost tuning
- Feature selection using recursive elimination

## 🎯 Business Impact & Applications

### Immediate Benefits
- **25% reduction** in temperature prediction errors
- **Automated classification** eliminating manual plant monitoring
- **Real-time insights** for proactive farm management

### Scalability Considerations
- Modular design supports new sensor types
- Configuration-driven for multi-farm deployment
- API-ready architecture for cloud integration

##  Future Enhancements

- **Deep Learning**: LSTM models for time-series forecasting
- **MLOps**: Model versioning, monitoring, and automated retraining
- **Real-time Deployment**: FastAPI service with Docker containerization
- **Advanced Features**: Weather integration and satellite imagery
- **Explainability**: SHAP values for model interpretability

---
