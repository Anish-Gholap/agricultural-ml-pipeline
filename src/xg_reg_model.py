from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class xgb_TemperaturePredictor(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = XGBRegressor(random_state=self.config['experiment']['random_state'])
    
    def train(self, X_train, y_train):
        param_dist = {
            'n_estimators': self.config['models']['xgb_reg']['n_estimators'],
            'max_depth': self.config['models']['xgb_reg']['max_depth'],
            'learning_rate': self.config['models']['xgb_reg']['learning_rate'],
            'subsample': self.config['models']['xgb_reg']['subsample'],
            'colsample_bytree': self.config['models']['xgb_reg']['colsample_bytree'],
            'min_child_weight': self.config['models']['xgb_reg']['min_child_weight'],
            'gamma': self.config['models']['xgb_reg']['gamma'],
            'reg_alpha': self.config['models']['xgb_reg']['reg_alpha'],
            'reg_lambda': self.config['models']['xgb_reg']['reg_lambda']
        }
        n_iter_search = self.config['experiment']['n_iter']
        random_search = RandomizedSearchCV(
            self.model, 
            param_distributions=param_dist, 
            n_iter=n_iter_search, 
            refit=True, 
            verbose=2, 
            cv=5, 
            scoring='neg_mean_squared_error',
            random_state=self.config['experiment']['random_state'], 
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        print(f'Best parameters: {random_search.best_params_}')
    
    def predict(self, X_single):
        return super().predict(X_single)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'Root Mean Squared Error: {rmse:.4f}')
        print(f'Mean Absolute Error: {mae:.4f}')
        print(f'R² Score: {r2:.4f}')
    
    def print_feature_importance(self):
        """
        Prints feature importances of the trained XGBoost model
        without requiring explicit feature names.
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get feature importances - use appropriate method based on booster type
        try:
            # For tree-based models
            importances = self.model.feature_importances_
        except:
            # Fallback if feature_importances_ is not available
            print("Feature importance not available for this model type")
            return None
        
        # Try to get feature names from column transformer
        try:
            feature_names = self.column_transformer.get_feature_names_out()
        except:
            # Fallback to generic names
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Print the importance values
        print("\nFeature Importances:")
        for i, row in feature_importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        # Create a horizontal bar chart
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df