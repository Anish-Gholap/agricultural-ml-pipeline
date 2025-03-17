from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class rf_TemperaturePredictor(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = RandomForestRegressor(random_state=self.config['experiment']['random_state'])
    
    def train(self, X_train, y_train):
        param_dist = {
            'n_estimators': self.config['models']['rf_reg']['n_estimators'],
            'max_depth': self.config['models']['rf_reg']['max_depth'],
            'min_samples_split': self.config['models']['rf_reg']['min_samples_split'],
            'min_samples_leaf': self.config['models']['rf_reg']['min_samples_leaf'],
            'max_features': self.config['models']['rf_reg']['max_features'],
            'bootstrap': self.config['models']['rf_reg']['bootstrap']
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
        Prints feature importances of the trained Random Forest model
        without requiring explicit feature names.
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
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
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df