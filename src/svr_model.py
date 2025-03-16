from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from base_model import BaseModel
import numpy as np

class svr_TemperaturePredictor(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = SVR()
    
    def train(self, X_train, y_train):
        param_dist = {
            'C': self.config['models']['svr']['C'],
            'gamma': self.config['models']['svr']['gamma'],
            'kernel': self.config['models']['svr']['kernel'],
            'epsilon': self.config['models']['svr']['epsilon'],
            'degree': self.config['models']['svr']['degree'],
            'coef0': self.config['models']['svr']['coef0']
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