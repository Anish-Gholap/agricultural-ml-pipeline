from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from base_model import BaseModel

class gb_PlantTypeStageClassifier(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = GradientBoostingClassifier(random_state=self.config['experiment']['random_state'])

    def train(self, X_train, y_train):
        param_dist = {
            'n_estimators': self.config['models']['gb']['n_estimators'],
            'learning_rate': self.config['models']['gb']['learning_rate'],
            'max_depth': self.config['models']['gb']['max_depth'],
            'subsample': self.config['models']['gb']['subsample'],
            'min_samples_split': self.config['models']['gb']['min_samples_split'],
            'min_samples_leaf': self.config['models']['gb']['min_samples_leaf'],
            'max_features': self.config['models']['gb']['max_features']
        }
        n_iter_search = self.config['experiment']['n_iter']
        random_search = RandomizedSearchCV(self.model, param_distributions=param_dist, n_iter=n_iter_search, refit=True, verbose=2, cv=5, random_state=self.config['experiment']['random_state'], n_jobs=-1)
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        print(f'Best parameters: {random_search.best_params_}')

    def predict(self, X_single):
        return super().predict(X_single)