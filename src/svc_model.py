from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from base_model import BaseModel

class svc_PlantTypeStageClassifier(BaseModel):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.model = SVC(random_state=self.config['experiment']['random_state'])
  
  def train(self, X_train, y_train):
      param_dist = {
          'C': self.config['models']['svm']['C'],
          'gamma': self.config['models']['svm']['gamma'],
          'kernel': self.config['models']['svm']['kernel'],
          'degree': self.config['models']['svm']['degree'],
          'coef0': self.config['models']['svm']['coef0']
      }
      n_iter_search = self.config['experiment']['n_iter']
      random_search = RandomizedSearchCV(self.model, param_distributions=param_dist, n_iter=n_iter_search, refit=True, verbose=2, cv=5, random_state=self.config['experiment']['random_state'], n_jobs=-1)
      random_search.fit(X_train, y_train)
      self.model = random_search.best_estimator_
      print(f'Best parameters: {random_search.best_params_}')

  def predict(self, X_single):
      return super().predict(X_single)