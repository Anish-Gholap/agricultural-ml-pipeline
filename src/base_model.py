import pandas as pd
import numpy as np
import sqlite3 as sql
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import os
import yaml
import joblib
from label_mappings import plant_stage_mapping, plant_type_mapping

class BaseModel:
  def __init__(self, config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
      self.config = yaml.safe_load(file)
    self.model = None
    self.column_transformer = None
    
  def load_data(self, db_path):
    current_dir = os.path.dirname(__file__)
    absolute_db_path = os.path.join(current_dir, db_path)
    absolute_db_path = os.path.abspath(absolute_db_path)
    connect = sql.connect(absolute_db_path)
    query = "SELECT * FROM farm_data"
    df = pd.read_sql_query(query, connect)
    connect.close()
      
    return df
  
  def preprocess_data(self, X, y=None, fit=False, task='regression'):
    num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    
    if task == 'regression':
      num_attribs = self.config['data']['numerical_features_reg']
      cat_attribs = self.config['data']['categorical_features_reg']
      
    else:
      num_attribs = self.config['data']['numerical_features_class']
      cat_attribs = self.config['data']['categorical_features_class']

    
    if fit:
      self.column_transformer = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
      ])
      X_transformed = self.column_transformer.fit_transform(X)
    
    else: 
      if self.column_transformer is None:
        raise ValueError("fit column transformer first")
      
      X_transformed = self.column_transformer.transform(X)
    
    if y is not None:
      return X_transformed, y  

    return X_transformed
  
  
  def preprocess_label(self, y):
    encoder = LabelEncoder()
    y_transformed = encoder.fit_transform(y)
    return y_transformed


  def prepare_df(self, df, task):
    # Fix dtype for Nutrients
    columns = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
    for column in columns:
      # Replace None with NaN
      df[column] = df[column].replace('None', None)
      # Remove ppm suffix
      df[column] = df[column].str.replace('ppm', '', regex=True)
      # Convert to float64
      df[column] = df[column].astype('float64')
    
    # Fix labels
    df['Plant Type'] = df['Plant Type'].replace(plant_type_mapping)
    df['Plant Stage'] = df['Plant Stage'].replace(plant_stage_mapping)
    
    # Create target labels
    df['Plant Type-Stage'] = df['Plant Type'] + '-' + df['Plant Stage']
    
    # Drop features not needed
    columns_drop = ['Plant Type', 'Plant Stage', 'Humidity Sensor (%)']
    df.drop(columns=columns_drop, inplace=True)
    
    df = df[(df['Light Intensity Sensor (lux)'] >= 0) & (df['EC Sensor (dS/m)'] >= 0) & (df['Temperature Sensor (°C)'] >= 0)]
    
    df_regression = df.copy()
    df_classficaiton = df.copy()
    
    regression_target = 'Temperature Sensor (°C)'
    classification_target = 'Plant Type-Stage'
    
    y_reg = df_regression[regression_target]
    X_reg = df_regression.drop(columns=regression_target).copy()
    
    y_class = df_classficaiton[classification_target]
    X_class = df_classficaiton.drop(columns=classification_target).copy()
    
    
    if task == 'regression':
      X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=self.config['experiment']['random_state'])
      X_train_reg, y_train_reg = self.preprocess_data(X_train_reg, y_train_reg, fit=True)
      X_test_reg, y_test_reg = self.preprocess_data(X_test_reg, y_test_reg, fit=False)
      
      return X_train_reg, X_test_reg, y_train_reg, y_test_reg
    
    else:
      X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=self.config['experiment']['random_state'], stratify=y_class)
      X_train_class, y_train_class = self.preprocess_data(X_train_class, y_train_class, fit=True, task=task)
      X_test_class, y_test_class = self.preprocess_data(X_test_class, y_test_class, fit=False, task=task)
      
      y_train_class = self.preprocess_label(y_train_class)
      y_test_class = self.preprocess_label(y_test_class)
      
      return X_train_class, X_test_class, y_train_class, y_test_class
    
  def preprocess_single(self, X_single):
        X_single = pd.DataFrame([X_single])
        X_single = self.preprocess(X_single, fit=False)
        return X_single

  def train(self, X_train, y_train):
      raise NotImplementedError

# For classification
  def evaluate(self, X_test, y_test):
    predictions = self.model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, predictions))

  def save_model(self, filepath):
    with open(filepath, 'wb') as file:
        joblib.dump((self.model, self.column_transformer), file)

  def load_model(self, filepath):
    with open(filepath, 'rb') as file:
        self.model, self.column_transformer = joblib.load(file)

  def predict(self, X_single):
    X_single_preprocessed = self.preprocess_single(X_single)
    prediction = self.model.predict(X_single_preprocessed)
    return prediction