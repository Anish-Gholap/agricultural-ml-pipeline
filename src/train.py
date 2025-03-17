import sys
import os
import argparse
import yaml
import sqlite3

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rf_model import rf_PlantTypeStageClassifier
from svc_model import svc_PlantTypeStageClassifier
from gb_model import gb_PlantTypeStageClassifier
from svr_model import svr_TemperaturePredictor
from rf_reg_model import rf_TemperaturePredictor
from xg_reg_model import xgb_TemperaturePredictor

def train(config_path, task):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config is None:
        raise ValueError("Configuration file is empty or could not be loaded.")
    
    if task == 'classification':
        model_classes = {
            'rf': rf_PlantTypeStageClassifier,
            'svc': svc_PlantTypeStageClassifier,
            'gb': gb_PlantTypeStageClassifier
        }
        print(f"Starting plant type-stage classification task")
    
    elif task == 'regression':
        model_classes = {
          'svr': svr_TemperaturePredictor,
          'rf_reg': rf_TemperaturePredictor,
          'xg_reg': xgb_TemperaturePredictor
        }
        print(f"Starting temperature prediction regression task")
    
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'classification' or 'regression'")
    
    
    for model_name in config['experiment']['models'][task]:
        model_class = model_classes.get(model_name)
        if not model_class:
            print(f'Unknown model: {model_name}')
            continue
        
        print(f'Training and evaluating {task} model: {model_name}')
        model = model_class(config_path)
        
        data = model.load_data(config['data']['path'])
        X_train, X_test, y_train, y_test = model.prepare_df(data, task)
        
        # Train the model
        model.train(X_train, y_train)
        
        # Save model
        script_dir = os.path.dirname(__file__)
        output_model_dir = os.path.abspath(os.path.join(script_dir, config['experiment']['output_model_path']))
        os.makedirs(output_model_dir, exist_ok=True)
        output_model_path = os.path.join(output_model_dir, f'{task}_{model_name}_best_model.pkl')
        model.save_model(output_model_path)
        
        print(f"Model saved to {output_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate agricultural models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'regression'],
                        help='Task type: classification (plant type-stage) or regression (temperature prediction)')
    
    args = parser.parse_args()
    train(args.config, args.task)