import sys
import os
import argparse
import yaml
import sqlite3

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rf_model import rf_PlantTypeStageClassifier
from svc_model import svc_PlantTypeStageClassifier
from svr_model import svr_TemperaturePredictor

def evaluate_models(config_path, task):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    if config is None:
        raise ValueError("Configuration file is empty or could not be loaded.")

    if task == 'classification':
        model_classes = {
            'rf': rf_PlantTypeStageClassifier,
            'svc': svc_PlantTypeStageClassifier
        }
        print(f"Starting plant type-stage classification task")
    
    elif task == 'regression':
        model_classes = {
          'svr': svr_TemperaturePredictor
        }
        print(f"Starting temperature prediction regression task")
    
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'classification' or 'regression'")

    for model_name in config['experiment']['models'][task]:
        model_class = model_classes.get(model_name)
        if not model_class:
            print(f'Unknown model: {model_name}')
            continue

        print(f'Evaluating model: {model_name}')
        model = model_class(config_path)

       
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, config['experiment']['output_model_path'], f'{task}_{model_name}_best_model.pkl')
        model_path = os.path.abspath(model_path)
        model.load_model(model_path)
        
       
        df = model.load_data(config['data']['path'])
        _, X_test, _, y_test = model.prepare_df(df, task)
        
     
        model.evaluate(X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate agriculture detection models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'regression'],
                      help='Task type: classification (plant type-stage) or regression (temperature prediction)')
    
    args = parser.parse_args()
    evaluate_models(args.config, args.task)
