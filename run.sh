#!/bin/bash

# Activate the virtual environment if needed
# source .venv/bin/activate

# Define the configuration file path
CONFIG_FILE="src/config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

echo "===== Starting Agricultural Machine Learning Pipeline ====="

# echo "Starting temperature prediction task..."
python src/train.py --config $CONFIG_FILE --task regression
python src/evaluate.py --config $CONFIG_FILE --task regression

echo "Starting plant type-stage classification task..."
python src/train.py --config $CONFIG_FILE --task classification  
python src/evaluate.py --config $CONFIG_FILE --task classification

echo "===== Pipeline execution completed ====="

exit 0