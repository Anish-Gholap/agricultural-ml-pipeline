import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def add_npk_ratio(df):
  df['npk_ratio'] = (df['Nutrient N Sensor (ppm)'] / df['Nutrient P Sensor (ppm)'] / df['Nutrient K Sensor (ppm)']) 
  
  return df

def add_light_Co2_ratio(df):
  df['light_co2_ratio'] = df['Light Intensity Sensor (lux)'] / df['CO2 Sensor (ppm)']
  
  return df

def add_ph_category(df):
    """
    Adds pH category feature to the dataframe based on clustered pH values.
    Creates a new column 'pH Category Clustered' with values 'low', 'optimal', or 'high'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Plant Type-Stage' and 'pH Sensor' columns
    
    Returns:
    --------
    pandas.DataFrame
        The original DataFrame with an additional 'pH Category Clustered' column
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Dictionary to store optimal ranges
    optimal_ph_ranges = {}
    
    # Group data by Plant Type-Stage
    grouped = df_copy.groupby('Plant Type-Stage')
    
    # For each group, find natural pH clusters
    for name, group in grouped:
        # Skip groups with too few samples
        if len(group) < 3:
            # Use global average if group is too small
            optimal_ph_ranges[name] = (6.0, 7.0)  # Default fallback values
            continue
            
        # Handle NaN values
        ph_values = group['pH Sensor'].dropna().values
        
        # Skip if no valid pH values
        if len(ph_values) < 3:
            optimal_ph_ranges[name] = (6.0, 7.0)  # Default fallback values
            continue
            
        # Reshape data for KMeans (needs 2D array)
        ph_values = ph_values.reshape(-1, 1)
        
        # Find 3 natural clusters in pH values
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(ph_values)
            
            # Sort cluster centers (low, medium, high)
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # Calculate boundaries between clusters
            low_high_boundary = (centers[0] + centers[1]) / 2
            high_low_boundary = (centers[1] + centers[2]) / 2
            
            # Store thresholds
            optimal_ph_ranges[name] = (low_high_boundary, high_low_boundary)
        except:
            # Fallback if clustering fails
            optimal_ph_ranges[name] = (6.0, 7.0)
    
    # Create pH category feature
    def categorize_ph_clustered(row):
        plant_type_stage = row['Plant Type-Stage']
        ph = row['pH Sensor']
        
        # Handle missing values
        if pd.isna(ph) or plant_type_stage not in optimal_ph_ranges:
            return np.nan
        
        low_threshold, high_threshold = optimal_ph_ranges[plant_type_stage]
        
        if ph < low_threshold:
            return 'low'
        elif ph > high_threshold:
            return 'high'
        else:
            return 'optimal'
    
    # Apply the categorization function
    df_copy['pH Category Clustered'] = df_copy.apply(categorize_ph_clustered, axis=1)
    
    return df_copy
  
def add_light_temp_ratio(df):
  df['light_temp_ratio'] = df['Light Intensity Sensor (lux)'] / df['Temperature Sensor (°C)']
  
  return df