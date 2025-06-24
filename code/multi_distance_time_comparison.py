#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Distance mmWave Heart Signal Comparison Tool

This program loads CSV files from 4 different distances (30cm, 45cm, 60cm, 90cm),
processes the heart waveform data, and creates an overlay comparison plot.

Features:
- Fixed sampling rate: fs = 1/0.090 Hz (≈11.11 Hz)
- Time window: 0 to 79 seconds (1 minute 19 seconds)
- Overlay plot with different colors for each distance
- All labels and text in English
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Global Settings ---

# Set matplotlib to use English fonts
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font setting error: {e}. English labels may not display correctly.")

# --- Core Functions ---

def load_csv_data(csv_file_path):
    """
    Load heart waveform data from CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame or None: DataFrame containing the data, None if failed.
    """
    if not csv_file_path or not os.path.isfile(csv_file_path):
        print(f"Error: Invalid CSV file path '{csv_file_path}'")
        return None
    
    try:
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        required_columns = ['Heart_Waveform', 'Frame_Number']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Warning: CSV file missing required columns: {', '.join(missing)}")
            return None
        
        return df
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}")
        return None

def process_heart_data(df, distance_label, end_time_seconds=79.0):
    """
    Process heart waveform data with fixed sampling rate and time window.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Heart_Waveform' data.
        distance_label (str): Label for the distance (e.g., "30cm").
        end_time_seconds (float): End time in seconds (default: 79 seconds).
        
    Returns:
        tuple: (time_axis, heart_data, info_dict) or (None, None, None)
    """
    if df is None or df.empty:
        print(f"Warning: {distance_label} DataFrame is empty")
        return None, None, None
    
    # Fixed sampling rate
    fs = 1 / 0.090  # ≈11.11 Hz
    
    # Sort by Frame_Number to ensure correct time sequence
    df_sorted = df.sort_values(by='Frame_Number')
    heart_data_raw = df_sorted['Heart_Waveform'].values
    
    # Check if heart_data_raw is numeric
    if not pd.api.types.is_numeric_dtype(heart_data_raw):
        print(f"Warning: {distance_label} 'Heart_Waveform' data is not numeric. Found dtype: {heart_data_raw.dtype}")
        try:
            heart_data = pd.to_numeric(heart_data_raw, errors='coerce')
            print(f"         Attempted conversion of 'Heart_Waveform' to numeric for {distance_label}.")
            if np.all(np.isnan(heart_data)): # Check after conversion
                print(f"Warning: {distance_label} 'Heart_Waveform' became all NaN after conversion.")
                return None, None, None
        except Exception as e_conv:
            print(f"         Failed to convert 'Heart_Waveform' to numeric for {distance_label}: {e_conv}")
            return None, None, None
    else:
        heart_data = heart_data_raw

    # Check for all NaNs in numeric data
    if np.all(np.isnan(heart_data)):
        print(f"Warning: {distance_label} 'Heart_Waveform' data is all NaN.")
        return None, None, None
        
    N = len(heart_data)
    
    if N == 0:
        print(f"Warning: {distance_label} no heart waveform data available (N=0 after initial checks)")
        return None, None, None
    
    # Generate time axis (seconds)
    time_axis_full = np.arange(N) / fs
    total_duration = time_axis_full[-1] if N > 1 else 0
    
    # Calculate the number of samples for the desired end time
    target_samples = int(end_time_seconds * fs)
    
    if N <= target_samples:
        # Use all available data
        time_axis = time_axis_full
        processed_heart_data = heart_data
        actual_end_time = total_duration
        info = f"({distance_label}: All {total_duration:.1f}s)"
    else:
        # Take first 79 seconds (target_samples)
        time_axis = time_axis_full[:target_samples]
        processed_heart_data = heart_data[:target_samples]
        actual_end_time = time_axis[-1]
        info = f"({distance_label}: 0 to {actual_end_time:.1f}s)"
    
    info_dict = {
        'distance': distance_label,
        'total_samples': len(processed_heart_data),
        'duration': actual_end_time,
        'sampling_rate': fs
    }
    
    return time_axis, processed_heart_data, info_dict

def create_overlay_plot(data_dict, output_file_path):
    """
    Create an overlay comparison plot for multiple distances.
    
    Args:
        data_dict (dict): Dictionary containing processed data for each distance.
        output_file_path (str): Path to save the output plot.
    """
    if not data_dict:
        print("Error: No data available for plotting")
        return False
    
    # Define colors for each distance
    colors = {
        '30cm': '#FF0000',  # Red
        '45cm': '#4ECDC4',  # Teal
        '60cm': '#FFFF00',  # yellow
        '90cm': '#2EFF00'   # Green
    }
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot each distance with different colors
    for distance, data in data_dict.items():
        time_axis, heart_data, info = data
        color = colors.get(distance, '#333333')  # Default to dark gray if color not found
        
        plt.plot(time_axis, heart_data, 
                color=color, 
                linewidth=1.2, 
                label=f'mmWave Heart Signal {distance}', 
                alpha=0.8)
        
        print(f"Plotted {distance}: {len(heart_data)} samples, duration: {time_axis[-1]:.1f}s")
    
    # Customize the plot
    plt.title('mmWave Heart Amplitude Signal Comparison\n(Multiple Distances: 30cm, 45cm, 60cm, 90cm)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Heart Amplitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    # Set x-axis limits and ensure 79 second mark is visible
    plt.xlim(0, 79)
    current_xticks = list(plt.gca().get_xticks())
    if 79 not in current_xticks:
        current_xticks.append(79)
    plt.xticks(current_xticks)
    
    # Format x-axis labels
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Overlay comparison plot saved: {output_file_path}")
        return True
    except Exception as e:
        print(f"Error: Failed to save plot: {e}")
        return False

def print_data_summary(data_dict):
    """
    Print a summary of the loaded and processed data.
    
    Args:
        data_dict (dict): Dictionary containing processed data for each distance.
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for distance, data in data_dict.items():
        time_axis, heart_data, info = data
        print(f"{distance.upper()}:")
        print(f"  - Samples: {info['total_samples']}")
        print(f"  - Duration: {info['duration']:.2f} seconds")
        print(f"  - Sampling Rate: {info['sampling_rate']:.2f} Hz")
        print(f"  - Amplitude Range: [{np.min(heart_data):.3f}, {np.max(heart_data):.3f}]")
        print(f"  - Mean Amplitude: {np.mean(heart_data):.3f}")
        print()

def main():
    """
    Main function to load, process, and visualize multi-distance heart signal data.
    """
    print("=== Multi-Distance mmWave Heart Signal Comparison Tool ===")
    
    # Define CSV file paths for each distance
    csv_files = {
        '30cm': "/path/to/30cm/data.csv",
        '45cm': "/path/to/45cm/data.csv",
        '60cm': "/path/to/60cm/data.csv",
        '90cm': "/path/to/90cm/data.csv"
    }
    
    # Output directory for the comparison plot
    output_directory = "/path/to/output/directory"
    output_filename = "multi_distance_heart_signal_comparison.png"
    output_file_path = os.path.join(output_directory, output_filename)
    
    # Load and process data for each distance
    processed_data = {}
    
    for distance, csv_path in csv_files.items():
        print(f"\nProcessing {distance} data...")
        
        # Check if file exists
        if not os.path.isfile(csv_path):
            print(f"Warning: File not found for {distance}: {csv_path}")
            continue
        
        # Load CSV data
        df = load_csv_data(csv_path)
        if df is None:
            print(f"Failed to load data for {distance}")
            continue
        
        # Process heart data
        time_axis, heart_data, info = process_heart_data(df, distance, end_time_seconds=79.0)
        if time_axis is None:
            print(f"Failed to process data for {distance}")
            continue
        
        # Store processed data
        processed_data[distance] = (time_axis, heart_data, info)
        print(f"Successfully processed {distance}: {len(heart_data)} samples")
    
    # Check if we have any valid data
    if not processed_data:
        print("Error: No valid data loaded. Please check file paths and data format.")
        return
    
    # Print data summary
    print_data_summary(processed_data)
    
    # Create overlay comparison plot
    print(f"\nGenerating overlay comparison plot...")
    if create_overlay_plot(processed_data, output_file_path):
        print("Multi-distance comparison completed successfully!")
    else:
        print("Failed to create comparison plot.")
    
    print(f"\nProcessed distances: {', '.join(processed_data.keys())}")

# --- Program Entry Point ---
if __name__ == "__main__":
    main()
    print("=== Program Finished ===")
