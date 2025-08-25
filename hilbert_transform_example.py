#!/usr/bin/env python3
"""
Hilbert Transform Processing Example

This script demonstrates how to use the HilbertTransformProcessor class to generate 
an Excel file with different frames for each sample containing the Hilbert transformation 
of signals without normalization.
"""

import os
from pathlib import Path

# Change to the parent directory to access the scr module
os.chdir(Path(__file__).parent)
print(f"Working directory: {Path.cwd()}")

# Import the HilbertTransformProcessor
from scr.clarius.main import HilbertTransformProcessor

def main():
    """Main function to demonstrate Hilbert transform processing."""
    
    # Define paths (update these to match your actual paths)
    samples_folder_path = r"C:\johanna_samples\raw_data_metastasis"  # Path to your samples folder
    roi_folder_path = r"C:\johanna_samples\gui_metastasis"         # Path to your ROI folder
    result_folder_path = r"C:\hilbert_transform_results"            # Path where to save results
    
    print("Starting Hilbert Transform Processing...")
    print(f"Samples folder: {samples_folder_path}")
    print(f"ROI folder: {roi_folder_path}")
    print(f"Results folder: {result_folder_path}")
    
    try:
        # Initialize HilbertTransformProcessor
        hilbert_processor = HilbertTransformProcessor(
            samples_folder_path=samples_folder_path,
            result_folder_path=result_folder_path,
            roi_folder_path=roi_folder_path,
            roi_size='large'  # or 'small'
        )
        
        print("\nHilbert transform processing completed successfully!")
        print(f"Results saved to: {result_folder_path}/ with individual sample folders")
        
        # Demonstrate reading the results
        demonstrate_results(result_folder_path)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

def demonstrate_results(result_folder_path):
    """Demonstrate how to read and analyze the results."""
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get sample folders
    sample_folders = [f for f in Path(result_folder_path).iterdir() if f.is_dir()]
    
    if not sample_folders:
        print(f"No sample folders found in {result_folder_path}")
        return
    
    print(f"\nAvailable sample folders: {[f.name for f in sample_folders]}")
    
    # Read data for the first sample
    if sample_folders:
        first_sample_folder = sample_folders[0]
        sample_name = first_sample_folder.name
        
        print(f"\nAnalyzing sample: {sample_name}")
        
        # Find sample file
        sample_file = first_sample_folder / "hilbert_amplitude.xlsx"
        
        if sample_file.exists():
            print(f"Found sample file: {sample_file}")
            
            # Get sheet names to see available frames
            excel_file_obj = pd.ExcelFile(sample_file)
            sheet_names = excel_file_obj.sheet_names
            frame_sheets = [s for s in sheet_names if s.startswith('Frame_')]
            print(f"Available frame sheets: {frame_sheets}")
            
            # Read first frame data
            if frame_sheets:
                first_frame_sheet = frame_sheets[0]
                df = pd.read_excel(sample_file, sheet_name=first_frame_sheet)
            
            print(f"\nData for sample '{sample_name}' (first frame):")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
            
            # Get unique lines
            lines = df['Line'].unique()
            print(f"\nNumber of lines: {len(lines)}")
            
            # Show summary statistics
            print(f"\nSummary statistics for {sample_name} (first frame):")
            print(df.describe())
            
            # Show data structure
            print(f"\nData structure:")
            print(f"Rows (Time points): {df.shape[0]}")
            print(f"Columns (Lines): {df.shape[1]}")
            print(f"Column names: {list(df.columns)}")
            print(f"Row names (first 5): {list(df.index[:5])}")
            
            # Optional: Create a simple plot
            try:
                create_sample_plot(df, sample_name)
            except Exception as e:
                print(f"Could not create plot: {e}")
        else:
            print(f"No sample file found in {first_sample_folder}")

def create_sample_plot(df, sample_name):
    """Create a sample plot of the Hilbert transform amplitude results."""
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: First line amplitude vs time
    plt.subplot(2, 2, 1)
    first_line = df.iloc[:, 0]  # First column (Line_1)
    plt.plot(first_line.index, first_line.values)
    plt.title('Line 1 - Hilbert Transform Amplitude')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 2: Middle line amplitude vs time
    plt.subplot(2, 2, 2)
    middle_line_idx = len(df.columns) // 2
    middle_line = df.iloc[:, middle_line_idx]
    plt.plot(middle_line.index, middle_line.values)
    plt.title(f'Line {middle_line_idx + 1} - Hilbert Transform Amplitude')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 3: Last line amplitude vs time
    plt.subplot(2, 2, 3)
    last_line = df.iloc[:, -1]  # Last column
    plt.plot(last_line.index, last_line.values)
    plt.title(f'Line {len(df.columns)} - Hilbert Transform Amplitude')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 4: Heatmap of all lines
    plt.subplot(2, 2, 4)
    plt.imshow(df.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('All Lines - Amplitude Heatmap')
    plt.xlabel('Time Point')
    plt.ylabel('Line Number')
    
    plt.tight_layout()
    plt.suptitle(f'Hilbert Transform Amplitude Results - {sample_name}', y=1.02)
    plt.show()

if __name__ == "__main__":
    main()
