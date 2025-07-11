# Standard Library Imports
from pathlib import Path
from typing import Union, Optional, Tuple
import os
import logging
import shutil
import math

# Third-Party Imports
import numpy as np
import pandas as pd
import yaml
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Local Imports
from .parser import ClariusTarUnpacker, ClariusParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
        
class ClariusDataUnpacker:
    """
    A class to handle unpacking and processing of Clarius ultrasound data.
    
    This class provides functionality to extract, rename, and process Clarius ultrasound
    data files from tar archives. It supports both single sample and multiple samples
    processing modes.
    """
    
    def __init__(self):
        """Initialize the ClariusDataUnpacker with basic logging configuration."""
        pass
        
    def _cleanup_original_files(self, folder_path: str) -> None:
        """
        Removes original timestamp-based files from an extracted folder,
        keeping only the renamed files (ones with 'large' or 'small' in their names).
        
        Args:
            folder_path (str): Path to the extracted folder to clean up
        """
        try:
            # Get all files in the folder
            files = os.listdir(folder_path)
            
            # Count files to delete
            deleted_count = 0
            
            # Remove original files (ones without 'large' or 'small' in their names)
            for file_name in files:
                if 'large' not in file_name and 'small' not in file_name:
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logging.error(f"Failed to delete {file_path}: {e}")
                        
            if deleted_count > 0:
                logging.info(f"Removed {deleted_count} original files from {folder_path}")
                
        except Exception as e:
            logging.error(f"Error cleaning up folder {folder_path}: {e}")

    def _rename_clarius_files(self, unpacker: ClariusTarUnpacker) -> bool:
        """
        Renames Clarius files based on device type (C3/L15) and size comparison.
        
        Args:
            unpacker (ClariusTarUnpacker): An instance of ClariusTarUnpacker that has already
                                          extracted and processed files.
            
        Returns:
            bool: True if renaming was successful, False otherwise.
        """
        logging.info("Starting to rename Clarius files")
        
        try:
            if not unpacker or not hasattr(unpacker, 'extracted_folders_path_list'):
                logging.error("Invalid unpacker instance or no extracted folders available")
                return False
            
            # Temporary storage for all devices and their line counts
            device_info = []

            # Collect device data and line counts from all folders
            for folder_path in unpacker.extracted_folders_path_list:
                logging.info(f"Processing folder: {folder_path}")
                yaml_files = [f for f in os.listdir(folder_path) if f.endswith(("rf.yml", "rf.yaml"))]
                
                if not yaml_files:
                    logging.warning(f"No YAML files found in {folder_path}")
                    continue
                    
                for yaml_file in yaml_files:
                    yaml_file_path = os.path.join(folder_path, yaml_file)
                    try:
                        with open(yaml_file_path, 'r') as file:
                            yaml_data = yaml.safe_load(file)

                        if not yaml_data:
                            logging.warning(f"Empty or invalid YAML data in {yaml_file_path}")
                            continue
                            
                        if 'size' not in yaml_data or 'number of lines' not in yaml_data['size']:
                            logging.warning(f"Missing required fields in {yaml_file_path}")
                            continue
                            
                        device_name = "C3" if 'probe' in yaml_data and 'radius' in yaml_data['probe'] else "L15"
                        number_of_lines = yaml_data['size']['number of lines']
                        device_info.append((folder_path, device_name, number_of_lines))
                        logging.info(f"Detected device {device_name} with {number_of_lines} lines")
                        
                    except Exception as e:
                        logging.error(f"Failed to read YAML file {yaml_file_path}: {e}")

            if not device_info:
                logging.error("No valid device information was found in any YAML files")
                return False

            device_info.sort(key=lambda x: x[2], reverse=True)
            largest_number_of_lines = device_info[0][2]

            copied_count = 0
            for folder_path, device_name, number_of_lines in device_info:
                size_label = "large" if number_of_lines == largest_number_of_lines else "small"
                logging.info(f"Processing {device_name} files ({size_label}) in {folder_path}")

                files = os.listdir(folder_path)
                for file_name in files:
                    if "large" not in file_name and "small" not in file_name:
                        old_file_path = os.path.join(folder_path, file_name)
                        file_base_name = os.path.basename(old_file_path)
                        file_name_without_extension, file_extension = os.path.splitext(file_base_name)

                        if '.raw' in file_name_without_extension:
                            parts = file_name_without_extension.split('.raw')
                            file_name_without_extension = parts[0]
                            file_extension = '.raw' + file_extension

                        new_file_name = f"{device_name}_{size_label}_" + file_name_without_extension.split("_", 1)[-1] + file_extension
                        new_file_path = os.path.join(folder_path, new_file_name)

                        try:
                            shutil.copy2(old_file_path, new_file_path)
                            copied_count += 1
                        except Exception as e:
                            logging.error(f"Failed to copy {old_file_path} to {new_file_path}: {e}")

            logging.info("Cleaning up original files...")
            for folder_path, _, _ in device_info:
                self._cleanup_original_files(folder_path)

            logging.info(f"Successfully completed renaming {copied_count} files and cleaned up original files")
            return True
            
        except Exception as e:
            logging.error(f"Error occurred during renaming: {e}")
            return False

    def _determine_probe_type(self, raw_file: Path) -> str:
        """
        Determine the probe type (C3_large, C3_small, L15_large, L15_small) from the raw file.
        
        Args:
            raw_file (Path): Path to the raw file to analyze
            
        Returns:
            str: Probe type identifier or empty string if cannot be determined
        """
        try:
            yaml_path = raw_file.parent / (raw_file.stem + '.yml')
            if not yaml_path.exists():
                yaml_path = raw_file.parent / (raw_file.stem + '.yaml')
            
            if not yaml_path.exists():
                logging.warning(f"No YAML file found for {raw_file}")
                return ""
                
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            if not yaml_data:
                logging.warning(f"Empty or invalid YAML data in {yaml_path}")
                return ""
                
            if 'size' not in yaml_data or 'number of lines' not in yaml_data['size']:
                logging.warning(f"Missing required fields in {yaml_path}")
                return ""
                
            device_name = "C3" if 'probe' in yaml_data and 'radius' in yaml_data['probe'] else "L15"
            number_of_lines = yaml_data['size']['number of lines']
            
            all_yaml_files = list(raw_file.parent.glob("*rf.yml")) + list(raw_file.parent.glob("*rf.yaml"))
            max_lines = max(
                yaml.safe_load(open(f))['size']['number of lines']
                for f in all_yaml_files
                if f.exists()
            )
            
            size_label = "large" if number_of_lines >= max_lines else "small"
            probe_type = f"{device_name}_{size_label}"
            
            logging.info(f"Determined probe type {probe_type} for {raw_file.name} (lines: {number_of_lines})")
            return probe_type
                
        except Exception as e:
            logging.error(f"Error determining probe type for {raw_file}: {str(e)}")
            return ""

    def cleanup_extracted_folders(self, samples_path: Union[str, Path] = "data/samples") -> None:
        """
        Clean up all folders ending with '_extracted' in the samples directory.
        
        Args:
            samples_path (Union[str, Path]): Path to the samples directory. Defaults to "data/samples"
        """
        samples_path = Path(samples_path)
        logging.debug(f"Cleaning up extracted folders in: {samples_path}")
        
        if not samples_path.exists():
            logging.error(f"Error: Samples directory '{samples_path}' does not exist")
            return
            
        extracted_folders = []
        for folder in samples_path.glob("**/*_extracted"):
            if folder.is_dir():
                extracted_folders.append(folder)
        
        if not extracted_folders:
            logging.info("No extracted folders found to clean up")
            return
        
        logging.info(f"Found {len(extracted_folders)} extracted folder(s) to clean up")
            
        for folder in extracted_folders:
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted: {folder}")
            except Exception as e:
                logging.error(f"Error deleting {folder}: {str(e)}")
        
        logging.info(f"Cleanup complete. Removed {len(extracted_folders)} extracted folder(s)")

    def _delete_excessive_extracted_folders(self, sample_folder_path: Union[str, Path]) -> None:
        """
        Delete extracted folders that contain duplicate renamed files.
        Only keeps the first occurrence of each renamed file pattern (C3_large, C3_small, etc.).
        
        Args:
            sample_folder_path (Union[str, Path]): Path to the sample directory containing extracted folders
        """
        seen_files = set()
        
        if isinstance(sample_folder_path, str):
            sample_folder_path = Path(sample_folder_path)
            
        logging.info(f"Checking for excessive extracted folders in: {sample_folder_path}")
        
        for folder in os.listdir(sample_folder_path):
            folder_path = os.path.join(sample_folder_path, folder)
            if os.path.isdir(folder_path) and folder.endswith("extracted"):
                logging.info(f"Checking folder: {folder_path}")
                current_folder_files = set()

                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if any(x in file for x in ["C3_large", "C3_small", "L15_large", "L15_small"]):
                            if file in seen_files:
                                logging.info(f"Duplicate file found: {file} in folder: {folder_path}. Deleting folder.")
                                try:
                                    shutil.rmtree(folder_path)
                                    logging.info(f"Successfully deleted folder: {folder_path}")
                                except Exception as e:
                                    logging.error(f"Error deleting folder {folder_path}: {e}")
                                break
                            else:
                                current_folder_files.add(file)
                                seen_files.add(file)
                else:
                    logging.info(f"Unique files in {folder_path}: {current_folder_files}")

    def _process_extracted_folders_with_parser(self, sample_folder_path: Union[str, Path]) -> None:
        """
        Process each extracted folder with ClariusParser and save rf_no_tgc_raw_data_3d as numpy file.
        
        Args:
            sample_folder_path (Union[str, Path]): Path to the sample directory containing extracted folders
        """
        if isinstance(sample_folder_path, str):
            sample_folder_path = Path(sample_folder_path)
            
        logging.info(f"Processing extracted folders with ClariusParser in: {sample_folder_path}")
        
        for folder in os.listdir(sample_folder_path):
            folder_path = os.path.join(sample_folder_path, folder)
            if os.path.isdir(folder_path) and folder.endswith("extracted"):
                logging.info(f"Processing folder: {folder_path}")
                
                try:
                    rf_raw_files = [f for f in os.listdir(folder_path) 
                                  if f.endswith('rf.raw') and any(x in f for x in ["C3_large", "C3_small", "L15_large", "L15_small"])]
                    env_tgc_yml_files = [f for f in os.listdir(folder_path) if f.endswith('env.tgc.yml')]
                    rf_yml_files = [f for f in os.listdir(folder_path) if f.endswith('rf.yml')]
                    
                    if not (rf_raw_files and env_tgc_yml_files and rf_yml_files):
                        logging.warning(f"Missing required files in {folder_path}")
                        continue
                    
                    rf_raw_path = os.path.join(folder_path, rf_raw_files[0])
                    env_tgc_yml_path = os.path.join(folder_path, env_tgc_yml_files[0])
                    rf_yml_path = os.path.join(folder_path, rf_yml_files[0])
                    
                    parser = ClariusParser(
                        rf_raw_path=rf_raw_path,
                        env_tgc_yml_path=env_tgc_yml_path,
                        rf_yml_path=rf_yml_path,
                        visualize=False,
                        use_tgc=False
                    )
                    
                    rf_data = parser.rf_no_tgc_raw_data_3d
                    rf_delay_samples = parser.rf_yml_obj.rf_delay_samples
                    full_depth_mm = parser.full_depth_mm
                    
                    rf_raw_name = os.path.splitext(rf_raw_files[0])[0]
                    output_npy_filename = os.path.join(folder_path, f"{rf_raw_name}_no_tgc.npy")
                    output_excel_filename = os.path.join(folder_path, f"{rf_raw_name}_delay_samples.xlsx")
                    
                    np.save(output_npy_filename, rf_data)
                    logging.info(f"Saved RF data to: {output_npy_filename}")
                    
                    df = pd.DataFrame({'rf_delay_samples': [rf_delay_samples]})
                    df.to_excel(output_excel_filename, index=False)
                    logging.info(f"Saved delay samples to: {output_excel_filename}")
                    
                except Exception as e:
                    logging.error(f"Error processing folder {folder_path}: {e}")

    def unpack_data(self, path: Union[str, Path], extraction_mode: str = "multiple_samples") -> Optional[ClariusTarUnpacker]:
        """
        Unpacks Clarius ultrasound data from tar files and processes them.
        
        Args:
            path (Union[str, Path]): Path to the directory containing the tar files.
                                    For single_sample mode: path to specific sample directory
                                    For multiple_samples mode: path to directory containing multiple sample folders
            extraction_mode (str): Mode of extraction. Options:
                                  - "single_sample": Process a single sample directory
                                  - "multiple_samples": Process multiple sample directories (default)
            
        Returns:
            Optional[ClariusTarUnpacker]: Instance of ClariusTarUnpacker if successful, None if failed
        """
        logging.info(f"Starting Clarius data unpacking process in {extraction_mode} mode")
        logging.info(f"Target path: {path}")
        
        self.cleanup_extracted_folders(path)
        
        if isinstance(path, str):
            path = Path(path)
            logging.debug(f"Converted string path to Path object: {path}")
        
        if not path.exists():
            logging.error(f"The specified path does not exist: {path}")
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        
        valid_modes = ["single_sample", "multiple_samples"]
        if extraction_mode not in valid_modes:
            logging.error(f"Invalid extraction mode: {extraction_mode}. Must be one of: {valid_modes}")
            raise ValueError(f"Invalid extraction mode. Must be one of: {valid_modes}")
        
        try:
            logging.info("Creating ClariusTarUnpacker instance...")
            
            if extraction_mode == "multiple_samples":
                all_extracted_folders = []
                
                for sample_dir in path.iterdir():
                    if sample_dir.is_dir():
                        logging.info(f"\nProcessing sample directory: {sample_dir}")
                        sample_unpacker = ClariusTarUnpacker(
                            path=str(sample_dir),
                            extraction_mode="single_sample"
                        )
                        all_extracted_folders.extend(sample_unpacker.extracted_folders_path_list)
                        temp_unpacker = type('TempUnpacker', (), {'extracted_folders_path_list': sample_unpacker.extracted_folders_path_list})()
                        logging.info(f"Renaming files for sample: {sample_dir.name}")
                        self._rename_clarius_files(temp_unpacker)
                        
                        logging.info(f"Cleaning up excessive extracted folders for sample: {sample_dir.name}")
                        self._delete_excessive_extracted_folders(sample_dir)
                        
                        logging.info(f"Processing extracted folders with ClariusParser for sample: {sample_dir.name}")
                        self._process_extracted_folders_with_parser(sample_dir)
                
                unpacker = type('TempUnpacker', (), {'extracted_folders_path_list': all_extracted_folders})()
                return unpacker
                
            else:  # single_sample mode
                unpacker = ClariusTarUnpacker(
                    path=str(path),
                    extraction_mode=extraction_mode
                )
                logging.info("Renaming Clarius files...")
                self._rename_clarius_files(unpacker)
                
                logging.info("Cleaning up excessive extracted folders...")
                self._delete_excessive_extracted_folders(path)
                
                logging.info("Processing extracted folders with ClariusParser...")
                self._process_extracted_folders_with_parser(path)
                
                return unpacker
            
        except Exception as e:
            logging.error(f"Error during unpacking: {str(e)}", exc_info=True)
            return None




class SingleSampleData:
    """
    A class to handle reading of numpy data files and delay samples from Excel files in extracted folders.
    
    This class provides functionality to read numpy files containing ultrasound data
    and corresponding delay samples from Excel files based on specified device type and size.
    """
    
    def __init__(self, 
                 sample_folder_path: Union[str, Path],
                 roi_file_path: Union[str, Path]):
        """
        Initialize the Data object with specified parameters.
        
        Args:
            sample_folder_path (Union[str, Path]): Path to the sample folder containing extracted data
            roi_file_path (Union[str, Path]): Path to the ROI Excel file
        """
        # Convert paths to Path objects if string
        self.sample_folder_path = Path(sample_folder_path)
        self.roi_file_path = Path(roi_file_path)
        
        # C3 probe parameters
        self.device = 'C3'
        self.size = 'large'
        self.sampling_frequency = 15e6
        self.freq_band = [1e6, 6e6]
        self.center_frequency = 2.5e6
                        
        # Extract sample name from path
        self.sample_name = self.sample_folder_path.name
        
        # Initialize data attributes
        self.data_3d = None
        self.data_3d_phantom = None  # Store phantom data
        self.delay_samples = None
        self.full_depth_cm = None
        self.roi_data = None
        self.data_3d_roi_normal = None  
        self.data_3d_roi_unnormal = None 
        self.data_3d_phantom_roi = None  # Store phantom data cut with ROI
        self.data_3d_roi_unnormal_ac_fix_alpha = None
        self.data_3d_roi_unnormal_ac_calculated_alpha = None
        self.full_depth_cm_roi = None  # Store ROI-cut version of full_depth_cm
        
        # Read data
        self.__run()
            
    def __run(self):
        
        # read data
        self.read_extracted_folder()
        self.create_depth_array()
        self.read_roi_data()
        self.correct_roi_data()
        self.cut_data_based_on_roi()
        self.read_phantom_numpy()
        self.cut_phantom_data_based_on_roi()
              
        # visualize	
        # self.visualize_signal_with_fft(self.data_3d, label='Original full signal')
        # self.visualize_signal_with_fft(self.data_3d_phantom, label='Phantom full signal')
        # self.visualize_signal_with_fft(self.data_3d_roi_unnormal, label='ROI unnormal tissue signal')
        # self.visualize_signal_with_fft(self.data_3d_roi_normal, label='ROI normal tissue signal')
        # self.visualize_signal_with_fft(self.data_3d_phantom_roi, label='Phantom ROI signal')
        
    def create_depth_array(self):
        """
        Create depth array based on sampling frequency and shape of data_3d.
        Uses speed of sound in tissue (1540 m/s) to convert time to distance.
        Returns depth in centimeters.
        """
        # Speed of sound in tissue (m/s)
        speed_of_sound = 1540
        
        # Calculate time array in seconds
        time_array = np.arange(0, self.data_3d.shape[1]) / self.sampling_frequency
        
        # Calculate depth using speed of sound
        # Divide by 2 because of round trip time (pulse-echo)
        # Multiply by 100 to convert meters to centimeters
        self.full_depth_cm = (time_array * speed_of_sound / 2) * 100
        
        logging.info(f"Created depth array with range: {self.full_depth_cm.min():.2f} to {self.full_depth_cm.max():.2f} cm")
        
    def read_roi_data(self):
        """Read ROI data from the specified Excel file."""
        try:
            if not self.roi_file_path.exists():
                logging.warning(f"No ROI file found at: {self.roi_file_path}")
                return
                
            logging.info(f"Loading ROI data from: {self.roi_file_path}")
            
            # Read the Excel file
            self.roi_data = pd.read_excel(self.roi_file_path)
            logging.info(f"Loaded ROI data with shape: {self.roi_data.shape}")
            
        except Exception as e:
            logging.error(f"Error reading ROI data: {str(e)}")
            raise
            
    def read_extracted_folder(self):
        """Read numpy data files and delay samples from Excel files in the extracted folder."""
        try:
            # Find all extracted folders
            extracted_folders = [f for f in self.sample_folder_path.glob("*_extracted") if f.is_dir()]
            
            if not extracted_folders:
                raise FileNotFoundError(f"No extracted folders found in {self.sample_folder_path}")
            
            # Process each extracted folder
            for folder in extracted_folders:
                # Look for numpy files matching the device type and size and ending with no_tgc.npy
                pattern = f"{self.device}_{self.size}_*_no_tgc.npy"
                numpy_files = list(folder.glob(pattern))
                
                if numpy_files:
                    # Load the first matching numpy file
                    numpy_file = numpy_files[0]
                    logging.info(f"Loading data from: {numpy_file}")
                    
                    # Load the numpy data
                    self.data_3d = np.load(numpy_file)
                    logging.info(f"Loaded data shape: {self.data_3d.shape}")
                    
                    # Look for corresponding Excel file with delay samples
                    excel_pattern = f"{self.device}_{self.size}_*_delay_samples.xlsx"
                    excel_files = list(folder.glob(excel_pattern))
                    
                    if excel_files:
                        # Load the first matching Excel file
                        excel_file = excel_files[0]
                        logging.info(f"Loading delay samples from: {excel_file}")
                        
                        # Read delay samples from Excel
                        df = pd.read_excel(excel_file)
                        if 'rf_delay_samples' in df.columns:
                            self.delay_samples = df['rf_delay_samples'].iloc[0]
                            logging.info(f"Loaded delay samples: {self.delay_samples}")
                        else:
                            logging.warning("No rf_delay_samples column found in Excel file")
                    else:
                        logging.warning(f"No delay samples Excel file found matching pattern: {excel_pattern}")
                    
                    break  # Stop after finding first matching file set
                    
            if self.data_3d is None:
                raise FileNotFoundError(f"No matching numpy files found in extracted folders for pattern: {pattern}")
                            
        except Exception as e:
            logging.error(f"Error reading data: {str(e)}")
            raise

    def correct_roi_data(self):
        """
        Add delay samples value to h1, h2, h1_new, h2_new columns in roi_data.
        """
        try:
            if self.roi_data is None:
                logging.warning("No ROI data available to correct")
                return
                
            if self.delay_samples is None:
                logging.warning("No delay samples available for correction")
                return
                
            # Columns to correct
            columns_to_correct = ['h1', 'h2', 'h1_new', 'h2_new']
            
            # Check which columns exist in the DataFrame
            existing_columns = [col for col in columns_to_correct if col in self.roi_data.columns]
            
            if not existing_columns:
                logging.warning(f"None of the target columns {columns_to_correct} found in ROI data")
                return
                
            logging.info(f"Adding delay samples ({self.delay_samples}) to columns: {existing_columns}")
            
            # Add delay samples to each column
            for column in existing_columns:
                self.roi_data[column] = self.roi_data[column] + self.delay_samples
                
            logging.info("Successfully corrected ROI data with delay samples")
            
        except Exception as e:
            logging.error(f"Error correcting ROI data: {str(e)}")
            raise

    def cut_data_based_on_roi(self):
        """
        Cut the 3D ultrasound data based on ROI information.
        This method uses both normal and unnormal ROI data to extract portions of the 3D data array.
        The ROI data contains both sets of boundaries:
        
        Normal (with delay samples):
        - h1_new, h2_new: corrected height boundaries (applied to shape[1])
        - v1_new, v2_new: corrected vertical boundaries (applied to shape[0])
        
        Unnormal (original):
        - h1, h2: original height boundaries (applied to shape[1])
        - v1, v2: original vertical boundaries (applied to shape[0])
        
        The cut data is stored in:
        - self.data_3d_roi_normal: using normal boundaries
        - self.data_3d_roi_unnormal: using unnormal boundaries
        - self.full_depth_mm_roi: cut version of full_depth_mm using h1, h2
        
        The entire temporal dimension (shape[2]) is preserved in both cases.
        """
        try:
            if self.data_3d is None:
                logging.warning("No 3D data available to cut")
                return
                
            if self.roi_data is None or self.roi_data.empty:
                logging.warning("No ROI data available for cutting")
                return
                
            # Check if required columns exist for both normal and unnormal
            required_columns = ['h1_new', 'h2_new', 'v1_new', 'v2_new', 'h1', 'h2', 'v1', 'v2']
            missing_columns = [col for col in required_columns if col not in self.roi_data.columns]
            
            if missing_columns:
                logging.error(f"Missing required columns in ROI data: {missing_columns}")
                return
                
            # Log data shapes for debugging
            logging.info(f"Original data shape: {self.data_3d.shape}")
            
            try:
                # Get normal ROI boundaries from the first row
                h1_normal = int(self.roi_data['h1_new'].iloc[0])  # Height start (applied to shape[1])
                h2_normal = int(self.roi_data['h2_new'].iloc[0])  # Height end (applied to shape[1])
                v1_normal = int(self.roi_data['v1_new'].iloc[0])  # Vertical start (applied to shape[0])
                v2_normal = int(self.roi_data['v2_new'].iloc[0])  # Vertical end (applied to shape[0])
                
                # Get unnormal ROI boundaries from the first row
                h1_unnormal = int(self.roi_data['h1'].iloc[0])  # Height start (applied to shape[1])
                h2_unnormal = int(self.roi_data['h2'].iloc[0])  # Height end (applied to shape[1])
                v1_unnormal = int(self.roi_data['v1'].iloc[0])  # Vertical start (applied to shape[0])
                v2_unnormal = int(self.roi_data['v2'].iloc[0])  # Vertical end (applied to shape[0])
                
                logging.info(f"Using normal ROI boundaries: v1={v1_normal}, v2={v2_normal}, h1={h1_normal}, h2={h2_normal}")
                logging.info(f"Using unnormal ROI boundaries: v1={v1_unnormal}, v2={v2_unnormal}, h1={h1_unnormal}, h2={h2_unnormal}")
                
                # Validate normal ROI boundaries
                if h1_normal >= h2_normal or v1_normal >= v2_normal:
                    logging.error(f"Invalid normal ROI boundaries: v1={v1_normal}, v2={v2_normal}, h1={h1_normal}, h2={h2_normal}")
                    return
                    
                # Validate unnormal ROI boundaries
                if h1_unnormal >= h2_unnormal or v1_unnormal >= v2_unnormal:
                    logging.error(f"Invalid unnormal ROI boundaries: v1={v1_unnormal}, v2={v2_unnormal}, h1={h1_unnormal}, h2={h2_unnormal}")
                    return
                    
                # Validate against data dimensions
                if v2_normal > self.data_3d.shape[0] or h2_normal > self.data_3d.shape[1]:
                    logging.error(f"Normal ROI boundaries exceed data dimensions. "
                                f"Data shape: {self.data_3d.shape}, ROI: v2={v2_normal}, h2={h2_normal}")
                    return
                    
                if v2_unnormal > self.data_3d.shape[0] or h2_unnormal > self.data_3d.shape[1]:
                    logging.error(f"Unnormal ROI boundaries exceed data dimensions. "
                                f"Data shape: {self.data_3d.shape}, ROI: v2={v2_unnormal}, h2={h2_unnormal}")
                    return
                
                # Cut the data for all frames using normal boundaries
                self.data_3d_roi_normal = self.data_3d[v1_normal:v2_normal, h1_normal:h2_normal, :]
                
                # Cut the data for all frames using unnormal boundaries
                self.data_3d_roi_unnormal = self.data_3d[v1_unnormal:v2_unnormal, h1_unnormal:h2_unnormal, :]
                
                # Cut the full_depth_mm array using unnormal boundaries (h1, h2)
                if self.full_depth_cm is not None:
                    self.full_depth_cm_roi = self.full_depth_cm[h1_unnormal:h2_unnormal]
                    logging.info(f"Cut full_depth_cm array to ROI. New size: {len(self.full_depth_cm_roi)}")
                
                # Validate results
                if self.data_3d_roi_normal.size == 0:
                    logging.error("Normal cut resulted in empty data")
                    return
                    
                if self.data_3d_roi_unnormal.size == 0:
                    logging.error("Unnormal cut resulted in empty data")
                    return
                    
                logging.info(f"Successfully cut data to normal ROI. New shape: {self.data_3d_roi_normal.shape}")
                logging.info(f"Successfully cut data to unnormal ROI. New shape: {self.data_3d_roi_unnormal.shape}")
                
            except Exception as e:
                logging.error(f"Error processing ROI boundaries: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error cutting data based on ROI: {str(e)}")
            logging.error(f"Data shapes - data_3d: {self.data_3d.shape if self.data_3d is not None else None}, "
                        f"ROI data rows: {len(self.roi_data) if self.roi_data is not None else None}")
            raise

    def read_phantom_numpy(self):
        """
        Read phantom numpy data from data/phantom directory.
        The phantom data follows the same naming convention as ROI data:
        {device}_{size}_rf.raw.no_tgc.npy
        """
        try:
            # Construct phantom data path
            phantom_dir = Path("data/phantom")
            phantom_file = f"{self.device}_{self.size}_rf.raw.no_tgc.npy"
            phantom_path = phantom_dir / phantom_file
            
            if not phantom_path.exists():
                logging.error(f"Phantom file not found: {phantom_path}")
                return
                
            logging.info(f"Loading phantom data from: {phantom_path}")
            
            # Load phantom data
            self.data_3d_phantom = np.load(phantom_path)
            logging.info(f"Loaded phantom data with shape: {self.data_3d_phantom.shape}")
            
        except Exception as e:
            logging.error(f"Error reading phantom data: {str(e)}")
            raise

    def cut_phantom_data_based_on_roi(self):
        """
        Cut the phantom 3D data based on ROI information.
        Uses only the unnormal (original) ROI boundaries to cut phantom data.
        """
        try:
            if self.data_3d_phantom is None:
                logging.warning("No phantom 3D data available to cut")
                return
                
            if self.roi_data is None or self.roi_data.empty:
                logging.warning("No ROI data available for cutting phantom")
                return
                
            # Log data shapes for debugging
            logging.info(f"Original phantom data shape: {self.data_3d_phantom.shape}")
            
            try:
                # Get unnormal ROI boundaries from the first row
                h1_unnormal = int(self.roi_data['h1'].iloc[0])  # Height start (applied to shape[1])
                h2_unnormal = int(self.roi_data['h2'].iloc[0])  # Height end (applied to shape[1])
                v1_unnormal = int(self.roi_data['v1'].iloc[0])  # Vertical start (applied to shape[0])
                v2_unnormal = int(self.roi_data['v2'].iloc[0])  # Vertical end (applied to shape[0])
                
                logging.info(f"Using unnormal ROI boundaries for phantom: v1={v1_unnormal}, v2={v2_unnormal}, h1={h1_unnormal}, h2={h2_unnormal}")
                
                # Validate unnormal ROI boundaries
                if h1_unnormal >= h2_unnormal or v1_unnormal >= v2_unnormal:
                    logging.error(f"Invalid unnormal ROI boundaries for phantom: v1={v1_unnormal}, v2={v2_unnormal}, h1={h1_unnormal}, h2={h2_unnormal}")
                    return
                    
                # Validate against phantom data dimensions
                if v2_unnormal > self.data_3d_phantom.shape[0] or h2_unnormal > self.data_3d_phantom.shape[1]:
                    logging.error(f"Unnormal ROI boundaries exceed phantom data dimensions. "
                                f"Data shape: {self.data_3d_phantom.shape}, ROI: v2={v2_unnormal}, h2={h2_unnormal}")
                    return
                
                # Cut the phantom data for all frames using unnormal boundaries
                self.data_3d_phantom_roi = self.data_3d_phantom[v1_unnormal:v2_unnormal, h1_unnormal:h2_unnormal, :]
                
                # Validate results
                if self.data_3d_phantom_roi.size == 0:
                    logging.error("ROI cut of phantom resulted in empty data")
                    return
                    
                logging.info(f"Successfully cut phantom data to ROI. New shape: {self.data_3d_phantom_roi.shape}")
                
            except Exception as e:
                logging.error(f"Error processing ROI boundaries for phantom: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error cutting phantom data based on ROI: {str(e)}")
            logging.error(f"Data shapes - phantom_data: {self.data_3d_phantom.shape if self.data_3d_phantom is not None else None}, "
                        f"ROI data rows: {len(self.roi_data) if self.roi_data is not None else None}")
            raise
 
    def visualize_signal_with_fft(self, data: np.ndarray, label: str = ''):
        """
        Visualize a signal and its FFT spectrum side by side.
        Uses the central line from the first frame of the 3D data.

        Args:
            data (np.ndarray): 3D data array to visualize
            label (str): Label for the plot title
        """
        try:
            # Get first frame and central line
            first_frame = data[:, :, 0]
            central_line_idx = first_frame.shape[0] // 2
            signal = first_frame[central_line_idx, :]

            # Calculate time axis in microseconds
            n = len(signal)
            time = np.arange(n) / self.sampling_frequency
            time_microseconds = time * 1e6  # Convert to microseconds

            # Calculate FFT
            fft_result = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(n, 1/self.sampling_frequency)
            
            # Convert frequency to MHz and get positive frequencies
            fft_freq_mhz = fft_freq / 1e6
            positive_freq_mask = fft_freq_mhz >= 0
            freq_positive = fft_freq_mhz[positive_freq_mask]
            fft_magnitude_positive = np.abs(fft_result)[positive_freq_mask]

            # Create figure
            plt.figure(figsize=(14, 4))

            # Plot the original signal
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
            plt.plot(time_microseconds, signal, color='blue')
            plt.title(f'1D Signal Plot - {label}')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amplitude')
            plt.grid(True)

            # Plot the positive FFT
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            plt.plot(freq_positive, fft_magnitude_positive, color='green')
            plt.title(f'FFT of Signal - {label}')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            logging.info(f"Visualized signal and FFT for {label}")

        except Exception as e:
            logging.error(f"Error in signal visualization: {str(e)}")
            raise

    def set_ac_signal_with_constant_alpha(self,
                                          alpha: float,
                                          window: str,
                                          nperseg: int,
                                          noverlap: int,
                                          visualize: bool = False):
        """
        Set the AC signal with a constant alpha by applying attenuation correction to the ROI data.
        This method uses STFT to apply depth and frequency dependent attenuation correction.
        """
        def visualize_correction_example(original_signal, corrected_signal, depth_cm):
            """
            Visualize example of signal before and after attenuation correction.
            
            Args:
                original_signal (np.ndarray): Original signal
                corrected_signal (np.ndarray): Corrected signal
                depth_cm (np.ndarray): Depth array in cm
            """
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot original signal
                plt.subplot(2, 1, 1)
                plt.plot(depth_cm, original_signal)
                plt.title('Original Signal')
                plt.xlabel('Depth (cm)')
                plt.ylabel('Amplitude')
                plt.grid(True)
                
                # Plot corrected signal
                plt.subplot(2, 1, 2)
                plt.plot(depth_cm, corrected_signal)
                plt.title('Attenuation Corrected Signal')
                plt.xlabel('Depth (cm)')
                plt.ylabel('Amplitude')
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
                
                logging.info("Visualized correction example")
                
            except Exception as e:
                logging.error(f"Error visualizing correction: {str(e)}")

        try:
            # Parameters
            speed_of_sound = 1540  # m/s - typical value for soft tissue
            
            # Get ROI data and reshape if needed
            roi_data = self.data_3d_roi_unnormal[:,:,0]  # Get first frame
            
            # Get depth array
            depth_cm = self.full_depth_cm_roi
            
            logging.info("Starting attenuation correction with constant alpha")
            logging.info(f"Alpha: {alpha} dB/cm/MHz")
            logging.info(f"Speed of sound: {speed_of_sound} m/s")
            logging.info(f"ROI data shape: {roi_data.shape}")
            
            # Process each line in the ROI data
            corrected_data = np.zeros_like(roi_data)
            
            for line_idx in range(roi_data.shape[0]):
                # Get current line
                signal = roi_data[line_idx, :]
                
                # Compute STFT
                f, t, Zxx = stft(signal, 
                               fs=self.sampling_frequency,
                               window=window,
                               nperseg=nperseg,
                               noverlap=noverlap)
                
                # Convert frequency to MHz
                f_MHz = f / 1e6
                
                # Use the actual depth values for attenuation correction
                # Interpolate depth array to match STFT time points if needed
                if len(t) != len(depth_cm):
                    depth_line = np.interp(
                        np.linspace(0, 1, len(t)),  # New points
                        np.linspace(0, 1, len(depth_cm)),  # Original points
                        depth_cm  # Original values
                    )
                else:
                    depth_line = depth_cm
                
                # Calculate attenuation factors
                attenuation_factors = np.exp(
                    -alpha * np.abs(f_MHz[:, None]) * depth_line[None, :] / (20 * np.log10(math.e))
                )
                
                # Calculate deattenuation factors
                deattenuation_factors = 1 / attenuation_factors
                
                # Apply deattenuation correction
                corrected_Zxx = Zxx * deattenuation_factors
                
                # Reconstruct signal using ISTFT
                _, corrected_signal = istft(corrected_Zxx,
                                         fs=self.sampling_frequency,
                                         window='hann',
                                         nperseg=64,
                                         noverlap=32)
                
                # Trim signal to match original size
                trim_size = (len(corrected_signal) - len(signal)) // 2
                corrected_signal = corrected_signal[trim_size:len(corrected_signal) - trim_size]
                
                # Store corrected signal
                corrected_data[line_idx, :] = corrected_signal
                
                if line_idx % 10 == 0:  # Log progress every 10 lines
                    logging.info(f"Processed {line_idx}/{roi_data.shape[0]} lines")
            
            # Update the ROI data with corrected data
            self.data_3d_roi_unnormal_ac_fix_alpha = corrected_data[:,:,np.newaxis]  # Add frame dimension
            
            logging.info("Completed attenuation correction")
            logging.info(f"Corrected data shape: {self.data_3d_roi_unnormal_ac_fix_alpha.shape}")
            
            # Optional: Visualize a sample line before and after correction
            if visualize:
                visualize_correction_example(roi_data[roi_data.shape[0]//2, :],
                                                corrected_data[corrected_data.shape[0]//2, :],
                                                depth_cm)
            
        except Exception as e:
            logging.error(f"Error in attenuation correction: {str(e)}")
            raise
            
    def set_ac_signal_with_calculated_alpha(self,
                                      window: str = 'hann',
                                      nperseg: int = 64,
                                      noverlap: int = 32,
                                      visualize: bool = False):
        """
        Set the AC signal with calculated alpha by applying attenuation correction to the ROI data.
        This method calculates the attenuation coefficient at the central frequency (2.5 MHz)
        and applies the correction across all frequencies.
        
        Args:
            window (str): Window function to use for STFT
            nperseg (int): Length of each segment for STFT
            noverlap (int): Number of points to overlap between segments
            visualize (bool): Whether to visualize the correction process
        """
        try:
            # Parameters
            speed_of_sound = 1540  # m/s - typical value for soft tissue
            center_freq = 2.5e6  # Central frequency in Hz (2.5 MHz)
            
            # Get ROI data and reshape if needed
            roi_data = self.data_3d_roi_unnormal[:,:,0]  # Get first frame
            
            # Get depth array directly from ROI
            depth_cm = self.full_depth_cm_roi
            
            logging.info("Starting attenuation correction with calculated alpha at central frequency")
            logging.info(f"Center frequency: {center_freq/1e6} MHz")
            logging.info(f"Speed of sound: {speed_of_sound} m/s")
            logging.info(f"ROI data shape: {roi_data.shape}")
            
            # Process each line in the ROI data
            corrected_data = np.zeros_like(roi_data)
            
            for line_idx in range(roi_data.shape[0]):
                # Get current line
                signal = roi_data[line_idx, :]
                
                # Compute STFT
                f, t, Zxx = stft(signal, 
                               fs=self.sampling_frequency,
                               window=window,
                               nperseg=nperseg,
                               noverlap=noverlap)
                
                # Convert frequency to MHz
                f_MHz = f / 1e6
                
                # Find index of frequency closest to 2.5 MHz
                center_freq_idx = np.argmin(np.abs(f - center_freq))
                center_freq_actual = f[center_freq_idx]
                
                logging.debug(f"Using frequency {center_freq_actual/1e6:.2f} MHz (closest to 2.5 MHz)")
                
                # Create depth array for STFT time points if needed
                if len(t) != len(depth_cm):
                    depth_line = np.interp(
                        np.linspace(0, 1, len(t)),  # New points
                        np.linspace(0, 1, len(depth_cm)),  # Original points
                        depth_cm  # Original values
                    )
                else:
                    depth_line = depth_cm
                
                # Create a filtered version of Zxx that retains only the central frequency
                filtered_Zxx = np.zeros_like(Zxx)
                filtered_Zxx[center_freq_idx] = Zxx[center_freq_idx]
                
                # Reconstruct the time-domain signal for the central frequency
                _, time_domain_signal = istft(filtered_Zxx,
                                           fs=self.sampling_frequency,
                                           window=window,
                                           nperseg=nperseg,
                                           noverlap=noverlap)
                
                # Ensure time_domain_signal matches depth_line length
                if len(time_domain_signal) > len(depth_line):
                    trim_size = (len(time_domain_signal) - len(depth_line)) // 2
                    time_domain_signal = time_domain_signal[trim_size:trim_size + len(depth_line)]
                elif len(time_domain_signal) < len(depth_line):
                    # Pad or interpolate if needed
                    time_domain_signal = np.interp(
                        np.linspace(0, 1, len(depth_line)),
                        np.linspace(0, 1, len(time_domain_signal)),
                        time_domain_signal
                    )
                
                # Calculate the logarithmic slope for alpha at central frequency
                log_amplitude = np.log(np.abs(time_domain_signal) + 1e-10)
                slope, intercept, _, _, _ = linregress(depth_line, log_amplitude)
                mu = -slope  # Negate slope to represent attenuation
                
                if visualize and line_idx == roi_data.shape[0]//2:  # Only visualize middle line
                    # Plot log amplitude vs depth for central frequency
                    plt.figure(figsize=(10, 4))
                    plt.plot(depth_line, log_amplitude, label=f'Frequency: {center_freq_actual/1e6:.2f} MHz')
                    regression_line = slope * depth_line + intercept
                    plt.plot(depth_line, regression_line, '--r', label=f'Regression Line (mu={mu:.2f} dB/cm)')
                    plt.title(f'Log Amplitude vs Depth at {center_freq_actual/1e6:.2f} MHz')
                    plt.xlabel('Depth (cm)')
                    plt.ylabel('Log Amplitude')
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                
                # Calculate attenuation factors using the single mu value
                attenuation_factors = np.exp(
                    -mu * np.abs(f_MHz[:, None]) * depth_line[None, :] / (20 * np.log10(math.e))
                )
                
                # Calculate deattenuation factors
                deattenuation_factors = 1 / attenuation_factors
                
                # Apply deattenuation correction
                corrected_Zxx = Zxx * deattenuation_factors
                
                # Reconstruct signal using ISTFT
                _, corrected_signal = istft(corrected_Zxx,
                                         fs=self.sampling_frequency,
                                         window=window,
                                         nperseg=nperseg,
                                         noverlap=noverlap)
                
                # Ensure corrected signal matches original signal length
                if len(corrected_signal) > len(signal):
                    trim_size = (len(corrected_signal) - len(signal)) // 2
                    corrected_signal = corrected_signal[trim_size:trim_size + len(signal)]
                elif len(corrected_signal) < len(signal):
                    # Pad or interpolate if needed
                    corrected_signal = np.interp(
                        np.linspace(0, 1, len(signal)),
                        np.linspace(0, 1, len(corrected_signal)),
                        corrected_signal
                    )
                
                # Store corrected signal
                corrected_data[line_idx, :] = corrected_signal
                
                if line_idx % 10 == 0:  # Log progress every 10 lines
                    logging.info(f"Processed {line_idx}/{roi_data.shape[0]} lines")
                
                # Visualize STFT and signals for middle line
                if visualize and line_idx == roi_data.shape[0]//2:
                    plt.figure(figsize=(12, 10))
                    
                    # Plot original STFT magnitude
                    plt.subplot(4, 1, 1)
                    plt.title('STFT Magnitude of Original Signal')
                    plt.pcolormesh(depth_line, f_MHz, np.log(np.abs(Zxx) + 1e-10), shading='gouraud')
                    plt.ylabel('Frequency (MHz)')
                    plt.colorbar(label='Magnitude')
                    plt.axhline(y=center_freq_actual/1e6, color='r', linestyle='--', 
                              label=f'Central Frequency ({center_freq_actual/1e6:.2f} MHz)')
                    plt.legend()
                    
                    # Plot corrected STFT magnitude
                    plt.subplot(4, 1, 2)
                    plt.title('STFT Magnitude of Corrected Signal')
                    plt.pcolormesh(depth_line, f_MHz, np.log(np.abs(corrected_Zxx) + 1e-10), shading='gouraud')
                    plt.ylabel('Frequency (MHz)')
                    plt.colorbar(label='Magnitude')
                    plt.axhline(y=center_freq_actual/1e6, color='r', linestyle='--')
                    
                    # Plot original signal
                    plt.subplot(4, 1, 3)
                    plt.title('Original Signal')
                    plt.plot(depth_cm, signal)
                    plt.xlabel('Depth (cm)')
                    plt.grid(True)
                    
                    # Plot corrected signal
                    plt.subplot(4, 1, 4)
                    plt.title(f'Corrected Signal (mu={mu:.2f} dB/cm)')
                    plt.plot(depth_cm, corrected_signal)
                    plt.xlabel('Depth (cm)')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
            
            # Update the ROI data with corrected data
            self.data_3d_roi_unnormal_ac_calculated_alpha = corrected_data[:,:,np.newaxis]  # Add frame dimension
            
            logging.info("Completed attenuation correction")
            logging.info(f"Corrected data shape: {self.data_3d_roi_unnormal_ac_calculated_alpha.shape}")
            
        except Exception as e:
            logging.error(f"Error in attenuation correction: {str(e)}")
            raise
            
  
class BSCSingleSample:
    
    def __init__(self,
                 single_sample_object: SingleSampleData,
                 normalization_method: str,
                 window: str,
                 nperseg: int,
                 noverlap: int,
                 alpha: float):
        
        self.single_sample_object = single_sample_object
        self.normalization_method = normalization_method      
        
        # device parameters
        self.freq_band = self.single_sample_object.freq_band
        self.center_frequency = self.single_sample_object.center_frequency
        self.sampling_frequency=self.single_sample_object.sampling_frequency
        
        # stft parameters
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        
        # alpha
        self.alpha = alpha
        
        # Run the BSC calculation
        self.__run()
        
    def __run(self):       

        if self.normalization_method == "normalized_with_phantom":
            self.calculate_bsc(roi_data=self.single_sample_object.data_3d_roi_unnormal[:,:,0],
                                          phantom_data=self.single_sample_object.data_3d_phantom_roi[:,:,0],
                                          window_depth_cm=self.single_sample_object.full_depth_cm_roi)
            
        elif self.normalization_method == "normalized_with_healthy_liver":
            self.calculate_bsc(roi_data=self.single_sample_object.data_3d_roi_unnormal[:,:,0],
                                          phantom_data=self.single_sample_object.data_3d_roi_normal[:,:,0],
                                          window_depth_cm=self.single_sample_object.full_depth_cm_roi)
            
            
        elif self.normalization_method == "normalized_with_constant_alpha":
            self.single_sample_object.set_ac_signal_with_constant_alpha(alpha=self.alpha,
                                                                        window=self.window,
                                                                        nperseg=self.nperseg,
                                                                        noverlap=self.noverlap,
                                                                        visualize=False)
            
            # Get 2D data from 3D array
            ac_data = self.single_sample_object.data_3d_roi_unnormal_ac_fix_alpha
            if len(ac_data.shape) == 3:
                ac_data = ac_data[:,:,0]
            
            self.calculate_bsc(roi_data=ac_data,
                                window_depth_cm=self.single_sample_object.full_depth_cm_roi,
                                use_phantom=False)
            
        elif self.normalization_method == "normalized_with_calculated_alpha":
            self.single_sample_object.set_ac_signal_with_calculated_alpha(
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                visualize=False
            )
            
            # Get 2D data from 3D array
            ac_data = self.single_sample_object.data_3d_roi_unnormal_ac_calculated_alpha
            if len(ac_data.shape) == 3:
                ac_data = ac_data[:,:,0]
            
            self.calculate_bsc(roi_data=ac_data,
                                window_depth_cm=self.single_sample_object.full_depth_cm_roi,
                                use_phantom=False)
            
        else:
            raise ValueError(f"Invalid normalization method: {self.normalization_method}")
            
    def calculate_bsc(self,
                   roi_data: np.ndarray,
                   phantom_data: Optional[np.ndarray] = None,
                   window_depth_cm: np.ndarray = None,
                   use_phantom: bool = True) -> float:
        """
        Calculate the backscatter coefficient (BSC) using the reference phantom method with STFT-based power spectra.
        This method uses Short-Time Fourier Transform for spectral analysis instead of traditional windowing.
        Can operate in two modes: with phantom normalization or without phantom normalization.
        
        Args:
            roi_data (np.ndarray): ROI data array (2D or 3D)
            phantom_data (Optional[np.ndarray]): Phantom data array (2D or 3D). Required if use_phantom=True
            window_depth_cm (np.ndarray): Array of depth values in cm
            use_phantom (bool): Whether to use phantom data for normalization. Defaults to True.
            
        Returns:
            float: Backscatter coefficient of the ROI (1/cm-sr)
        """
        center_frequency = self.center_frequency
        sampling_frequency=self.sampling_frequency
                
        # stft parameters
        window = self.window
        nperseg = self.nperseg
        noverlap = self.noverlap
        
        analysis_method = "with phantom" if use_phantom else "without phantom"
        logging.info(f"Starting BSC calculation {analysis_method} using HybridEcho method")
        logging.info(f"Center frequency: {center_frequency/1e6:.1f} MHz")
        logging.info(f"STFT parameters - Window: {window}, Segment length: {nperseg}, Overlap: {noverlap}")
        
        def compute_stft_power_spec(rf_data: np.ndarray,
                                  sampling_frequency: float) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute the power spectrum using STFT.
            
            Args:
                rf_data (np.ndarray): Input RF data array (2D or 3D)
                sampling_frequency (float): Sampling frequency
                
            Returns:
                Tuple[np.ndarray, np.ndarray]: Frequencies and averaged power spectrum
            """
            # Ensure input is 2D
            if len(rf_data.shape) == 3:
                rf_data = rf_data[:,:,0]  # Take first frame if 3D
            
            logging.debug(f"Computing STFT power spectrum for data shape: {rf_data.shape}")
            
            # Initialize array to store power spectra for each line
            power_spectra = []
            
            # Process each line in the data
            logging.info(f"Processing {rf_data.shape[0]} lines with STFT")
            for line_idx in range(rf_data.shape[0]):
                # Compute STFT for the current line
                f, t, Zxx = stft(rf_data[line_idx], fs=sampling_frequency, window=window,
                               nperseg=nperseg, noverlap=noverlap)
                
                # Calculate power spectrum for this line
                ps_line = np.mean(np.abs(Zxx)**2, axis=1)
                power_spectra.append(ps_line)
            
            # Convert list to numpy array and average across all lines
            power_spectra = np.array(power_spectra)  # Shape: (n_lines, n_frequencies)
            
            logging.info(f"Completed STFT analysis for all lines")
            logging.info(f"Computed power spectrum with shape - Input: {rf_data.shape}, Output: {power_spectra.shape}")
            
            return f, power_spectra

        def plot_power_spectra(f: np.ndarray, ps: np.ndarray, window_depth_cm: np.ndarray, label: str):
            """
            Plot power spectrum based on depth and frequency.
            
            Args:
                f (np.ndarray): Frequency array
                ps (np.ndarray): Power spectrum data
                window_depth_cm (np.ndarray): Depth values in cm
            """
            try:
                # Create depth array matching the number of lines in power spectra
                depths = np.linspace(window_depth_cm[0], window_depth_cm[-1], ps.shape[0])
                
                # Convert frequency to MHz for better visualization
                freq_mhz = f / 1e6
                
                # Create figure
                plt.figure(figsize=(8, 6))
                
                # Plot power spectrum with swapped axes
                plt.pcolormesh(depths, freq_mhz, ps.T, shading='auto', cmap='viridis')
                plt.colorbar(label='Power (dB)')
                plt.title(f'Power Spectrum {label}')
                plt.xlabel('Depth (cm)')
                plt.ylabel('Frequency (MHz)')
                
                plt.tight_layout()
                plt.show()
                
                logging.info("Successfully plotted power spectrum")
                
            except Exception as e:
                logging.error(f"Error plotting power spectrum: {str(e)}")
                
        try:
            # Validate inputs
            if use_phantom and phantom_data is None:
                raise ValueError("phantom_data is required when use_phantom=True")

            # Log input shapes
            logging.info(f"Input ROI data shape: {roi_data.shape}")
            if use_phantom:
                logging.info(f"Input phantom data shape: {phantom_data.shape}")

            # Calculate power spectra using STFT
            logging.info("Calculating power spectra using STFT")
            f, ps_sample = compute_stft_power_spec(
                roi_data, sampling_frequency
            )
            ps_sample = 20 * np.log10(ps_sample)
            
            if use_phantom:
                _, ps_phantom = compute_stft_power_spec(
                    phantom_data, sampling_frequency
                )
                ps_phantom = 20 * np.log10(ps_phantom)
                
                # Calculate signal ratio
                power_ratio_2d = ps_sample / ps_phantom  # Element-wise ratio
                logging.info(f"Central frequency power ratio shape: {power_ratio_2d.shape}")
                
                # Print shapes for debugging
                logging.info(f"Shape of f: {f.shape}")
                logging.info(f"Shape of ps_sample: {ps_sample.shape}")
                logging.info(f"Shape of ps_phantom: {ps_phantom.shape}")
            else:
                # Use sample power spectrum directly when not using phantom
                power_ratio_2d = ps_sample
                logging.info(f"Power spectrum shape: {power_ratio_2d.shape}")
                
                # Print shapes for debugging
                logging.info(f"Shape of f: {f.shape}")
                logging.info(f"Shape of ps_sample: {ps_sample.shape}")
            
            logging.info("Power spectra calculation completed")
                          
            # plot results
            #plot_power_spectra(f, power_ratio_2d, window_depth_cm, label="Power Ratio")
            #plot_power_spectra(f, ps_sample, window_depth_cm, label="ROI")
            #plot_power_spectra(f, ps_phantom, window_depth_cm, label="Phantom")
            
            # compatible depth with bsc
            compatible_depth_with_bsc = np.linspace(window_depth_cm[0], window_depth_cm[-1], power_ratio_2d.shape[0])
            
            # set data
            self.power_ratio_2d = power_ratio_2d
            self.f = f
            self.depth_cm = compatible_depth_with_bsc
           
        except Exception as e:
            logging.error(f"Error calculating BSC with HybridEcho method: {str(e)}")
            return None
    
 


class BSC:
    
    def __init__(self,
                 samples_folder_path: str,
                 result_folder_path: str,
                 roi_folder_path: str,
                 normalization_method: str,
                 window: str,
                 nperseg: int,
                 noverlap: int,
                 alpha: float):
        
        self.samples_folder_path = samples_folder_path
        self.result_folder_path = result_folder_path
        self.roi_folder_path = roi_folder_path
        self.normalization_method = normalization_method
        
        # stft parameters
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        
        # alpha
        self.alpha = alpha
        
        # run
        self.__run()
        
    def save_bsc_results_as_csv(self, bsc_obj):
        """
        Save BSC results to CSV files in a matching folder structure.
        
        Args:
            bsc_obj: BSC object containing results to save
        """
        try:
            # Get sample name from the original sample path
            sample_name = bsc_obj.single_sample_object.sample_name
            
            # Create result directory path mirroring the samples structure
            result_dir = Path(self.result_folder_path) / sample_name
            result_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Saving BSC results for sample {sample_name} to {result_dir}")
            
            # Save power ratio data with 6 decimal places
            power_ratio_path = result_dir / "power_ratio.csv"
            np.savetxt(power_ratio_path, bsc_obj.power_ratio_2d, delimiter=',', fmt='%.6f')
            logging.info(f"Saved power ratio data to {power_ratio_path}")
            
            # Save frequency data in MHz with 3 decimal places
            freq_path = result_dir / "frequencies_MHz.csv"
            freq_mhz = bsc_obj.f / 1e6  # Convert to MHz
            np.savetxt(freq_path, freq_mhz, delimiter=',', fmt='%.3f')
            logging.info(f"Saved frequency data to {freq_path}")
            
            # Save depth data with 3 decimal places
            depth_path = result_dir / "depths_cm.csv"
            np.savetxt(depth_path, bsc_obj.depth_cm, delimiter=',', fmt='%.3f')
            logging.info(f"Saved depth data to {depth_path}")
            
        except Exception as e:
            logging.error(f"Error saving BSC results: {str(e)}")
            raise
        
    def __run(self):
        
        # get sample folder path from sample_folder_path
        samples_folder_path = Path(self.samples_folder_path)
        roi_folder_path = Path(self.roi_folder_path)
        
        # get all subfolders in sample_folder_path
        subfolders = [f for f in samples_folder_path.iterdir() if f.is_dir()]
        
        # Process each sample
        for subfolder in subfolders:
            try:
                logging.info(f"Processing sample: {subfolder.name}")
                
                # Construct ROI file path
                roi_file_path = roi_folder_path / f"{subfolder.name}.xlsx"
                
                # Create SingleSampleData object first
                single_sample_data = SingleSampleData(
                    sample_folder_path=subfolder,
                    roi_file_path=roi_file_path
                )
                
                # Create BSCSingleSample object using the SingleSampleData object
                bsc_single_sample_obj = BSCSingleSample(
                    single_sample_object=single_sample_data,
                    normalization_method=self.normalization_method,
                    window=self.window,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    alpha=self.alpha
                )
                
                # Save BSC results
                self.save_bsc_results_as_csv(bsc_single_sample_obj)
                logging.info(f"Completed processing sample: {subfolder.name}")
                
            except Exception as e:
                logging.error(f"Error processing sample {subfolder.name}: {str(e)}")
                continue  # Continue with next sample even if one fails 
            
            
            
            
            
            