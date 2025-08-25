# Standard Library Imports
from pathlib import Path
from typing import Union, Optional, Tuple, List
import os
import logging
import shutil
import math

# Third-Party Imports
import numpy as np
import pandas as pd
import yaml
from scipy.signal import stft, istft
from scipy.interpolate import RegularGridInterpolator
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
                    # List all files in the folder for debugging
                    all_files = os.listdir(folder_path)
                    logging.info(f"Files in {folder_path}: {all_files}")
                    
                    rf_raw_files = [f for f in os.listdir(folder_path) 
                                  if f.endswith('rf.raw') and any(x in f for x in ["C3_large", "C3_small", "L15_large", "L15_small"])]
                    env_tgc_yml_files = [f for f in os.listdir(folder_path) if f.endswith('env.tgc.yml')]
                    rf_yml_files = [f for f in os.listdir(folder_path) if f.endswith('rf.yml')]
                    
                    logging.info(f"Found RF raw files: {rf_raw_files}")
                    logging.info(f"Found env.tgc.yml files: {env_tgc_yml_files}")
                    logging.info(f"Found rf.yml files: {rf_yml_files}")
                    
                    # Check if any of the required file lists are empty
                    if not rf_raw_files:
                        logging.warning(f"No RF raw files found in {folder_path}")
                        continue
                    if not env_tgc_yml_files:
                        logging.warning(f"No env.tgc.yml files found in {folder_path}")
                        continue
                    if not rf_yml_files:
                        logging.warning(f"No rf.yml files found in {folder_path}")
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

    def _validate_extracted_folder(self, folder_path: Path) -> bool:
        """
        Validates that an extracted folder contains all necessary files.
        
        Args:
            folder_path (Path): Path to the extracted folder to validate
            
        Returns:
            bool: True if folder contains all necessary files, False otherwise
        """
        required_file_patterns = [
            "*_rf.raw.lzo",
            "*_rf.yml"
        ]
        
        missing_files = []
        for pattern in required_file_patterns:
            matching_files = list(folder_path.glob(pattern))
            if not matching_files:
                missing_files.append(pattern)
        
        if missing_files:
            logging.warning(f"Extracted folder {folder_path} is missing required files: {missing_files}")
            return False
        
        logging.info(f"Extracted folder {folder_path} validation successful - all required files present")
        return True

    def _cleanup_incomplete_extracted_folders(self, path: Path) -> None:
        """
        Removes extracted folders that don't contain all necessary files.
        
        Args:
            path (Path): Path to the directory containing extracted folders
        """
        logging.info("Validating extracted folders and removing incomplete ones...")
        
        # Find all extracted folders
        extracted_folders = [f for f in path.iterdir() if f.is_dir() and "extracted" in f.name]
        
        for folder in extracted_folders:
            if not self._validate_extracted_folder(folder):
                try:
                    logging.info(f"Removing incomplete extracted folder: {folder}")
                    shutil.rmtree(folder)
                except Exception as e:
                    logging.error(f"Error removing incomplete folder {folder}: {e}")

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
                        
                        # Validate and cleanup incomplete extracted folders for this sample
                        logging.info(f"Validating extracted folders for sample: {sample_dir.name}")
                        self._cleanup_incomplete_extracted_folders(sample_dir)
                        
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
                
                # Validate and cleanup incomplete extracted folders
                logging.info("Validating extracted folders...")
                self._cleanup_incomplete_extracted_folders(path)
                
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
                 roi_file_path: Union[str, Path],
                 roi_size: str = 'large'):  # Add roi_size parameter with default value
        """
        Initialize the Data object with specified parameters.
        
        Args:
            sample_folder_path (Union[str, Path]): Path to the sample folder containing extracted data
            roi_file_path (Union[str, Path]): Path to the ROI Excel file
            roi_size (str): Size of ROI to use, either 'large' or 'small'. Defaults to 'large'
        """
        # Validate roi_size
        if roi_size.lower() not in ['large', 'small']:
            raise ValueError("roi_size must be either 'large' or 'small'")
            
        # Convert paths to Path objects if string
        self.sample_folder_path = Path(sample_folder_path)
        self.roi_file_path = Path(roi_file_path)
        self.roi_size = roi_size.lower()
        
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
        """Read ROI data from the specified Excel file and sheet based on roi_size."""
        try:
            if not self.roi_file_path.exists():
                logging.warning(f"No ROI file found at: {self.roi_file_path}")
                return
                
            # Select sheet based on roi_size
            sheet_name = 'Large_ROI' if self.roi_size == 'large' else 'Small_ROI'
            logging.info(f"Loading {sheet_name} from: {self.roi_file_path}")
            
            try:
                # Read the specific sheet from Excel file
                self.roi_data = pd.read_excel(self.roi_file_path, sheet_name=sheet_name)
                logging.info(f"Loaded ROI data with shape: {self.roi_data.shape}")
            except ValueError as ve:
                logging.error(f"Sheet '{sheet_name}' not found in {self.roi_file_path}")
                # If specified sheet not found, try to read available sheets
                available_sheets = pd.ExcelFile(self.roi_file_path).sheet_names
                logging.info(f"Available sheets: {available_sheets}")
                if available_sheets:
                    # Use first available sheet as fallback
                    logging.warning(f"Using first available sheet: {available_sheets[0]}")
                    self.roi_data = pd.read_excel(self.roi_file_path, sheet_name=available_sheets[0])
                else:
                    raise ValueError(f"No valid sheets found in {self.roi_file_path}")
            
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
            phantom_file = f"{self.device}_{self.size}_rf_no_tgc.npy"
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
        Processes the full 3D array, applying correction to each frame independently.
        """
        try:
            # Parameters
            speed_of_sound = 1540  # m/s - typical value for soft tissue
            
            # Get ROI data
            roi_data = self.data_3d_roi_unnormal  # Full 3D array
            n_lines, n_samples, n_frames = roi_data.shape
            
            # Get depth array
            depth_cm = self.full_depth_cm_roi
            
            logging.info("Starting attenuation correction with constant alpha")
            logging.info(f"Alpha: {alpha} dB/cm/MHz")
            logging.info(f"Speed of sound: {speed_of_sound} m/s")
            logging.info(f"ROI data shape: {roi_data.shape}")
            
            # Initialize 3D array for corrected data
            corrected_data = np.zeros_like(roi_data)
            
            # Process each frame
            for frame_idx in range(n_frames):
                # Process each line in the current frame
                for line_idx in range(n_lines):
                    # Get current line from current frame
                    signal = roi_data[line_idx, :, frame_idx]
                    
                    # Compute STFT
                    f, t, Zxx = stft(signal, 
                                   fs=self.sampling_frequency,
                                   window=window,
                                   nperseg=nperseg,
                                   noverlap=noverlap)
                    
                    # Convert frequency to MHz
                    f_MHz = f / 1e6
                    
                    # Create depth array for STFT time points
                    depth_line = np.linspace(depth_cm[0], depth_cm[-1], len(t))
                    
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
                                             window=window,
                                             nperseg=nperseg,
                                             noverlap=noverlap)
                    
                    # Ensure corrected signal matches original signal length
                    if len(corrected_signal) > n_samples:
                        # Trim excess samples
                        trim_size = (len(corrected_signal) - n_samples) // 2
                        corrected_signal = corrected_signal[trim_size:trim_size + n_samples]
                    elif len(corrected_signal) < n_samples:
                        # Pad with zeros if needed
                        pad_size = (n_samples - len(corrected_signal)) // 2
                        corrected_signal = np.pad(corrected_signal, (pad_size, n_samples - len(corrected_signal) - pad_size))
                    
                    # Store corrected signal in 3D array
                    corrected_data[line_idx, :, frame_idx] = corrected_signal
                    
            # Update the ROI data with corrected data
            self.data_3d_roi_unnormal_ac_fix_alpha = corrected_data
            
            logging.info("Completed attenuation correction")
            logging.info(f"Corrected data shape: {self.data_3d_roi_unnormal_ac_fix_alpha.shape}")
            
            # Optional: Visualize a sample line from the middle frame before and after correction
            if visualize:
                mid_frame = n_frames // 2
                self.visualize_correction_example(
                    roi_data[n_lines//2, :, mid_frame],
                    corrected_data[n_lines//2, :, mid_frame],
                    depth_cm
                )
            
        except Exception as e:
            logging.error(f"Error in attenuation correction: {str(e)}")
            raise
            
    def visualize_correction_example(self, original_signal: np.ndarray, corrected_signal: np.ndarray, depth_cm: np.ndarray):
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

    def set_ac_signal_with_calculated_alpha(self,
                                      window: str = 'hann',
                                      nperseg: int = 64,
                                      noverlap: int = 32,
                                      visualize: bool = False):
        """
        Set the AC signal with calculated alpha by applying attenuation correction to the ROI data.
        This method calculates the attenuation coefficient at the central frequency (2.5 MHz)
        and applies the correction across all frequencies for each frame in the 3D data.
        
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
            
            # Get ROI data
            roi_data = self.data_3d_roi_unnormal  # Full 3D array
            n_lines, n_samples, n_frames = roi_data.shape
            
            # Get depth array directly from ROI
            depth_cm = self.full_depth_cm_roi
            
            logging.info("Starting attenuation correction with calculated alpha at central frequency")
            logging.info(f"Center frequency: {center_freq/1e6} MHz")
            logging.info(f"Speed of sound: {speed_of_sound} m/s")
            logging.info(f"ROI data shape: {roi_data.shape}")
            
            # Initialize 3D array for corrected data
            corrected_data = np.zeros_like(roi_data)
            
            # Process each frame
            for frame_idx in range(n_frames):
                # Process each line in the current frame
                for line_idx in range(n_lines):
                    # Get current line from current frame
                    signal = roi_data[line_idx, :, frame_idx]
                    
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
                    
                    # Create depth array for STFT time points
                    depth_line = np.linspace(depth_cm[0], depth_cm[-1], len(t))
                    
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
                    if len(corrected_signal) > n_samples:
                        # Trim excess samples
                        trim_size = (len(corrected_signal) - n_samples) // 2
                        corrected_signal = corrected_signal[trim_size:trim_size + n_samples]
                    elif len(corrected_signal) < n_samples:
                        # Pad with zeros if needed
                        pad_size = (n_samples - len(corrected_signal)) // 2
                        corrected_signal = np.pad(corrected_signal, (pad_size, n_samples - len(corrected_signal) - pad_size))
                    
                    # Store corrected signal in 3D array
                    corrected_data[line_idx, :, frame_idx] = corrected_signal
            
            # Update the ROI data with corrected data
            self.data_3d_roi_unnormal_ac_calculated_alpha = corrected_data
            
            logging.info("Completed attenuation correction")
            logging.info(f"Corrected data shape: {self.data_3d_roi_unnormal_ac_calculated_alpha.shape}")
            
            # Optional: Visualize a sample line from the middle frame before and after correction
            if visualize:
                mid_frame = n_frames // 2
                self.visualize_correction_example(
                    roi_data[n_lines//2, :, mid_frame],
                    corrected_data[n_lines//2, :, mid_frame],
                    depth_cm
                )
            
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
            # Calculate BSC for tissue data
            self.energy_dict, self.frequencies = self.calculate_bsc_3d(self.single_sample_object.data_3d_roi_unnormal)
            
            # Calculate BSC for phantom data (using first frame)
            phantom_data_2d = self.single_sample_object.data_3d_phantom_roi[:,:,0]  # Take first frame
            self.energy_dict_phantom, _ = self.calculate_bsc_2d(phantom_data_2d)
            
            # Normalize using phantom reference
            self.energy_dict = self.normalize_3d_with_2d(
                energy_dict_3d=self.energy_dict,
                energy_dict_2d=self.energy_dict_phantom
            )

        elif self.normalization_method == "normalized_with_healthy_liver":
            # Calculate BSC for abnormal tissue data
            self.energy_dict, self.frequencies = self.calculate_bsc_3d(self.single_sample_object.data_3d_roi_unnormal)
            
            # Calculate BSC for normal tissue data (using first frame)
            normal_data_2d = self.single_sample_object.data_3d_roi_normal[:,:,0]  # Take first frame
            self.energy_dict_normal, _ = self.calculate_bsc_2d(normal_data_2d)
            
            # Normalize using normal tissue reference
            self.energy_dict = self.normalize_3d_with_2d(
                energy_dict_3d=self.energy_dict,
                energy_dict_2d=self.energy_dict_normal
            )
            
        elif self.normalization_method == "normalized_with_constant_alpha":
            self.single_sample_object.set_ac_signal_with_constant_alpha(alpha=self.alpha,
                                                                        window=self.window,
                                                                        nperseg=self.nperseg,
                                                                        noverlap=self.noverlap,
                                                                        visualize=False)
                        
            self.energy_dict, self.frequencies = self.calculate_bsc_3d(self.single_sample_object.data_3d_roi_unnormal_ac_fix_alpha)
            
        elif self.normalization_method == "normalized_with_calculated_alpha":
            self.single_sample_object.set_ac_signal_with_calculated_alpha(
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                visualize=False
            )
            
            self.energy_dict, self.frequencies = self.calculate_bsc_3d(self.single_sample_object.data_3d_roi_unnormal_ac_calculated_alpha)
            
        else:
            raise ValueError(f"Invalid normalization method: {self.normalization_method}")
            
    def calculate_bsc_3d(self,
                   roi_data_3d: np.ndarray):
            """
            Calculate the backscatter coefficient (BSC) using the reference phantom method with STFT-based power spectra.
            Computes signal energy at each frequency for each position in the 3D ROI data.
            
            Args:
                roi_data_3d (np.ndarray): Input RF data array (3D) with shape (lines, samples, frames)
            
            Returns:
                Tuple[Dict[float, np.ndarray], np.ndarray]: A tuple containing:
                    - Dictionary mapping frequencies (in MHz) to their corresponding
                      3D energy arrays with shape (lines, time_points, frames)
                    - Array of frequency values in MHz
            """
            center_frequency = self.center_frequency
            sampling_frequency = self.sampling_frequency
                    
            # stft parameters
            window = self.window
            nperseg = self.nperseg
            noverlap = self.noverlap
            
            logging.info(f"Center frequency: {center_frequency/1e6:.1f} MHz")
            logging.info(f"STFT parameters - Window: {window}, Segment length: {nperseg}, Overlap: {noverlap}")
            
            n_lines, n_samples, n_frames = roi_data_3d.shape
            logging.info(f"Computing energy spectrum for 3D data with shape: {roi_data_3d.shape}")
            
            # First compute STFT for one line to get frequency array and time points
            # Use the first line of the first frame as reference
            f, t, Zxx_ref = stft(roi_data_3d[0, :, 0], 
                                fs=sampling_frequency, 
                                window=window,
                                nperseg=nperseg, 
                                noverlap=noverlap)
            
            n_freqs = len(f)
            n_times = Zxx_ref.shape[1]  # Use actual time points from STFT output
            
            logging.info(f"STFT output dimensions - Frequencies: {n_freqs}, Time points: {n_times}")
            
            # Initialize dictionary to store frequency-energy mappings
            energy_dict = {}
            
            # For each frequency, create an energy array with STFT dimensions
            for freq_idx in range(n_freqs):
                current_freq = f[freq_idx] / 1e6  # Convert to MHz
                
                # Initialize energy array for this frequency with STFT dimensions and frames
                energy_array = np.zeros((n_lines, n_times, n_frames), dtype=float)
                
                # Process each line
                for line_idx in range(n_lines):
                    # Process each frame
                    for frame_idx in range(n_frames):
                        # Compute STFT for this line and frame
                        _, _, Zxx = stft(roi_data_3d[line_idx, :, frame_idx],
                                       fs=sampling_frequency,
                                       window=window,
                                       nperseg=nperseg,
                                       noverlap=noverlap)
                        
                        # Extract power at current frequency for all time points
                        energy_array[line_idx, :, frame_idx] = np.abs(Zxx[freq_idx, :])**2
                
                # Add energy array to dictionary with frequency as key
                energy_dict[current_freq] = energy_array
                logging.info(f"Completed energy calculation for frequency {current_freq:.2f} MHz")
            
            logging.info(f"Completed energy computation for all frequencies")
            logging.info(f"Number of frequency bands: {len(energy_dict)}")
            logging.info(f"Each energy array shape: {next(iter(energy_dict.values())).shape}")
            
            # Return results instead of storing as class attributes
            return energy_dict, f / 1e6  # Return dictionary and frequency values in MHz
            
    def calculate_bsc_2d(self,
                        roi_data_2d: np.ndarray):
        """
        Calculate the backscatter coefficient (BSC) using the reference phantom method with STFT-based power spectra.
        Computes signal energy at each frequency for each position in the 2D ROI data.
        
        Args:
            roi_data_2d (np.ndarray): Input RF data array (2D) with shape (lines, samples)
        
        Returns:
            Tuple[Dict[float, np.ndarray], np.ndarray]: A tuple containing:
                - Dictionary mapping frequencies to their corresponding
                  2D energy arrays with shape (lines, time_points)
                - Array of frequency values in MHz
        """
        center_frequency = self.center_frequency
        sampling_frequency = self.sampling_frequency
                
        # stft parameters
        window = self.window
        nperseg = self.nperseg
        noverlap = self.noverlap
        
        logging.info(f"Center frequency: {center_frequency/1e6:.1f} MHz")
        logging.info(f"STFT parameters - Window: {window}, Segment length: {nperseg}, Overlap: {noverlap}")
        
        n_lines, n_samples = roi_data_2d.shape
        logging.info(f"Computing energy spectrum for 2D data with shape: {roi_data_2d.shape}")
        
        # First compute STFT for one line to get frequency array and time points
        # Use the first line as reference
        f, t, Zxx_ref = stft(roi_data_2d[0, :], 
                            fs=sampling_frequency, 
                            window=window,
                            nperseg=nperseg, 
                            noverlap=noverlap)
        
        n_freqs = len(f)
        n_times = Zxx_ref.shape[1]  # Use actual time points from STFT output
        
        logging.info(f"STFT output dimensions - Frequencies: {n_freqs}, Time points: {n_times}")
        
        # Initialize dictionary to store frequency-energy mappings
        energy_dict = {}
        
        # For each frequency, create an energy array with STFT dimensions
        for freq_idx in range(n_freqs):
            current_freq = f[freq_idx] / 1e6  # Convert to MHz
            
            # Initialize energy array for this frequency with STFT dimensions
            energy_array = np.zeros((n_lines, n_times), dtype=float)
            
            # Process each line
            for line_idx in range(n_lines):
                # Compute STFT for this line
                _, _, Zxx = stft(roi_data_2d[line_idx, :],
                               fs=sampling_frequency,
                               window=window,
                               nperseg=nperseg,
                               noverlap=noverlap)
                
                # Extract power at current frequency for all time points
                energy_array[line_idx, :] = np.abs(Zxx[freq_idx, :])**2
            
            # Add energy array to dictionary with frequency as key
            energy_dict[current_freq] = energy_array
            logging.info(f"Completed energy calculation for frequency {current_freq:.2f} MHz")
        
        logging.info(f"Completed energy computation for all frequencies")
        logging.info(f"Number of frequency bands: {len(energy_dict)}")
        logging.info(f"Each energy array shape: {next(iter(energy_dict.values())).shape}")
        
        # Convert frequencies to MHz
        frequencies_mhz = f / 1e6
        
        # Return results
        return energy_dict, frequencies_mhz  # Return dictionary and frequency values in MHz

    def normalize_3d_with_2d(self, energy_dict_3d: dict, energy_dict_2d: dict) -> dict:
        """
        Normalize a 3D energy dictionary using a 2D energy dictionary.
        Each frame in the 3D data is normalized using the same 2D reference data.
        
        Args:
            energy_dict_3d (dict): Dictionary mapping frequencies to 3D energy arrays (lines, time_points, frames)
            energy_dict_2d (dict): Dictionary mapping frequencies to 2D energy arrays (lines, time_points)
            
        Returns:
            dict: Normalized 3D energy dictionary
        """
        normalized_dict = {}
        
        for freq in energy_dict_3d.keys():
            if freq in energy_dict_2d:
                # Get shapes for debugging
                data_3d_shape = energy_dict_3d[freq].shape
                data_2d_shape = energy_dict_2d[freq].shape
                
                logging.info(f"Normalizing frequency {freq:.2f} MHz - 3D shape: {data_3d_shape}, 2D shape: {data_2d_shape}")
                
                # Check if the number of lines and time points match
                if data_3d_shape[0] != data_2d_shape[0] or data_3d_shape[1] != data_2d_shape[1]:
                    #logging.warning(f"Shape mismatch for frequency {freq:.2f} MHz. "
                    #              f"3D: {data_3d_shape}, 2D: {data_2d_shape}. "
                    #              f"Interpolating 2D data to match 3D dimensions.")
                    
                    # Interpolate 2D data to match 3D dimensions
                    n_lines_3d, n_time_points_3d, n_frames_3d = data_3d_shape
                    n_lines_2d, n_time_points_2d = data_2d_shape
                    
                    # Create interpolation grid for 2D data
                    lines_2d = np.linspace(0, 1, n_lines_2d)
                    time_2d = np.linspace(0, 1, n_time_points_2d)
                    
                    # Create target grid for 3D data
                    lines_3d = np.linspace(0, 1, n_lines_3d)
                    time_3d = np.linspace(0, 1, n_time_points_3d)
                    
                    # Interpolate 2D data to match 3D dimensions
                    # Create interpolation grid for 2D data
                    interpolator = RegularGridInterpolator((lines_2d, time_2d), energy_dict_2d[freq], method='linear')
                    
                    # Create target grid for 3D data
                    lines_3d_grid, time_3d_grid = np.meshgrid(lines_3d, time_3d, indexing='ij')
                    points = np.column_stack((lines_3d_grid.ravel(), time_3d_grid.ravel()))
                    
                    # Interpolate
                    data_2d_interpolated = interpolator(points).reshape(n_lines_3d, n_time_points_3d)
                    
                    # Add a new axis for frames to enable broadcasting
                    data_2d_expanded = data_2d_interpolated[..., np.newaxis]
                else:
                    # Shapes match, use original 2D data
                    data_2d_expanded = energy_dict_2d[freq][..., np.newaxis]
                
                # Normalize all frames using broadcasting
                normalized_dict[freq] = energy_dict_3d[freq] / data_2d_expanded
                
                logging.info(f"Successfully normalized frequency {freq:.2f} MHz")
            else:
                logging.warning(f"Frequency {freq:.2f} MHz not found in reference data")
                
        return normalized_dict

            


    
 


class BSC:
    
    def __init__(self,
                 samples_folder_path: str,
                 result_folder_path: str,
                 roi_folder_path: str,
                 normalization_method: str,
                 window: str,
                 nperseg: int,
                 noverlap: int,
                 alpha: float,
                 roi_size: str):
        
        self.samples_folder_path = samples_folder_path
        self.result_folder_path = result_folder_path
        self.roi_folder_path = roi_folder_path
        self.normalization_method = normalization_method
        self.roi_size = roi_size
        
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
        Save BSC energy results to separate Excel files for each frequency.
        Each Excel file will contain frame-by-frame information for that frequency.
        Time points are rows and lines are columns in the output.
        
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
            
            # Get the energy dictionary and frequencies
            energy_dict = bsc_obj.energy_dict
            frequencies = bsc_obj.frequencies
            
            # For each frequency, create a separate Excel file
            for freq in frequencies:
                # Get the energy data for this frequency
                energy_data = energy_dict[freq]  # Shape: (lines, time_points, frames)
                n_lines, n_time_points, n_frames = energy_data.shape
                
                # Create Excel writer for this frequency
                freq_file = result_dir / f"frequency_{freq:.2f}MHz.xlsx"
                with pd.ExcelWriter(freq_file, engine='openpyxl') as writer:
                    # For each frame, create a separate sheet
                    for frame_idx in range(n_frames):
                        # Get frame data and transpose it
                        frame_data = energy_data[:, :, frame_idx].T  # Transpose to make time_points rows and lines columns
                        
                        # Convert to DataFrame with transposed orientation
                        df = pd.DataFrame(
                            frame_data,
                            index=[f"Time_{i+1}" for i in range(n_time_points)],
                            columns=[f"Line_{i+1}" for i in range(n_lines)]
                        )
                        
                        # Save to sheet
                        sheet_name = f"Frame_{frame_idx+1}"
                        df.to_excel(writer, sheet_name=sheet_name)
                
                logging.info(f"Saved energy data for frequency {freq:.2f} MHz to {freq_file}")
            
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
                    roi_file_path=roi_file_path,
                    roi_size=self.roi_size
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
            
            
            
            
            
            
class HilbertTransformProcessor:
    """
    A class to process samples and create an Excel file with different frames for each sample
    containing the Hilbert transformation of signals without normalization.
    """
    
    def __init__(self,
                 samples_folder_path: str,
                 result_folder_path: str,
                 roi_folder_path: str,
                 roi_size: str = 'large'):
        """
        Initialize the HilbertTransformProcessor.
        
        Args:
            samples_folder_path (str): Path to the samples folder
            result_folder_path (str): Path where to save the Excel file
            roi_folder_path (str): Path to the ROI folder
            roi_size (str): Size of ROI to use, either 'large' or 'small'. Defaults to 'large'
        """
        self.samples_folder_path = Path(samples_folder_path)
        self.result_folder_path = Path(result_folder_path)
        self.roi_folder_path = Path(roi_folder_path)
        self.roi_size = roi_size
        
        # Validate roi_size
        if roi_size.lower() not in ['large', 'small']:
            raise ValueError("roi_size must be either 'large' or 'small'")
        
        # Create result directory
        self.result_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Run the processing
        self.__run()
    
    def __run(self):
        """Main processing method."""
        try:
            logging.info("Starting Hilbert transform processing")
            
            # Get all sample folders
            sample_folders = [f for f in self.samples_folder_path.iterdir() if f.is_dir()]
            
            if not sample_folders:
                logging.error(f"No sample folders found in {self.samples_folder_path}")
                return
            
            logging.info(f"Found {len(sample_folders)} sample folders")
            
            # Process each sample
            for sample_folder in sample_folders:
                try:
                    logging.info(f"Processing sample: {sample_folder.name}")
                    
                    # Process the sample and get Hilbert transform data
                    hilbert_data_dict = self._process_single_sample(sample_folder)
                    
                    if hilbert_data_dict is not None:
                        # Print signal shape information
                        self._print_signal_shape_info(hilbert_data_dict, sample_folder.name)
                        
                        # Save results for this sample
                        self.save_hilbert_results_as_excel(hilbert_data_dict, sample_folder.name)
                        logging.info(f"Saved Hilbert transform data for {sample_folder.name}")
                    
                except Exception as e:
                    logging.error(f"Error processing sample {sample_folder.name}: {str(e)}")
                    continue
            
            logging.info(f"Hilbert transform processing completed for all samples")
            
        except Exception as e:
            logging.error(f"Error in Hilbert transform processing: {str(e)}")
            raise
    
    def _process_single_sample(self, sample_folder: Path) -> Optional[dict]:
        """
        Process a single sample and return Hilbert transform data.
        
        Args:
            sample_folder (Path): Path to the sample folder
            
        Returns:
            Optional[dict]: Dictionary containing Hilbert transform data organized by frame or None if failed
        """
        try:
            # Construct ROI file path
            roi_file_path = self.roi_folder_path / f"{sample_folder.name}.xlsx"
            
            if not roi_file_path.exists():
                logging.warning(f"ROI file not found for sample {sample_folder.name}: {roi_file_path}")
                return None
            
            # Create SingleSampleData object
            single_sample_data = SingleSampleData(
                sample_folder_path=sample_folder,
                roi_file_path=roi_file_path,
                roi_size=self.roi_size
            )
            
            # Get the ROI data (unnormalized)
            roi_data = single_sample_data.data_3d_roi_unnormal
            
            if roi_data is None:
                logging.warning(f"No ROI data available for sample {sample_folder.name}")
                return None
            
            # Calculate Hilbert transform for each frame
            hilbert_results = self._calculate_hilbert_transform(roi_data, sample_folder.name)
            
            return hilbert_results
            
        except Exception as e:
            logging.error(f"Error processing sample {sample_folder.name}: {str(e)}")
            return None
    
    def _calculate_hilbert_transform(self, roi_data: np.ndarray, sample_name: str) -> dict:
        """
        Calculate Hilbert transform for the ROI data and extract only amplitude.
        Returns data in the same format as BSC for consistent processing.
        
        Args:
            roi_data (np.ndarray): 3D ROI data array (lines, samples, frames)
            sample_name (str): Name of the sample
            
        Returns:
            dict: Dictionary containing Hilbert transform amplitude data organized by frame
        """
        try:
            from scipy.signal import hilbert
            
            n_lines, n_samples, n_frames = roi_data.shape
            logging.info(f"Calculating Hilbert transform amplitude for {sample_name} - Shape: {roi_data.shape}")
            
            # Initialize dictionary to store results by frame
            hilbert_data_dict = {}
            
            # Process each frame
            for frame_idx in range(n_frames):
                logging.info(f"Processing frame {frame_idx + 1}/{n_frames} for {sample_name}")
                
                # Get current frame
                frame_data = roi_data[:, :, frame_idx]  # Shape: (lines, samples)
                
                # Initialize amplitude array for this frame
                amplitude_data = np.zeros((n_lines, n_samples))
                
                # Calculate Hilbert transform for each line
                for line_idx in range(n_lines):
                    # Get current line
                    line_signal = frame_data[line_idx, :]
                    
                    # Calculate Hilbert transform
                    analytic_signal = hilbert(line_signal)
                    
                    # Extract only amplitude
                    amplitude = np.abs(analytic_signal)
                    
                    # Store amplitude data
                    amplitude_data[line_idx, :] = amplitude
                
                # Store frame data in dictionary
                hilbert_data_dict[frame_idx + 1] = amplitude_data
            
            logging.info(f"Completed Hilbert transform amplitude calculation for {sample_name}")
            logging.info(f"Number of frames processed: {len(hilbert_data_dict)}")
            
            return hilbert_data_dict
            
        except Exception as e:
            logging.error(f"Error calculating Hilbert transform amplitude for {sample_name}: {str(e)}")
            raise
    
    def save_hilbert_results_as_excel(self, hilbert_data_dict: dict, sample_name: str) -> None:
        """
        Save Hilbert transform amplitude results to Excel files following BSC pattern.
        Creates one Excel file with only the first frame information.
        Time points are rows and lines are columns in the output.
        
        Args:
            hilbert_data_dict: Dictionary containing Hilbert transform data organized by frame
            sample_name: Name of the sample
        """
        try:
            # Create result directory path mirroring the samples structure
            result_dir = Path(self.result_folder_path) / sample_name
            result_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Saving Hilbert transform results for sample {sample_name} to {result_dir}")
            
            # Create Excel file for Hilbert transform amplitude
            hilbert_file = result_dir / "hilbert_amplitude.xlsx"
            
            with pd.ExcelWriter(hilbert_file, engine='openpyxl') as writer:
                # Get only the first frame data
                frame_numbers = list(hilbert_data_dict.keys())
                if frame_numbers:
                    first_frame_idx = frame_numbers[0]
                    amplitude_data = hilbert_data_dict[first_frame_idx]
                    
                    # Get frame data and transpose it (like BSC)
                    # amplitude_data shape: (lines, samples)
                    n_lines, n_samples = amplitude_data.shape
                    
                    # Transpose to make time_points rows and lines columns (like BSC)
                    frame_data_transposed = amplitude_data.T  # Shape: (samples, lines)
                    
                    # Convert to DataFrame with transposed orientation
                    df = pd.DataFrame(
                        frame_data_transposed,
                        index=[f"Time_{i+1}" for i in range(n_samples)],
                        columns=[f"Line_{i+1}" for i in range(n_lines)]
                    )
                    
                    # Save to sheet
                    sheet_name = "Frame_1"
                    df.to_excel(writer, sheet_name=sheet_name)
                    
                    logging.info(f"Saved first frame (Frame_{first_frame_idx}) data to sheet {sheet_name}")
                else:
                    logging.warning(f"No frames available for sample {sample_name}")
            
            logging.info(f"Saved Hilbert transform amplitude data (first frame only) to {hilbert_file}")
            
        except Exception as e:
            logging.error(f"Error saving Hilbert transform results: {str(e)}")
            raise
    
    def _print_signal_shape_info(self, hilbert_data_dict: dict, sample_name: str) -> None:
        """
        Print signal shape information for a sample.
        
        Args:
            hilbert_data_dict: Dictionary containing Hilbert transform data organized by frame
            sample_name: Name of the sample
        """
        try:
            if not hilbert_data_dict:
                logging.warning(f"No data available for sample {sample_name}")
                return
            
            # Get frame information
            frame_numbers = list(hilbert_data_dict.keys())
            num_frames = len(frame_numbers)
            
            # Get shape information from first frame
            first_frame_data = hilbert_data_dict[frame_numbers[0]]
            n_lines, n_samples = first_frame_data.shape
            
            # Print shape information
            logging.info(f"=== Signal Shape Information for {sample_name} ===")
            logging.info(f"Number of frames: {num_frames}")
            logging.info(f"Number of lines: {n_lines}")
            logging.info(f"Number of samples per line: {n_samples}")
            logging.info(f"Frame range: {min(frame_numbers)} to {max(frame_numbers)}")
            logging.info(f"Total data points: {num_frames * n_lines * n_samples:,}")
            logging.info(f"Data shape per frame: ({n_lines}, {n_samples})")
            logging.info(f"================================================")
            
        except Exception as e:
            logging.error(f"Error printing signal shape info for {sample_name}: {str(e)}") 
            
            
            
            
            
            