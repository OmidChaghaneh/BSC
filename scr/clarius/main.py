# Standard Library Imports
from pathlib import Path
from typing import Union, Optional, Tuple
import os
import logging
import shutil

# Third-Party Imports
import numpy as np
import pandas as pd
import yaml

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
                    output_depth_filename = os.path.join(folder_path, f"{rf_raw_name}_full_depth_mm.npy")
                    
                    np.save(output_npy_filename, rf_data)
                    logging.info(f"Saved RF data to: {output_npy_filename}")
                    
                    df = pd.DataFrame({'rf_delay_samples': [rf_delay_samples]})
                    df.to_excel(output_excel_filename, index=False)
                    logging.info(f"Saved delay samples to: {output_excel_filename}")
                    
                    np.save(output_depth_filename, full_depth_mm)
                    logging.info(f"Saved full depth to: {output_depth_filename}")
                    
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
                 device: str = "C3",
                 size: str = "large"):
        """
        Initialize the Data object with specified parameters.
        
        Args:
            sample_folder_path (Union[str, Path]): Path to the sample folder containing extracted data
            device (str): Device type ("C3" or "L15")
            size (str): Size of the data ("large" or "small")
        """
        # Convert path to Path object if string
        self.sample_folder_path = Path(sample_folder_path)
        self.device = device
        self.size = size
                
        # Extract sample name from path
        self.sample_name = self.sample_folder_path.name
        
        # Initialize data attributes
        self.data_3d = None
        self.delay_samples = None
        self.full_depth_mm = None
        self.roi_data = None
        self.data_3d_roi_normal = None  
        self.data_3d_roi_unnormal = None 
        self.data_3d_phantom = None  # Store phantom data
        self.data_3d_phantom_roi = None  # Store phantom data cut with ROI
        
        # Configuration parameters
        self.sampling_frequency = 40e6  # 40 MHz sampling frequency
        self.freq_band = [1e6, 10e6]  # Analysis frequency band 1-10 MHz
        self.center_frequency = 5e6  # 5 MHz center frequency
        self.ref_attenuation = 0.5  # dB/cm/MHz for reference phantom
        self.ref_bsc = 1e-3  # 1/cm-sr for reference phantom
        self.axial_res = 0.1  # mm, axial resolution
        
        # Read data
        self.__run()
            
    def __run(self):
        self.read_extracted_folder()
        self.read_roi_data()
        self.correct_roi_data()
        self.cut_data_based_on_roi()
        self.read_phantom_numpy()
        self.cut_phantom_data_based_on_roi()
        
    def read_roi_data(self):
        """Read ROI data from the corresponding Excel file in the ROIs folder."""
        try:
            # Construct the path to the ROI Excel file
            roi_file_path = self.sample_folder_path.parent.parent / "ROIs" / f"{self.sample_name}.xlsx"
            
            if not roi_file_path.exists():
                logging.warning(f"No ROI file found at: {roi_file_path}")
                return
                
            logging.info(f"Loading ROI data from: {roi_file_path}")
            
            # Read the Excel file
            self.roi_data = pd.read_excel(roi_file_path)
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

                    # Look for corresponding full depth mm numpy file
                    depth_pattern = f"{self.device}_{self.size}_*_full_depth_mm.npy"
                    depth_files = list(folder.glob(depth_pattern))

                    if depth_files:
                        # Load the first matching depth file
                        depth_file = depth_files[0]
                        logging.info(f"Loading full depth from: {depth_file}")
                        
                        # Load the depth data
                        self.full_depth_mm = np.load(depth_file)
                        logging.info(f"Loaded full depth: {self.full_depth_mm}")
                    else:
                        logging.warning(f"No full depth file found matching pattern: {depth_pattern}")
                    
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


class BSC:
    
    def __init__(self, sample_folder_path: Union[str, Path],
                 device: str = "C3",
                 size: str = "large",
                 mode: str = "normalize_phantom"):
        
        self.sample_folder_path = Path(sample_folder_path)
        self.device = device
        self.size = size
        self.mode = mode
    
        if self.mode == "normalize_phantom":
            self.bsc_normalize_phantom()
        elif self.mode == "normalize_healthy_liver":
            self.bsc_normalize_healthy_liver()
        elif self.mode == "ac_fix_alpha":
            self.bsc_ac_fix_alpha()
        elif self.mode == "ac_calculated_aplha":
            pass
        
    def bsc_normalize_phantom(self):
        pass
    
    def bsc_normalize_healthy_liver(self):
        pass
    
    def bsc_ac_fix_alpha(self):
        pass
    
    def bsc_ac_calculated_aplha(self):
        pass
    
    
    def calculate_bsc_AEK(self,
                   roi_data: np.ndarray,
                   phantom_data: np.ndarray,
                   sampling_frequency: float = 40e6,
                   freq_band: list = [1e6, 10e6],
                   center_frequency: float = 5e6,
                   ref_attenuation: float = 0.5,
                   ref_bsc: float = 1e-3,
                   axial_res: float = 0.1) -> float:
        """
        Calculate the backscatter coefficient (BSC) using the reference phantom method.
        This method uses the ROI data and phantom data to compute BSC based on Yao et al. (1990).
        
        Args:
            roi_data (np.ndarray): ROI data array (2D or 3D)
            phantom_data (np.ndarray): Phantom data array (2D or 3D)
            sampling_frequency (float, optional): Sampling frequency in Hz. Defaults to 40MHz.
            freq_band (list, optional): Analysis frequency band [start, end] in Hz. Defaults to [1-10MHz].
            center_frequency (float, optional): Center frequency in Hz. Defaults to 5MHz.
            ref_attenuation (float, optional): Reference phantom attenuation in dB/cm/MHz. Defaults to 0.5.
            ref_bsc (float, optional): Reference phantom BSC in 1/cm-sr. Defaults to 1e-3.
            axial_res (float, optional): Axial resolution in mm. Defaults to 0.1.
            
        Returns:
            float: Backscatter coefficient of the ROI (1/cm-sr)
        """
        NUM_FOURIER_POINTS = 8192
        
        def repmat(a: np.ndarray, m: int, n: int) -> np.ndarray:
            """
            Replicates a matrix similar to MATLAB's repmat function.
            """
            return np.tile(a, (m, n))
            
        def compute_hanning_power_spec(rf_data: np.ndarray, start_frequency: int, end_frequency: int, 
                                    sampling_frequency: int) -> Tuple[np.ndarray, np.ndarray]:
            """Compute the power spectrum of 3D spatial RF data using a Hanning window."""
            # Create Hanning Window Function for the axial dimension
            unrm_wind = np.hanning(rf_data.shape[0])
            wind_func_computations = unrm_wind * np.sqrt(len(unrm_wind) / sum(np.square(unrm_wind)))
            wind_func = repmat(
                wind_func_computations.reshape((rf_data.shape[0], 1)), 1, rf_data.shape[1]
            )

            # Frequency Range
            frequency = np.linspace(0, sampling_frequency, NUM_FOURIER_POINTS)
            f_low = round(start_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
            f_high = round(end_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
            freq_chop = frequency[f_low:f_high]

            # Get PS
            if rf_data.ndim == 3:
                power_spectra = []
                for i in range(rf_data.shape[2]):
                    fft = np.square(
                        abs(np.fft.fft(np.transpose(np.multiply(rf_data[:,:,i], wind_func)), NUM_FOURIER_POINTS) * rf_data[:,:,i].size)
                    )
                    full_ps = np.mean(fft, axis=0)
                    power_spectra.append(full_ps[f_low:f_high])
                ps = np.mean(power_spectra, axis=0)
            elif rf_data.ndim == 2:
                fft = np.square(
                    abs(np.fft.fft(np.transpose(np.multiply(rf_data, wind_func)), NUM_FOURIER_POINTS) * rf_data.size)
                )
                full_ps = np.mean(fft, axis=0)
                ps = full_ps[f_low:f_high]
            else:
                raise ValueError("Invalid RF data dimensions. Expected 2D or 3D data.")

            return freq_chop, ps
            
        try:
            # If 3D data is provided, use the center slice
            if roi_data.ndim == 3:
                center_frame = roi_data.shape[2] // 2
                roi_data = roi_data[:, :, center_frame]
            if phantom_data.ndim == 3:
                center_frame = phantom_data.shape[2] // 2
                phantom_data = phantom_data[:, :, center_frame]
            
            # Calculate power spectra using Hanning window
            f, ps_roi = compute_hanning_power_spec(
                roi_data, freq_band[0], freq_band[1], sampling_frequency
            )
            ps_roi = 20 * np.log10(ps_roi)
            
            _, ps_phantom = compute_hanning_power_spec(
                phantom_data, freq_band[0], freq_band[1], sampling_frequency
            )
            ps_phantom = 20 * np.log10(ps_phantom)
            
            # Find index for center frequency
            freq_idx = np.argmin(np.abs(f - center_frequency))
            
            # Get power spectrum values at center frequency
            ps_sample = ps_roi[freq_idx]
            ps_ref = ps_phantom[freq_idx]
            
            # Calculate signal ratio
            s_ratio = ps_sample / ps_ref
            
            # Convert attenuation coefficients from dB to Neper
            np_conversion_factor = np.log(10) / 20
            ref_att_coef = ref_attenuation * np_conversion_factor  # Np/cm/MHz
            
            # Calculate window depth in cm
            window_depth_cm = roi_data.shape[0] * axial_res / 10  # cm
            
            # Convert attenuation to current frequency
            freq_mhz = center_frequency / 1e6
            ref_att_coef *= freq_mhz  # Np/cm
            
            # Calculate attenuation compensation
            att_comp = np.exp(4 * window_depth_cm * ref_att_coef)
            
            # Calculate BSC
            bsc = s_ratio * ref_bsc * att_comp
            
            logging.info(f"Calculated BSC: {bsc:.2e} 1/cm-sr")
            return bsc
            
        except Exception as e:
            logging.error(f"Error calculating BSC: {str(e)}")
            return None
    