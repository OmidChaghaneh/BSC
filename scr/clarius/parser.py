# Standard Library Imports
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Tuple

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from tqdm import tqdm

# Local Module Imports
from .objects import DataOutputStruct, InfoStruct
from .transforms import scanConvert

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.INFO)


# tar file unpacker    
###################################################################################  
class ClariusTarUnpacker():
    """
    A class for extracting and processing `.tar` archives containing `.lzo` and `.raw` files.
    
    Attributes:
        tar_files_path (str): The path to the directory containing `.tar` files.
        extraction_mode (str): Extraction mode - either "single" or "multiple".
        lzo_py_file_path (str): Path to the LZO executable for decompression (Windows only).
    """
    ###################################################################################
    
    def __init__(self, path: str, extraction_mode: str) -> None:  
        """
        Initializes the ClariusTarUnpacker class and starts the extraction process.
        
        Args:
            tar_files_path (str): The directory containing `.tar` files.
            extraction_mode (str): Mode of extraction - "single" or "multiple".
        
        Raises:
            ValueError: If `extraction_mode` is not "single" or "multiple".
        """
        
        self.path = path
        self.extraction_mode = extraction_mode
        
        # Using lzop.py file from the local directory
        self.lzo_py_file_path = os.path.join(os.path.dirname(__file__), 'lzop.py')
        
        # single tar extraction attibutes
        self.single_tar_extraction: bool = None
        self.tar_path: str = None
        
        if self.extraction_mode == "single_sample":
            """Extracts data from a single sample containing multiple tar files.
            The provided path should point to a directory containing multiple tar files. 
            Each tar file within this directory will be processed sequentially, 
            extracting its contents into the appropriate output location."""
            if self.check_input_path():
                self.__run_single_sample_extraction()

        elif self.extraction_mode == "multiple_samples":
            """Processes multiple samples, where each sample is a separate directory 
            that potentially contains multiple tar files. The given path should be a 
            directory containing multiple subdirectories, each representing an individual 
            sample. Each subdirectory is processed independently, extracting the tar files 
            within it."""
            if self.check_input_path():
                self.__run_multiple_samples_extraction()

        elif self.extraction_mode == "single_tar":
            """Extracts data from a single tar file.
            The provided path should point directly to a single tar file. The file 
            will be extracted to a designated output directory, maintaining its internal 
            structure. This mode is useful when processing a standalone tar file rather 
            than multiple files in a batch."""
            if self.check_input_path():
                self.__run_single_tar_extraction()

        else:
            """Handles invalid extraction modes by raising an error."""
            raise ValueError(f"Invalid mode: {self.extraction_mode}")
        
    ###################################################################################
        
    def __repr__(self):
        """
        Returns a string representation of the object.

        This method provides a developer-friendly representation of the instance,
        typically including key attributes for debugging purposes.

        Returns:
            str: A string representation of the instance.
        """

        return f"{self.__class__.__name__}"

    ###################################################################################
    
    def __run_single_sample_extraction(self):
        """Runs the extraction process for a single directory."""
        self.extract_tar_files()
        self.set_path_of_extracted_folders()
        self.convert_env_tgc_to_env_tgc_yml()
        self.set_path_of_lzo_files_inside_extracted_folders()
        self.read_lzo_files()
        self.set_path_of_raw_files_inside_extracted_folders()
        self.delete_hidden_files_in_extracted_folders()

    ###################################################################################

    def __run_multiple_samples_extraction(self):
        """Extracts data from all directories inside `self.path`."""
        try:
            # Retrieve all subdirectory paths
            folder_paths = [
                os.path.join(self.path, folder)
                for folder in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, folder))
            ]

            # Process each folder for data extraction
            for folder_path in folder_paths:
                self.path = folder_path  # Update path before extraction
                self.__run_single_sample_extraction()

        except Exception as e:
            logging.error(f"An error occurred while extracting data: {e}")

    ###################################################################################
    
    def __run_single_tar_extraction(self):
        """Handles extraction when a single tar file is provided."""
        self.single_tar_extraction = True
        self.tar_path = self.path
        self.path = self.get_folder_path_from_file_path(self.tar_path)
        self.__run_single_sample_extraction()
            
    ###################################################################################
        
    def extract_tar_files(self):
        """
        Extracts all non-hidden tar files in the specified sample folder.
        If `self.single_tar_extraction` is True, `self.path` should be a `.tar` file.
        """

        if not self.single_tar_extraction:
            # Ensure the given path is a directory
            if not os.path.isdir(self.path):
                logging.error(f"Path '{self.path}' is not a directory.")
                return

            for item_name in os.listdir(self.path):
                item_path = os.path.join(self.path, item_name)

                # Ignore hidden files
                if item_name.startswith('.'):
                    continue
                
                # Check if the item is a tar archive
                if os.path.isfile(item_path) and item_name.endswith('.tar') and tarfile.is_tarfile(item_path):
                    file_name = Path(item_name).stem
                    extracted_folder = os.path.join(self.path, f"{file_name}_extracted")
                    os.makedirs(extracted_folder, exist_ok=True)

                    try:
                        with tarfile.open(item_path, 'r') as tar:
                            tar.extractall(path=extracted_folder)
                            logging.info(f"Extracted '{item_name}' into '{extracted_folder}'")
                    except (tarfile.TarError, OSError) as e:
                        logging.error(f"Error extracting '{item_name}': {e}")

        elif self.single_tar_extraction:
            # Handle single tar extraction
            if os.path.isfile(self.tar_path) and self.tar_path.endswith('.tar') and tarfile.is_tarfile(self.tar_path):
                file_name = Path(self.tar_path).stem
                extracted_folder = os.path.join(os.path.dirname(self.tar_path), f"{file_name}_extracted")
                os.makedirs(extracted_folder, exist_ok=True)

                try:
                    with tarfile.open(self.tar_path, 'r') as tar:
                        tar.extractall(path=extracted_folder)
                        logging.info(f"Extracted '{self.tar_path}' into '{extracted_folder}'")
                except (tarfile.TarError, OSError) as e:
                    logging.error(f"Error extracting '{self.tar_path}': {e}")
            else:
                logging.error(f"Invalid tar file: '{self.tar_path}'")

    ###################################################################################
    
    def set_path_of_extracted_folders(self):
        """Finds and stores paths of extracted folders inside `self.path`."""
        logging.info("Searching for extracted folders...")

        # Find all directories containing 'extracted' in their name
        self.extracted_folders_path_list = [
            os.path.join(self.path, item)
            for item in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, item)) and "extracted" in item
        ]

        # Log each extracted folder found
        for folder in self.extracted_folders_path_list:
            logging.info(f"Found extracted folder: {folder}")

        # Log summary
        logging.info(f"Total extracted folders found: {len(self.extracted_folders_path_list)}")
    
    ###################################################################################
    
    def convert_env_tgc_to_env_tgc_yml(self):
        """
        Renames a single file ending with 'env.tgc' to 'env.tgc.yml' in the extracted folders.
        """
        for folder in self.extracted_folders_path_list:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('env.tgc'):
                        new_file_name = file.replace('env.tgc', 'env.tgc.yml')
                        old_file_path = os.path.join(root, file)
                        new_file_path = os.path.join(root, new_file_name)
                        if os.path.exists(new_file_path):
                            os.remove(new_file_path)
                            logging.info(f"Existing file {new_file_name} removed.") 
                        try:
                            os.rename(old_file_path, new_file_path)
                            logging.info(f"Renamed {file} to {new_file_name}")
                        except OSError as e:
                            logging.error(f"Error renaming {file} to {new_file_name}: {e}")
                        return  # Exit after processing the first matching file
                        
    ###################################################################################

    def set_path_of_lzo_files_inside_extracted_folders(self):
        """Finds and stores paths of .lzo files inside extracted folders."""
        logging.info("Starting to search for LZO files inside extracted folders...")

        # Ensure extracted folders list is available
        if not self.extracted_folders_path_list:
            logging.warning("No extracted folders found. Please check the extracted folders path list.")
            return

        # Now search for .lzo files in the extracted folders
        lzo_files = []
        for folder in self.extracted_folders_path_list:
            for root, dirs, files in os.walk(folder):  # Walk through each extracted folder
                for file in files:
                    if file.endswith('.lzo'):
                        lzo_file_path = os.path.join(root, file)
                        lzo_files.append(lzo_file_path)
                        logging.info(f"Found LZO file: {lzo_file_path}")

        self.lzo_files_path_list = lzo_files  # Store the paths in the class attribute

        # Log the number of LZO files found
        logging.info(f"Total LZO files found: {len(lzo_files)}")
            
    ###################################################################################

    def read_lzo_files(self):
        """
        Detects the operating system and decompresses `.lzo` files accordingly.

        **Workflow:**
        1. Determines whether the system is Windows or macOS.
        2. Logs the detected operating system.
        3. If running on **Windows**:
            - Constructs the path to the LZO executable.
            - Checks if the executable exists.
            - Iterates through the `.lzo` files and decompresses them using `lzop.py`.
        4. If running on **macOS**:
            - Attempts to decompress `.lzo` files using the `lzop` command.
            - If `lzop` is missing, checks for Homebrew.
            - Installs `lzop` via Homebrew if necessary.
            - Decompresses `.lzo` files after ensuring `lzop` is installed.
        5. Logs successes and failures, handling potential errors like:
            - `FileNotFoundError`
            - `PermissionError`
            - `subprocess.CalledProcessError`
            - Other unexpected exceptions.

        **Returns:**
        - Logs the status of each decompression attempt.
        - Exits the program if `lzop` is missing on macOS and cannot be installed.
        """
        # Set self.os based on the platform
        os_name = platform.system().lower()           
        
        if 'windows' in os_name:
            self.os = "windows"
        elif 'darwin' in os_name:
            self.os = "mac"
        elif 'linux' in os_name:
            self.os = "linux"
        else:
            self.os = "unknown"
            
        logging.info(f'Detected operating system: {self.os}')
               
        if self.os == "windows":

            # Log the path being checked
            logging.info(f'Checking path for LZO executable: {self.lzo_py_file_path}')

            # Check if the executable exists
            if not os.path.isfile(self.lzo_py_file_path):
                logging.error(f'LZO executable not found: {self.lzo_py_file_path}')
                return

            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run([self.lzo_py_file_path, '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except subprocess.CalledProcessError as e:
                    #logging.error(f'Error decompressing {lzo_file_path}: {e}')
                    pass
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')
                    
        elif self.os == "mac":
            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except FileNotFoundError:
                    # check if homebrew is installed
                    brew_path = shutil.which("brew")
                    # if homebrew is not installed, tell the user to install it manually and exit 
                    if brew_path is None:
                        logging.error("Homebrew is required to install lzop. A description how to install Homebrew can be found in the README.md file.")
                        sys.exit()
                   
                    # install lzop using homebrew
                    subprocess.run(['brew', 'update'], check=True)
                    subprocess.run(['arch', '-arm64', 'brew', 'install', 'lzop'], check=True)
                    logging.info("Successfully installed lzop using Homebrew.")
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')

                except subprocess.CalledProcessError as e:
                    logging.error(f'Error decompressing {lzo_file_path}: {e}')
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')            
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')    
                    
        elif self.os == "linux":
            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except FileNotFoundError as e:
                    logging.error(f"lzop must be installed to decompress LZO files. Please install lzop and try again.")
                    sys.exit()

                except subprocess.CalledProcessError as e:
                    logging.error(f'Error decompressing {lzo_file_path}: {e}')
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')            
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')    

    ###################################################################################

    def set_path_of_raw_files_inside_extracted_folders(self):
        """Searches for .raw files inside extracted folders and stores their paths."""
        logging.info("Starting to search for RAW files inside extracted folders...")

        # Ensure extracted folders list is available
        if not self.extracted_folders_path_list:
            logging.warning("No extracted folders found. Please check the extracted folders path list.")
            return

        # Now search for .raw files in the extracted folders
        raw_files = []
        for folder in self.extracted_folders_path_list:
            for root, dirs, files in os.walk(folder):  # Walk through each extracted folder
                for file in files:
                    if file.endswith('.raw'):
                        raw_file_path = os.path.join(root, file)
                        raw_files.append(raw_file_path)
                        logging.info(f"Found RAW file: {raw_file_path}")

        self.raw_files_path_list = raw_files  # Store the paths in the class attribute

        # Log the number of RAW files found
        logging.info(f"Total RAW files found: {len(raw_files)}")

    ###################################################################################
    
    def delete_hidden_files_in_extracted_folders(self):
        """Deletes hidden files (starting with a dot) in extracted folders."""
        # Iterate through each extracted folder path
        for folder_path in self.extracted_folders_path_list:
            try:
                # List all files in the folder
                for filename in os.listdir(folder_path):
                    # Check if the file is hidden (starts with a dot)
                    if filename.startswith('.'):
                        file_path = os.path.join(folder_path, filename)
                        os.remove(file_path)  # Delete the hidden file
                        logging.info(f"Deleted hidden file: {file_path}")
            except Exception as e:
                logging.error(f"Error while deleting hidden files in {folder_path}: {e}")
          
    ###################################################################################
    @staticmethod
    def get_folder_path_from_file_path(file_path: str) -> str:
        """Returns the absolute directory path of the given file."""
        return os.path.dirname(os.path.abspath(file_path))
    
    ###################################################################################
    
    def check_input_path(self):
        """
        Validates the input path based on the specified extraction mode.

        This method checks whether the provided path exists and conforms to 
        the expected format for different extraction modes:

        - "single_sample": The path must be a directory containing at least 
        one `.tar` file.
        - "multiple_samples": The path must be a directory containing at least 
        one subdirectory.
        - "single_tar": The path must be a valid `.tar` file.

        Returns:
            bool: True if the path is valid for the specified extraction mode, 
                otherwise False with a warning message.
        """
        if not os.path.exists(self.path):
            logging.warning(f"The path '{self.path}' does not exist.")
            return False

        if self.extraction_mode == "single_sample":
            if not os.path.isdir(self.path):
                logging.warning(f"The path '{self.path}' is not a directory.")
                return False
            tar_files = [f for f in os.listdir(self.path) if f.endswith(".tar")]
            if not tar_files:
                logging.warning(f"No .tar files found in '{self.path}'.")
                return False

        elif self.extraction_mode == "multiple_samples":
            if not os.path.isdir(self.path):
                logging.warning(f"The path '{self.path}' is not a directory.")
                return False
            subfolders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
            if not subfolders:
                logging.warning(f"No subdirectories found in '{self.path}'.")
                return False

        elif self.extraction_mode == "single_tar":
            if not os.path.isfile(self.path) or not self.path.endswith(".tar"):
                logging.warning(f"The path '{self.path}' is not a valid .tar file.")
                return False
            if not tarfile.is_tarfile(self.path):
                logging.warning(f"The file '{self.path}' is not a valid tar archive.")
                return False

        else:
            logging.warning(f"Unknown extraction mode '{self.extraction_mode}'.")
            return False

        return True
    
    ###################################################################################
    
###################################################################################



# parser
###################################################################################  

def get_signal_envelope_xd(signal_xd: np.ndarray, hilbert_transform_axis: int) -> np.ndarray:
    """
    Computes the envelope of an x-dimensional signal using the Hilbert transform.

    This function mirrors the input signal along all its dimensions, applies the 
    Hilbert transform along the specified axis, and then extracts the original 
    signal envelope from the analytic signal.

    Args:
        signal_xd (np.ndarray): The input x-dimensional signal represented as a NumPy array.
        hilbert_transform_axis (int): The axis along which the Hilbert transform is applied.

    Returns:
        np.ndarray: The computed signal envelope of the input signal.
    """
    logging.debug("Starting to set the xD signal envelope.")

    # Step 1: Mirror the signal
    logging.debug("Mirroring the signal.")
    signal_xd_mirrored = signal_xd.copy()
    
    for axis in range(signal_xd_mirrored.ndim):
        logging.debug(f"Reversing signal along axis {axis}.")
        # Reverse the signal along the current axis
        reversed_signal = np.flip(signal_xd_mirrored, axis)
        
        logging.debug(f"Concatenating original and reversed signal along axis {axis}.")
        # Concatenate the original and reversed signal along the current axis
        signal_xd_mirrored = np.concatenate((signal_xd_mirrored, reversed_signal), axis=axis)

    # Step 2: Apply the Hilbert transform to the mirrored signal
    logging.debug("Applying Hilbert transform to the mirrored signal.")
    hilbert_signal_mirrored = np.apply_along_axis(hilbert, arr=signal_xd_mirrored, axis=hilbert_transform_axis)
    
    # Step 3: Restore the original signal from the mirrored Hilbert signal
    logging.debug("Restoring the original signal from the mirrored Hilbert signal.")
    restored_signal = hilbert_signal_mirrored.copy()
    
    # Calculate the original size of the signal along each axis
    original_sizes = tuple(restored_signal.shape[axis] // 2 for axis in range(restored_signal.ndim))
    logging.debug(f"Original sizes of the signal: {original_sizes}.")
    
    # Slice to get the original signal
    logging.debug("Slicing to get the original signal.")
    signal_xd_unmirrored = restored_signal[tuple(slice(0, size) for size in original_sizes)]
    
    # Step 4: Compute the envelope of the analytic signal
    logging.debug("Computing the envelope of the analytic signal.")
    signal_envelope_xd = np.abs(signal_xd_unmirrored)

    logging.debug("Finished setting the xD signal envelope.")
    
    return signal_envelope_xd

###################################################################################

class ClariusParser():

    ###################################################################################
    
    def __init__(self, rf_raw_path: str, env_tgc_yml_path: str, rf_yml_path: str, 
                visualize: bool=False, use_tgc: bool=False):
        # Make sure all inputted files exist
        assert Path(rf_raw_path).exists() and Path(rf_yml_path).exists(), \
                "One or more input files do not exist. Please check the paths."
    
        # raw files path
        assert rf_raw_path.endswith("_rf.raw"), "The rf_raw_path must end with .raw"
        self.rf_raw_path = rf_raw_path
        self.visualize = visualize
        self.use_tgc = use_tgc

        # yml files path
        assert rf_yml_path.endswith("_rf.yml"), "The rf_yml_path must end with .yml or .yaml"
        assert env_tgc_yml_path is None or (env_tgc_yml_path.endswith("env.tgc.yml") and Path(env_tgc_yml_path).exists()), \
                "The env_tgc_yml_path must exist and end with .yml or .yaml or be None"
        self.rf_yml_path = rf_yml_path
        self.env_tgc_yml_path = env_tgc_yml_path

        # yml objects
        self.rf_yml_obj: ClariusParser.YmlParser
        self.env_yml_obj: ClariusParser.YmlParser
        self.env_tgc_yml_obj: ClariusParser.YmlParser
        
        # data from rf.raw file
        self.rf_raw_hdr: dict
        self.rf_raw_timestamps_1d: np.ndarray
        self.rf_raw_data_3d: np.ndarray 
        
        # data from env.raw file
        self.env_raw_hdr: dict
        self.env_raw_timestamps_1d: np.ndarray
        self.env_raw_data_3d: np.ndarray 
        
        # tgc
        self.default_tgc_data: dict = {}
        self.clean_tgc_data: dict = {}
        
        # no tgc
        self.rf_no_tgc_raw_data_3d: np.ndarray
        self.no_tgc_envelope_3d: np.ndarray
        
        # visualization
        self.default_frame: int = 0
        self.SPEED_OF_SOUND: int = 1540 # [m/s]
        self.hilbert_transform_axis: int = 1
        
        # data structure
        self.clarius_info_struct = ClariusParser.ClariusInfoStruct()
        #self.clarius_data_struct = DataOutputStruct()
        
        self.scan_converted = None
        
        self.max_samples_per_line_C3 = 2896
        self.max_samples_per_line_L15 = 2928

        self.__run()
                
    ###################################################################################
    
    def __run(self):
        # Process YAML files
        self.rf_yml_obj = ClariusParser.YmlParser(self.rf_yml_path)
        self.env_tgc_yml_obj = ClariusParser.YmlParser(self.env_tgc_yml_path) if self.env_tgc_yml_path else None

        # Process raw files
        self.read_rf_raw()

        # Create full depth array
        self.full_depth_mm, self.signal_length, self.delay_samples = self.create_full_imaging_depth_array()
        
        # create no TGC signal
        self.set_default_tgc_data()
        self.remove_tgc()
        
        # create full signal
        #self.set_full_signal()
        
        # Compute no TGC envelope
        #self.no_tgc_envelope_3d = get_signal_envelope_xd(self.rf_no_tgc_raw_data_3d,
        #                                                hilbert_transform_axis=self.hilbert_transform_axis)
        
        # Format outputs
        #self.set_data_of_clarius_info_struct()
        #self.reconstruct_bmode()
        
        # visualize
        if not self.visualize: return
        visualizer = ClariusParser.ParserVisualizations(self.rf_raw_data_3d, self.rf_no_tgc_raw_data_3d, 
                                                        self.default_frame, self.rf_yml_obj.rf_sampling_rate,
                                                        self.full_depth_mm, self.delay_samples,
                                                        self.hilbert_transform_axis)
            
        visualizer.image_envelope_2d("TGC",
                                     title="rf_raw",
                                     clip_fact=self.clarius_info_struct.clipFact,
                                     dyn_range=self.clarius_info_struct.dynRange)
        
        visualizer.image_envelope_2d("No TGC",
                                     title="rf_no_tgc_raw",
                                     clip_fact=self.clarius_info_struct.clipFact,
                                     dyn_range=self.clarius_info_struct.dynRange)
        
        visualizer.plot_1d_signal_and_fft("TGC", title="rf_raw")
        visualizer.plot_1d_signal_and_fft("No TGC", title="rf_no_tgc_raw")
        
    ###################################################################################
    
    def read_rf_raw(self):
        """
        Reads raw RF data from a binary file, extracting header information, timestamps, and frame data.
        
        This function follows the format used by Clarius' ultrasound raw data, as found in their GitHub repository:
        https://github.com/clariusdev/raw/blob/master/common/python/rdataread.py
        
        The function parses:
        - A 5-field header (id, frames, lines, samples, samplesize)
        - Timestamps for each frame
        - RF data for each frame
        
        The loaded data is stored as instance attributes:
        - `self.rf_raw_hdr`: Dictionary containing header information
        - `self.rf_raw_timestamps_1d`: NumPy array of frame timestamps
        - `self.rf_raw_data_3d`: NumPy array of the RF data
        
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file format is incorrect.
        """
        logging.info("Reading raw RF file: %s", self.rf_raw_path)

        # Define header fields and initialize dictionary
        hdr_keys = ('id', 'frames', 'lines', 'samples', 'samplesize')
        hdr = {}

        try:
            with open(self.rf_raw_path, 'rb') as raw_bytes:
                logging.info("Opened file successfully.")
                
                # Read and parse header
                hdr = {key: int.from_bytes(raw_bytes.read(4), byteorder='little') for key in hdr_keys}
                logging.info("Parsed header: %s", hdr)
                
                # Validate header values
                if any(value <= 0 for value in hdr.values()):
                    logging.error("Invalid header values: %s", hdr)
                    return
                
                frames, lines, samples, samplesize = hdr['frames'], hdr['lines'], hdr['samples'], hdr['samplesize']
                frame_size = lines * samples * samplesize
                
                # Initialize arrays
                timestamps = np.zeros(frames, dtype=np.int64)
                data = np.zeros((lines, samples, frames), dtype=np.int16)
                
                # Read timestamps and data for each frame
                for frame in range(frames):
                    timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
                    frame_data = raw_bytes.read(frame_size)
                    
                    if len(frame_data) != frame_size:
                        logging.error("Unexpected frame size at frame %d: Expected %d bytes, got %d", frame, frame_size, len(frame_data))
                        return
                    
                    data[:, :, frame] = np.frombuffer(frame_data, dtype=np.int16).reshape((lines, samples))
                
                logging.info("Successfully read %d RF frames.", frames)
                
        except FileNotFoundError:
            logging.error("File not found: %s", self.rf_raw_path)
            raise
        except Exception as e:
            logging.error("Error reading RF data: %s", str(e))
            raise

        logging.info("Loaded %d raw RF frames of size %d x %d (lines x samples)", data.shape[2], data.shape[0], data.shape[1])
        
        self.rf_raw_hdr, self.rf_raw_timestamps_1d, self.rf_raw_data_3d = hdr, timestamps, data
           
    ###################################################################################
    
    def set_default_tgc_data(self) -> None:
        """
        Extracts and processes TGC (Time Gain Compensation) data from the RF YAML object.

        This function parses key-value pairs from the RF TGC data, extracts depth and 
        dB values, and stores them in a structured format within `self.default_tgc_data`.

        Attributes:
            self.default_tgc_data (list[dict]): A list of dictionaries, each containing:
                - 'depth' (float): The extracted depth value from the length key.
                - 'dB' (float): The extracted dB value from the dB key.

        Returns:
            None
        """
        rf_tgc = self.rf_yml_obj.rf_tgc  # Extract the original data
        self.default_tgc_data = []  # Initialize empty list

        for entry in rf_tgc:
            keys = list(entry.keys())  # Extract keys dynamically
            if len(keys) == 2:
                length_key = keys[0]  # e.g., "0.00mm"
                db_key = keys[1]  # e.g., "5.00dB"

                # Extract depth from length_key
                depth_match = re.findall(r"[\d.]+", length_key)
                depth_value = float(depth_match[0]) if depth_match else None  # Convert to float

                # Extract dB from db_key
                db_match = re.findall(r"[\d.]+", db_key)
                db_value = float(db_match[0]) if db_match else None  # Convert to float

                if depth_value is not None and db_value is not None:
                    self.default_tgc_data.append({'depth': depth_value, 'dB': db_value})

    ###################################################################################

    def remove_tgc(self, visualize=False) -> None:
        """
        Processes RF data to remove the time-gain compensation (TGC) effect.

        Attributes:
            self.clean_tgc_data (dict): A dictionary where:
                - Keys (str): Timestamps from `env_tgc_yml_obj.timestamps`.
                - Values (list[dict]): Processed TGC data lists, falling back to 
                previous valid data or default TGC data if necessary.

        Returns:
            None
        """
        previous_data_dict = None

        # Ensure each timestamp has valid TGC data
        if self.env_tgc_yml_path is not None and len(self.env_tgc_yml_obj.timestamps.items()):
            for timestamp, data_list in self.env_tgc_yml_obj.timestamps.items():
                logging.debug(f"Processing {timestamp}: {data_list}")

                if isinstance(data_list, list) and data_list:  # Check if it's a non-empty list
                    data_dict = data_list  
                    previous_data_dict = data_dict 
                else:
                    logging.warning(f"No valid data for {timestamp}, using previous valid data.")
                    data_dict = previous_data_dict if previous_data_dict else self.default_tgc_data

                self.clean_tgc_data[timestamp] = data_dict
                logging.info(f"Final data stored for {timestamp}: {data_dict}")
            

            num_frames = self.rf_raw_data_3d.shape[2]
            
            rf_no_tgc_raw = np.copy(self.rf_raw_data_3d).astype(np.float64)

            for frame in range(num_frames):
                # Initialize TGC array with zeros
                tgc_array_dB = np.zeros(len(self.full_depth_mm), dtype=np.float16)
                values = list(self.clean_tgc_data.values())[frame]

                # Extract depth and dB values
                depths_mm = np.array([entry['depth'] for entry in values])
                tgc_dB = np.array([entry['dB'] for entry in values])

                # Perform interpolation
                for i, depth in enumerate(self.full_depth_mm):
                    mask = depths_mm > depth
                    if mask.any():
                        idx = mask.argmax()
                        if idx == 0:
                            tgc_array_dB[i] = tgc_dB[idx]
                        else:
                            x1, y1 = depths_mm[idx - 1], tgc_dB[idx - 1]
                            x2, y2 = depths_mm[idx], tgc_dB[idx]
                            tgc_array_dB[i] = y1 + (y2 - y1) * (depth - x1) / (x2 - x1)
                    else:
                        tgc_array_dB[i] = tgc_dB[-1]

                # Visualization (optional)
                if visualize:
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.full_depth_mm, tgc_array_dB, label=f'Frame {frame}', color='blue')
                    plt.title(f'Attenuation vs Depth for Frame {frame}')
                    plt.xlabel('Depth (mm)')
                    plt.ylabel('Attenuation (dB)')
                    plt.grid()
                    plt.legend()
                    plt.show()

                # Apply correction
                trimmed_tgc_dB = tgc_array_dB[self.delay_samples: self.delay_samples + self.signal_length]
                trimmed_tgc_coefficient = 10 ** (trimmed_tgc_dB / 20)   
                
                for line in range(rf_no_tgc_raw.shape[0]):
                    rf_no_tgc_raw[line, :, frame] /= trimmed_tgc_coefficient
            
        else:
            logging.warning("No TGC data inputted. No TGC was used.")
            rf_no_tgc_raw = np.copy(self.rf_raw_data_3d).astype(np.float64)
                
        self.rf_no_tgc_raw_data_3d = rf_no_tgc_raw
        
    ###################################################################################

    def create_full_imaging_depth_array(self):
        """
        Generates an array representing the full imaging depth in millimeters.

        This function calculates the depth array based on the imaging depth and delay samples,
        ensuring accurate mapping of signal depth.

        Returns:
        --------
        tuple: (full_imaging_depth_array_1d_mm, trimmed_signal_length, delay_samples)
            - full_imaging_depth_array_1d_mm: 1D NumPy array of depth values in mm.
            - trimmed_signal_length: Length of the trimmed RF signal.
            - delay_samples: Number of delay samples before the valid signal starts.
        """
        trimmed_signal_length = self.rf_raw_data_3d.shape[1]
        delay_samples = self.rf_yml_obj.rf_delay_samples
        imaging_depth_mm = self.extract_first_num(self.rf_yml_obj.rf_imaging_depth)

        # Compute full depth array
        full_signal_length = trimmed_signal_length + delay_samples
        depth_array_mm = np.linspace(0, imaging_depth_mm, full_signal_length, dtype=np.float16)

        return depth_array_mm, trimmed_signal_length, delay_samples

    ###################################################################################
    
    def set_full_signal(self):
        """
        Extends the RF signal data by adding delay samples and padding to match standard dimensions.
        
        This method:
        1. Adds delay samples to the beginning of the signal
        2. Pads the signal to match standard dimensions based on probe type (C3 or L15)
        
        Raises:
            ValueError: If probe configuration is invalid or missing
            IndexError: If array shapes are incompatible for padding
            AttributeError: If required attributes are missing
        """
        try:
            # Add delay samples
            delay_samples = self.rf_yml_obj.rf_delay_samples
            logging.info(f"Adding {delay_samples} delay samples to the beginning of signals")
            
            try:
                original_shape = self.rf_raw_data_3d.shape
                logging.debug(f"Original signal shape before padding: {original_shape}")
                
                # Add delay samples to both signals
                self.rf_raw_data_3d = np.pad(self.rf_raw_data_3d, ((0, 0), (0, delay_samples), (0, 0)), mode='constant')
                self.rf_no_tgc_raw_data_3d = np.pad(self.rf_no_tgc_raw_data_3d, ((0, 0), (0, delay_samples), (0, 0)), mode='constant')
                logging.info(f"Signal shapes after delay padding: {self.rf_raw_data_3d.shape}")
            
            except (ValueError, IndexError) as e:
                logging.error(f"Error during delay sample padding: {str(e)}")
                raise
            
            try:
                # Determine probe type and pad accordingly
                if not hasattr(self.rf_yml_obj, 'rf_probe'):
                    raise AttributeError("RF probe configuration missing")
                    
                if self.rf_yml_obj.rf_probe.get("radius") is not None:
                    target_samples = self.max_samples_per_line_C3
                    probe_type = "C3"
                else:
                    target_samples = self.max_samples_per_line_L15
                    probe_type = "L15"
                
                logging.info(f"Detected {probe_type} probe - padding signals to {target_samples} samples per line")
                
                # Calculate required padding
                current_samples = self.rf_raw_data_3d.shape[1]
                padding_needed = target_samples - current_samples
                
                if padding_needed < 0:
                    raise ValueError(f"Current samples ({current_samples}) exceeds target samples ({target_samples})")
                
                if padding_needed > 0:
                    try:
                        self.rf_raw_data_3d = np.pad(self.rf_raw_data_3d, ((0, 0), (0, padding_needed), (0, 0)), mode='constant')
                        self.rf_no_tgc_raw_data_3d = np.pad(self.rf_no_tgc_raw_data_3d, ((0, 0), (0, padding_needed), (0, 0)), mode='constant')
                        logging.info(f"Added {padding_needed} padding samples to reach standard dimensions")
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error during final padding: {str(e)}")
                        raise
                
                final_shape = self.rf_raw_data_3d.shape
                logging.info(f"Full signal creation complete. Final shape: {final_shape}")
                
            except (ValueError, IndexError, AttributeError) as e:
                logging.error(f"Error during probe-specific padding: {str(e)}")
                raise
                
        except Exception as e:
            logging.error(f"Unexpected error in set_full_signal: {str(e)}")
            raise
        
    ###################################################################################
    
    def set_data_of_clarius_info_struct(self):
        
        try:
            self.clarius_info_struct.width1 = self.rf_yml_obj.rf_probe["radius"] * 2
            self.scan_converted = True
        except KeyError:
            self.scan_converted = False
            
        self.clarius_info_struct.endDepth1 = self.extract_first_num(self.rf_yml_obj.rf_imaging_depth) / 100 # [m]
        self.clarius_info_struct.startDepth1 = self.clarius_info_struct.endDepth1 / 4 # [m]
        self.clarius_info_struct.samplingFrequency = self.extract_first_num(self.rf_yml_obj.rf_sampling_rate) * 1e6 # [Hz]
        self.clarius_info_struct.tilt1 = 0
        self.clarius_info_struct.samplesPerLine = self.rf_yml_obj.rf_size['samples per line']
        self.clarius_info_struct.numLines = self.rf_yml_obj.rf_size['number of lines']
        self.clarius_info_struct.sampleSize = self.rf_yml_obj.rf_size['sample size']        
        self.clarius_info_struct.centerFrequency = self.extract_first_num(self.rf_yml_obj.rf_transmit_frequency) * 1e6 # [Hz]
        self.clarius_info_struct.minFrequency = 0
        self.clarius_info_struct.maxFrequency = self.clarius_info_struct.centerFrequency * 2
        self.clarius_info_struct.lowBandFreq = int(self.clarius_info_struct.centerFrequency / 2)
        self.clarius_info_struct.upBandFreq = int(self.clarius_info_struct.centerFrequency * 1.5)
        self.clarius_info_struct.clipFact = 0.95
        self.clarius_info_struct.dynRange = 50
        
    ###################################################################################
    
    def reconstruct_bmode(self):
        if self.use_tgc:
            rf_atgc = self.rf_raw_data_3d
            tgc_envelope_3d = get_signal_envelope_xd(self.rf_raw_data_3d, hilbert_transform_axis=self.hilbert_transform_axis)
            bmode = 20 * np.log10(tgc_envelope_3d)
        else:
            rf_atgc = self.rf_no_tgc_raw_data_3d
            bmode = 20 * np.log10(self.no_tgc_envelope_3d)
        bmode = np.transpose(bmode, (1, 0, 2))
        rf_atgc = np.transpose(rf_atgc, (1, 0, 2))
        
        # B-Mode pixel formatting
        max_value = np.amax(bmode); clip_max = self.clarius_info_struct.clipFact * max_value
        bmode = np.clip(bmode, clip_max - self.clarius_info_struct.dynRange, clip_max)
        bmode -= np.amin(bmode); bmode *= 255 / np.amax(bmode)
        
        if self.scan_converted:
            
            scBmodeStruct, hCm1, wCm1 = scanConvert(bmode[:,:,0],
                                                    self.clarius_info_struct.width1,
                                                    self.clarius_info_struct.tilt1,
                                                    self.clarius_info_struct.startDepth1, 
                                                    self.clarius_info_struct.endDepth1,
                                                    desiredHeight=2000)
            
            scBmodes = np.array([scanConvert(bmode[:,:,i],
                                             self.clarius_info_struct.width1,
                                             self.clarius_info_struct.tilt1,
                                             self.clarius_info_struct.startDepth1, 
                                             self.clarius_info_struct.endDepth1,
                                             desiredHeight=2000)[0].scArr for i in tqdm(range(rf_atgc.shape[2]))])
            
            self.clarius_info_struct.yResRF =  self.clarius_info_struct.endDepth1*1000 / scBmodeStruct.scArr.shape[0]
            self.clarius_info_struct.xResRF = self.clarius_info_struct.yResRF * (scBmodeStruct.scArr.shape[0]/scBmodeStruct.scArr.shape[1]) # placeholder
            self.clarius_info_struct.axialRes = hCm1*10 / scBmodeStruct.scArr.shape[0]
            self.clarius_info_struct.lateralRes = wCm1*10 / scBmodeStruct.scArr.shape[1]
            self.clarius_info_struct.depth = hCm1*10 #mm
            self.clarius_info_struct.width = wCm1*10 #mm
            
            self.clarius_data_struct.scBmodeStruct = scBmodeStruct
            self.clarius_data_struct.scBmode = scBmodes
            self.clarius_data_struct.bMode = np.transpose(bmode, (2, 0, 1))
            self.clarius_data_struct.rf = np.transpose(rf_atgc, (2, 0, 1))
            
        else:
            self.clarius_info_struct.yResRF = self.clarius_info_struct.endDepth1*1000 / bmode.shape[0] # mm/pixel
            self.clarius_info_struct.xResRF = self.clarius_info_struct.yResRF * (bmode.shape[0]/bmode.shape[1]) # placeholder
            self.clarius_info_struct.axialRes = self.clarius_info_struct.yResRF #mm
            self.clarius_info_struct.lateralRes = self.clarius_info_struct.xResRF #mm
            self.clarius_info_struct.depth = self.clarius_info_struct.endDepth1*1000 #mm
            self.clarius_info_struct.width = self.clarius_info_struct.endDepth1*1000 #mm
            
            self.clarius_data_struct.bMode = np.transpose(bmode, (2, 0, 1))
            self.clarius_data_struct.rf = np.transpose(rf_atgc, (2, 0, 1))
            
    ###################################################################################
    @staticmethod
    def extract_first_num(input_string: str) -> float | int | None:
        """
        Extracts the first number (integer or floating-point) from a string.
        Returns an int if the number is an integer, otherwise returns a float.
        If no number is found, returns None.
        """
        match = re.search(r"\d+\.\d+|\d+", input_string)  # Match float first, then integer
        if match:
            number = match.group()
            return float(number) if '.' in number else int(number)  # Convert appropriately
        return None
    
    ###################################################################################
    class YmlParser():
        """
        This class reads YAML file data related to ultrasound imaging parameters. 
        It extracts information from two YAML files:
        
        1. `*rf.yml` - Contains data such as sampling frequency, probe details, imaging parameters, 
        and transmission settings.
        2. `*env.tgc.yml` - Contains time gain compensation (TGC) data.

        The extracted data is used to generate new imaging data without TGC, 
        requiring TGC values for processing.
        """
        ###################################################################################
        
        def __init__(self, yml_path):
            
            self.path: str = yml_path
            self.valid_versions: list[str] = ['12.0.1-673']
            
            # rf.yml
            self.rf_software_version: str
            self.rf_iso_time_date: str
            self.rf_probe: dict
            self.rf_frames: int
            self.rf_frame_rate: str
            self.rf_transmit_frequency: str
            self.rf_imaging_depth: str
            self.rf_focal_depth: str
            self.rf_auto_gain: bool
            self.rf_mla: bool
            self.rf_tgc: dict
            self.rf_size: dict
            self.rf_type: str
            self.rf_compression: str
            self.rf_sampling_rate: float
            self.rf_delay_samples: int
            self.rf_lines: dict
            self.rf_focus: dict
            
            # env.tgc.yml
            self.frames: int
            self.timestamps: dict
            
            # Determine extension
            filename = os.path.basename(self.path)  # Get the filename only
            parts = filename.rsplit("_", 1)  # Split at the last underscore
            assert len(parts) >= 2, "Filename does not contain an underscore."
            self.extension = parts[-1]  # Get the last part after the underscore
            
            if self.extension == "rf.yml":
                self.load_rf_yml()
            elif self.extension == "env.tgc.yml":
                self.load_env_tgc_yml()
            else:
                raise ValueError(f"Unsupported file extension: {self.extension}. Supported extensions are 'rf.yml' and 'env.tgc.yml'.")
            
        ###################################################################################
        
        def assess_software_version(self, software_version):
            """
            Checks if the current software version (from env.yml or rf.yml) 
            is in the list of valid versions.

            Logs an informational message if the version is valid and a warning if it is not.
            """
            if software_version in self.valid_versions:
                logging.info(f"Version {software_version} is valid.")
            else:
                logging.warning(f"Version {software_version} is not valid. This might cause some problems.")
    
        ###################################################################################
        
        def load_rf_yml(self):
            """
            Loads and parses an RF YAML file if the file extension is "rf.yml".
            The method reads the YAML file, extracts relevant fields, and maps them to class attributes.
            
            Attributes Populated:
            - software_version (str): Software version from the YAML file.
            - iso_time_date (str): ISO formatted timestamp.
            - probe_version (str): Version of the probe.
            - probe_elements (int): Number of probe elements.
            - probe_pitch (float): Probe pitch value.
            - probe_radius (float): Probe radius value.
            - frames (int): Number of frames in the file.
            - frame_rate (float): Frame rate value.
            - transmit_frequency (float): Transmit frequency.
            - imaging_depth (float): Imaging depth.
            - focal_depth (float): Focal depth.
            - auto_gain (bool): Whether auto gain is enabled.
            - mla (bool): Multi-line acquisition status.
            - tgc (dict): Time gain compensation settings.
            - size (dict): Size specifications.
            - type (str): Type information.
            - compression (str): Compression method.
            - sampling_rate (float): Sampling rate value.
            - delay_samples (int): Number of delay samples.
            - lines (dict): Line specifications.
            - focus (list): List of focus parameters.
            
            Raises:
            - Exception: If an error occurs while loading the YAML file.
            """
            try:
                with open(self.path, 'r') as file:
                    data = yaml.safe_load(file)

                # Mapping YAML fields to class attributes
                self.rf_software_version = data.get("software version", None)
                self.rf_iso_time_date = data.get("iso time/date", None)
                self.rf_probe = data.get("probe", {})
                self.rf_frames = data.get("frames", None)
                self.rf_frame_rate = data.get("frame rate", None)
                self.rf_transmit_frequency = data.get("transmit frequency", None)
                self.rf_imaging_depth = data.get("imaging depth", None)
                self.rf_focal_depth = data.get("focal depth", None)
                self.rf_auto_gain: bool = data.get("auto gain", None)
                self.rf_mla: bool = data.get("mla", None)
                self.rf_tgc = data.get("tgc", {})
                self.rf_size = data.get("size", {})
                self.rf_type = data.get("type", None)
                self.rf_compression = data.get("compression", None)
                self.rf_sampling_rate = data.get("sampling rate", None)
                self.rf_delay_samples = data.get("delay samples", None)
                self.rf_lines = data.get("lines", {})
                self.rf_focus = data.get("focus", [{}])
                self.assess_software_version(self.rf_software_version)
                
            except Exception as e:
                logging.error(f"Error loading YAML file: {e}")

        ###################################################################################

        def load_env_tgc_yml(self):
            """
            Loads and parses an environmental TGC YAML file if the file extension is "env.tgc.yml".
            The method reads the YAML file, extracts relevant fields, and maps them to class attributes.
            
            Attributes Populated:
            - frames (int): Number of frames specified in the file.
            - timestamps (dict): Dictionary mapping timestamps to lists of depth and dB values.
            Each entry in the list is a dictionary with:
                - depth (float): Depth value in millimeters.
                - dB (float): Gain value in decibels.
            
            Raises:
            - Exception: If an error occurs while loading the YAML file.
            """
            try:
                with open(self.path, 'r') as file:
                    lines = file.readlines()

                self.timestamps = {}
                current_timestamp = None
                self.frames = None

                for line in lines:
                    line = line.strip()

                    if line.startswith("frames:"):
                        self.frames = int(line.split(":")[1].strip())

                    elif line.startswith("timestamp:"):
                        # Extract timestamp value
                        current_timestamp = int(line.split(":")[1].strip())
                        self.timestamps[current_timestamp] = []

                    elif current_timestamp is not None and line.startswith("- {"):
                        # Extract depth and dB values
                        values = line.replace("- {", "").replace("}", "").split(", ")
                        depth = float(values[0].replace("mm", "").strip())
                        dB = float(values[1].replace("dB", "").strip())

                        # Store in list under the current timestamp
                        self.timestamps[current_timestamp].append({"depth": depth, "dB": dB})
                
            except Exception as e:
                logging.error(f"Error loading YAML file: {e}")

        ###################################################################################
        
    ###################################################################################
    class ParserVisualizations():
        """
        This class is responsible for visualizing the data processed by the ClariusParser.
        It includes methods to plot 1D signals, their FFTs, and 2D images of the envelope.
        """
        
        def __init__(self, rf_raw_data_3d: np.ndarray, rf_no_tgc_raw_data_3d: np.ndarray, default_frame: int, 
                     rf_sampling_rate: int, full_depth_mm: np.ndarray, delay_samples: int,
                     hilbert_transform_axis: int = 1):
            self.rf_raw_data_3d = rf_raw_data_3d
            self.rf_no_tgc_raw_data_3d = rf_no_tgc_raw_data_3d
            self.default_frame = default_frame
            self.hilbert_transform_axis = hilbert_transform_axis
            self.rf_sampling_rate = rf_sampling_rate
            self.full_depth_mm = full_depth_mm
            self.delay_samples = delay_samples
        
        ###################################################################################
          
        def image_envelope_2d(self, envelope_type: str, title: str, clip_fact: float, dyn_range: float) -> None:
            """
            Plots a 2D signal envelope in decibels.

            This function takes a 2D array representing a signal envelope, applies 
            a rotation and flipping transformation, converts it to a logarithmic 
            decibel scale, applies clipping and normalization, and displays it as an image plot.

            Args:
                envelope_type (str): 'TGC' or 'No TGC'
                title (str): The title of the plot.
                clip_fact (float): Clipping factor for dynamic range.
                dyn_range (float): Dynamic range in dB.

            Returns:
                None
            """
            logging.info("Starting the plot function.")
                        
            if envelope_type == "TGC":
                tgc_signal_2d = self.rf_raw_data_3d[:, :, self.default_frame]
                envelope = get_signal_envelope_xd(tgc_signal_2d, hilbert_transform_axis=self.hilbert_transform_axis)
            elif envelope_type == "No TGC":
                no_tgc_signal_2d = self.rf_no_tgc_raw_data_3d[:, :, self.default_frame]
                envelope = get_signal_envelope_xd(no_tgc_signal_2d, hilbert_transform_axis=self.hilbert_transform_axis)
            else:
                raise ValueError("Invalid envelope type. Use 'TGC' or 'No TGC'.")
            
            # Flip the rotated array horizontally
            rotated_flipped_array = self.rotate_flip(envelope)

            log_envelope_2d = 20 * np.log10(np.abs(1 + rotated_flipped_array))

            # Step 2: Clip and normalize
            clipped_max = clip_fact * np.amax(log_envelope_2d)
            log_envelope_2d = np.clip(log_envelope_2d, clipped_max - dyn_range, clipped_max)
            log_envelope_2d -= np.amin(log_envelope_2d)
            log_envelope_2d *= (255 / np.amax(log_envelope_2d))

            logging.debug("Calculated and normalized log envelope 2D.")

            plt.figure(figsize=(8, 6))
            plt.imshow(log_envelope_2d, cmap='gray', aspect='auto')
            plt.title(title)
            plt.colorbar()
            logging.info("Displayed 2D Signal Envelope.")

            plt.tight_layout()
            plt.show()
            logging.info("Plotting completed and displayed.")

        ###################################################################################

        def plot_1d_signal_and_fft(self, signal_type: str, title: str):
            """
            Plots a 1D signal and its corresponding FFT.
            
            Args:
                signal_type (str): Type of the signal to be plotted ('TGC' or 'No TGC').
                title (str): Title for the plots.
            """
            if signal_type == "TGC":
                tgc_signal_2d = self.rf_raw_data_3d[:, :, self.default_frame]
                signal_1d = tgc_signal_2d[tgc_signal_2d.shape[0] // 2, :]
            elif signal_type == "No TGC":
                no_tgc_signal_2d = self.rf_no_tgc_raw_data_3d[:, :, self.default_frame]
                signal_1d = no_tgc_signal_2d[no_tgc_signal_2d.shape[0] // 2, :]
            else:
                raise ValueError("Invalid signal type. Use 'TGC' or 'No TGC'.")
                
            sampling_rate_Hz = ClariusParser.extract_first_num(self.rf_sampling_rate) * 1e6
                            
            # Get envelope
            envelope_1d = get_signal_envelope_xd(signal_1d, hilbert_transform_axis=0)
            
            # Set trimmed imaging depth array in cm
            trimmed_depth_array_1d_cm = np.array(self.full_depth_mm[self.delay_samples:] * 0.1, dtype=np.float64)  # Convert mm to cm

            # Calculate FFT
            fft_signal = np.fft.fft(signal_1d)
            fft_magnitude = np.abs(fft_signal)

            # Frequency axis in MHz using the sampling frequency
            num_samples = len(signal_1d)
            freq_Hz = np.fft.fftfreq(num_samples, d=(1 / (sampling_rate_Hz)))  # Frequency in Hz

            # Only take positive frequencies
            positive_freq_indices = freq_Hz >= 0
            freq_positive_MHz = freq_Hz[positive_freq_indices] / 1e6
            fft_magnitude_positive = fft_magnitude[positive_freq_indices]

            # Create figure
            plt.figure(figsize=(14, 4))

            # Plot the original signal with envelope
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
            plt.plot(trimmed_depth_array_1d_cm, signal_1d, color='blue', label='Signal')  
            plt.plot(trimmed_depth_array_1d_cm, envelope_1d, color='red', linestyle='dashed', label='Envelope')  
            plt.title(f'1D Signal Plot {title}')
            plt.xlabel('Depth (cm)')  
            plt.ylabel('Amplitude')
            plt.legend()  # Add legend
            plt.grid(True)

            # Plot the positive FFT
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            plt.plot(freq_positive_MHz, fft_magnitude_positive, color='green')  
            plt.title(f'FFT of Signal {title}')
            plt.xlabel('Frequency (MHz)')  
            plt.ylabel('Magnitude')
            plt.grid(True)

            plt.tight_layout()  
            plt.show()
            
        ###################################################################################
        @staticmethod
        def rotate_flip(two_dimension_array: np.ndarray) -> np.ndarray:
            """
            Rotates the input 2D array counterclockwise by 90 degrees and then flips it vertically.
            
            Args:
                two_dimension_array (np.ndarray): The input 2D array to be transformed.
            
            Returns:
                np.ndarray: The rotated and flipped 2D array.
            """
            # Rotate the input array counterclockwise by 90 degrees
            rotated_array = np.rot90(two_dimension_array)
            
            # Flip the rotated array horizontally
            rotated_flipped_array = np.flipud(rotated_array)
            
            return rotated_flipped_array
        
    ###################################################################################
    class ClariusInfoStruct(InfoStruct):
        """
        ClariusInfoStruct is a subclass of InfoStruct that represents information 
        about an ultrasound scan's data structure. 

        Attributes:
            samplesPerLine (int): The number of samples per line in the scan.
            numLines (int): The total number of lines in the scan.
            sampleSize (str): The size of each sample, typically represented as a string.
        """
        
        def __init__(self):
            """
            Initializes a ClariusInfoStruct object with essential attributes related 
            to ultrasound scan data structure.
            """
            super().__init__()
            
            self.samplesPerLine: int  # Number of samples per line
            self.numLines: int  # Total number of lines
            self.sampleSize: str  # Size of each sample
    
###################################################################################



# entrypoint function
###################################################################################

def clariusRfParser(imgFilename: str,
                    imgTgcFilename: str,
                    infoFilename: str, 
                    phantomFilename: str,
                    phantomTgcFilename: str,
                    phantomInfoFilename: str,
                    visualize: bool = False,
                    use_tgc: bool = False) -> tuple:

    main_sample_obj    = ClariusParser(imgFilename, imgTgcFilename, infoFilename, visualize=visualize, use_tgc=use_tgc)
    phantom_sample_obj = ClariusParser(phantomFilename, phantomTgcFilename, phantomInfoFilename, visualize=visualize, use_tgc=use_tgc)

    imgData       = main_sample_obj.clarius_data_struct
    imgInfo       = main_sample_obj.clarius_info_struct
    scanConverted = main_sample_obj.scan_converted
    
    refData       = phantom_sample_obj.clarius_data_struct   
    refInfo       = phantom_sample_obj.clarius_info_struct
    scanConverted = phantom_sample_obj.scan_converted
            
    return imgData, imgInfo, refData, refInfo, scanConverted

###################################################################################

def clariusRfParserWrapper(img_folder: str, ref_folder: str, visualize: bool = False, use_tgc: bool = False) \
    -> Tuple[DataOutputStruct, InfoStruct, DataOutputStruct, InfoStruct]:
    """Parse Clarius RF data. Entry-point of entire parser.

    Args:
        img_folder (str): The folder path containing the Clarius RF data files
        ref_folder (str): The folder path containing the Clarius RF phantom data files

    Returns:
        Tuple: Image data, image info, phantom data, and phantom info.
    """
    # Convert to Path objects
    img_folder = Path(img_folder)
    ref_folder = Path(ref_folder)
    
    # Find required files in image folder
    img_raw_files = list(img_folder.glob("*rf.raw"))
    img_tgc_files = list(img_folder.glob("*env.tgc.yml"))
    img_info_files = list(img_folder.glob("*rf.yml"))
    
    # Find required files in reference folder
    ref_raw_files = list(ref_folder.glob("*rf.raw"))
    ref_tgc_files = list(ref_folder.glob("*env.tgc.yml"))
    ref_info_files = list(ref_folder.glob("*rf.yml"))
    
    # Check if we found required files
    assert len(img_raw_files) == 1, f"No rf.raw files found in {img_folder}"
    assert len(img_tgc_files) == 1, f"No env.tgc.yml files found in {img_folder}"
    assert len(img_info_files) == 1, f"No rf.yml files found in {img_folder}"
    assert len(ref_raw_files) == 1, f"No rf.raw files found in {ref_folder}"
    assert len(ref_tgc_files) == 1, f"No env.tgc.yml files found in {ref_folder}"
    assert len(ref_info_files) == 1, f"No rf.yml files found in {ref_folder}"
    
    # Get paths
    img_raw = str(img_raw_files[0])
    img_tgc = str(img_tgc_files[0]) 
    img_info = str(img_info_files[0])
    ref_raw = str(ref_raw_files[0])
    ref_tgc = str(ref_tgc_files[0])
    ref_info = str(ref_info_files[0])
    
    # Use the existing clariusRfParser function from clarius.py
    return clariusRfParser(
        imgFilename=img_raw,
        imgTgcFilename=img_tgc,
        infoFilename=img_info,
        phantomFilename=ref_raw,
        phantomTgcFilename=ref_tgc,
        phantomInfoFilename=ref_info,
        visualize=visualize,
        use_tgc=use_tgc
    ) 
    
###################################################################################
