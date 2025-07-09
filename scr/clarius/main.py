from pathlib import Path
from typing import Union, Optional
import os
import yaml
import logging
from .parser import ClariusTarUnpacker
import logging
import shutil
from pathlib import Path
    

def unpack_clarius_data(
    path: Union[str, Path],
    extraction_mode: str = "multiple_samples",
    ) -> Optional[ClariusTarUnpacker]:
    """
    Unpacks Clarius ultrasound data from tar files and processes them.
    
    This function creates an instance of ClariusTarUnpacker to extract and process
    Clarius ultrasound data files. It handles both single sample and multiple samples
    extraction modes.
    
    Args:
        path (Union[str, Path]): Path to the directory containing the tar files.
                                For single_sample mode: path to specific sample directory
                                For multiple_samples mode: path to directory containing multiple sample folders
        extraction_mode (str): Mode of extraction. Options:
                              - "single_sample": Process a single sample directory
                              - "multiple_samples": Process multiple sample directories (default)
        
    Returns:
        Optional[ClariusTarUnpacker]: Instance of ClariusTarUnpacker if successful, None if failed
    
    Raises:
        ValueError: If extraction_mode is not one of the valid options
        FileNotFoundError: If the specified path does not exist
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Starting Clarius data unpacking process in {extraction_mode} mode")
    logging.info(f"Target path: {path}")
    
    # First, clean up any existing extracted folders
    logging.info("Cleaning up existing extracted folders...")
    cleanup_extracted_folders(path)
    
    # Convert path to Path object if it's a string
    if isinstance(path, str):
        path = Path(path)
        logging.debug(f"Converted string path to Path object: {path}")
    
    # Validate path exists
    if not path.exists():
        logging.error(f"The specified path does not exist: {path}")
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    # Validate extraction mode
    valid_modes = ["single_sample", "multiple_samples"]
    if extraction_mode not in valid_modes:
        logging.error(f"Invalid extraction mode: {extraction_mode}. Must be one of: {valid_modes}")
        raise ValueError(f"Invalid extraction mode. Must be one of: {valid_modes}")
    
    try:
        logging.info("Creating ClariusTarUnpacker instance...")
        
        if extraction_mode == "multiple_samples":
            # Process each sample directory separately first
            all_extracted_folders = []
            
            for sample_dir in path.iterdir():
                if sample_dir.is_dir():
                    logging.info(f"\nProcessing sample directory: {sample_dir}")
                    # Create unpacker for this sample
                    sample_unpacker = ClariusTarUnpacker(
                        path=str(sample_dir),
                        extraction_mode="single_sample"
                    )
                    # Add this sample's folders to our complete list
                    all_extracted_folders.extend(sample_unpacker.extracted_folders_path_list)
                    # Create a temporary unpacker just for renaming
                    temp_unpacker = type('TempUnpacker', (), {'extracted_folders_path_list': sample_unpacker.extracted_folders_path_list})()
                    logging.info(f"Renaming files for sample: {sample_dir.name}")
                    rename_clarius_files(temp_unpacker)
                    
                    # Delete excessive extracted folders after renaming
                    logging.info(f"Cleaning up excessive extracted folders for sample: {sample_dir.name}")
                    delete_excessive_extracted_folders(sample_dir)
                    
                    # Process the remaining extracted folders with ClariusParser
                    logging.info(f"Processing extracted folders with ClariusParser for sample: {sample_dir.name}")
                    process_extracted_folders_with_parser(sample_dir)
            
            # Create final unpacker instance with all folders
            unpacker = type('TempUnpacker', (), {'extracted_folders_path_list': all_extracted_folders})()
            return unpacker
            
        else:  # single_sample mode
            unpacker = ClariusTarUnpacker(
                path=str(path),
                extraction_mode=extraction_mode
            )
            # Rename the Clarius files after unpacking
            logging.info("Renaming Clarius files...")
            rename_clarius_files(unpacker)
            
            # Delete excessive extracted folders after renaming
            logging.info("Cleaning up excessive extracted folders...")
            delete_excessive_extracted_folders(path)
            
            # Process the remaining extracted folders with ClariusParser
            logging.info("Processing extracted folders with ClariusParser...")
            process_extracted_folders_with_parser(path)
            
            return unpacker
        
    except Exception as e:
        logging.error(f"Error during unpacking: {str(e)}", exc_info=True)
        return None


def cleanup_original_files(folder_path: str) -> None:
    """
    Removes original timestamp-based files from an extracted folder,
    keeping only the renamed files (ones with 'large' or 'small' in their names).
    
    Args:
        folder_path (str): Path to the extracted folder to clean up
    """
    import os
    
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

def rename_clarius_files(unpacker: ClariusTarUnpacker) -> bool:
    """
    Renames Clarius files based on device type (C3/L15) and size comparison.
    
    This function processes the extracted files from a ClariusTarUnpacker instance,
    analyzes the YAML files to determine device types and line counts, and renames
    the files accordingly with a format: {device_type}_{size_label}_*.
    After renaming, it removes the original timestamp-based files.
    
    Args:
        unpacker (ClariusTarUnpacker): An instance of ClariusTarUnpacker that has already
                                      extracted and processed files.
        
    Returns:
        bool: True if renaming was successful, False otherwise.
    """
    import yaml
    import os
    import shutil
    
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
            # Look for both .yml and .yaml files
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
                        
                    # Check if required fields exist
                    if 'size' not in yaml_data or 'number of lines' not in yaml_data['size']:
                        logging.warning(f"Missing required fields in {yaml_file_path}")
                        continue
                        
                    device_name = "C3" if 'probe' in yaml_data and 'radius' in yaml_data['probe'] else "L15"
                    number_of_lines = yaml_data['size']['number of lines']
                    device_info.append((folder_path, device_name, number_of_lines))
                    logging.info(f"Detected device {device_name} with {number_of_lines} lines")
                    
                except Exception as e:
                    logging.error(f"Failed to read YAML file {yaml_file_path}: {e}")

        # Check if we found any valid device information
        if not device_info:
            logging.error("No valid device information was found in any YAML files")
            return False

        # Sort by number of lines to classify sizes
        device_info.sort(key=lambda x: x[2], reverse=True)
        largest_number_of_lines = device_info[0][2]  # The highest number of lines

        # Rename files based on the line counts
        copied_count = 0
        for folder_path, device_name, number_of_lines in device_info:
            size_label = "large" if number_of_lines == largest_number_of_lines else "small"
            logging.info(f"Processing {device_name} files ({size_label}) in {folder_path}")

            # Process files in the folder
            files = os.listdir(folder_path)
            for file_name in files:
                if "large" not in file_name and "small" not in file_name:
                    old_file_path = os.path.join(folder_path, file_name)
                    file_base_name = os.path.basename(old_file_path)
                    file_name_without_extension, file_extension = os.path.splitext(file_base_name)

                    # Handle case where there might be multiple extensions (e.g., .raw.lzo)
                    if '.raw' in file_name_without_extension:
                        parts = file_name_without_extension.split('.raw')
                        file_name_without_extension = parts[0]
                        file_extension = '.raw' + file_extension

                    new_file_name = f"{device_name}_{size_label}_" + file_name_without_extension.split("_", 1)[-1] + file_extension
                    new_file_path = os.path.join(folder_path, new_file_name)

                    try:
                        # Copy the file instead of renaming
                        shutil.copy2(old_file_path, new_file_path)
                        copied_count += 1
                    except Exception as e:
                        logging.error(f"Failed to copy {old_file_path} to {new_file_path}: {e}")

        # After all files are renamed, clean up original files in each folder
        logging.info("Cleaning up original files...")
        for folder_path, _, _ in device_info:
            cleanup_original_files(folder_path)

        logging.info(f"Successfully completed renaming {copied_count} files and cleaned up original files")
        return True
        
    except Exception as e:
        logging.error(f"Error occurred during renaming: {e}")
        return False

def determine_probe_type(raw_file: Path) -> str:
    """
    Determine the probe type (C3_large, C3_small, L15_large, L15_small) from the raw file.
    
    Args:
        raw_file (Path): Path to the raw file to analyze
        
    Returns:
        str: Probe type identifier or empty string if cannot be determined
    """
    import yaml
    
    try:
        # Find corresponding YAML file
        yaml_path = raw_file.parent / (raw_file.stem + '.yml')
        if not yaml_path.exists():
            yaml_path = raw_file.parent / (raw_file.stem + '.yaml')
        
        if not yaml_path.exists():
            logging.warning(f"No YAML file found for {raw_file}")
            return ""
            
        # Read YAML data
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        if not yaml_data:
            logging.warning(f"Empty or invalid YAML data in {yaml_path}")
            return ""
            
        # Check required fields
        if 'size' not in yaml_data or 'number of lines' not in yaml_data['size']:
            logging.warning(f"Missing required fields in {yaml_path}")
            return ""
            
        # Determine device type
        device_name = "C3" if 'probe' in yaml_data and 'radius' in yaml_data['probe'] else "L15"
        number_of_lines = yaml_data['size']['number of lines']
        
        # Compare with other files in the same folder to determine size
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

def cleanup_extracted_folders(samples_path: Union[str, Path] = "data/samples") -> None:
    """
    Clean up all folders ending with '_extracted' in the samples directory.
    
    Args:
        samples_path (Union[str, Path]): Path to the samples directory. Defaults to "data/samples"
    """
    # Configure logging if not already configured
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Convert to Path object if string
    samples_path = Path(samples_path)
    logging.debug(f"Cleaning up extracted folders in: {samples_path}")
    
    if not samples_path.exists():
        logging.error(f"Error: Samples directory '{samples_path}' does not exist")
        return
        
    # Find all extracted folders
    extracted_folders = []
    for folder in samples_path.glob("**/*_extracted"):
        if folder.is_dir():
            extracted_folders.append(folder)
    
    if not extracted_folders:
        logging.info("No extracted folders found to clean up")
        return
    
    logging.info(f"Found {len(extracted_folders)} extracted folder(s) to clean up")
        
    # Delete each extracted folder
    for folder in extracted_folders:
        try:
            shutil.rmtree(folder)
            logging.info(f"Deleted: {folder}")
        except Exception as e:
            logging.error(f"Error deleting {folder}: {str(e)}")
    
    logging.info(f"Cleanup complete. Removed {len(extracted_folders)} extracted folder(s)")

def delete_excessive_extracted_folders(sample_folder_path: Union[str, Path]) -> None:
    """
    Delete extracted folders that contain duplicate renamed files.
    Only keeps the first occurrence of each renamed file pattern (C3_large, C3_small, etc.).
    
    Args:
        sample_folder_path (Union[str, Path]): Path to the sample directory containing extracted folders
    """
    seen_files = set()  # To keep track of seen filenames
    
    # Convert to Path object if string
    if isinstance(sample_folder_path, str):
        sample_folder_path = Path(sample_folder_path)
        
    logging.info(f"Checking for excessive extracted folders in: {sample_folder_path}")
    
    for folder in os.listdir(sample_folder_path):
        folder_path = os.path.join(sample_folder_path, folder)
        if os.path.isdir(folder_path) and folder.endswith("extracted"):
            logging.info(f"Checking folder: {folder_path}")
            current_folder_files = set()  # To track files in the current folder

            # Iterate through files in the current extracted folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(x in file for x in ["C3_large", "C3_small", "L15_large", "L15_small"]):
                        # Check if this file has been seen before
                        if file in seen_files:
                            logging.info(f"Duplicate file found: {file} in folder: {folder_path}. Deleting folder.")
                            # Attempt to remove the directory and its contents
                            try:
                                shutil.rmtree(folder_path)  # Removes directory and all its contents
                                logging.info(f"Successfully deleted folder: {folder_path}")
                            except Exception as e:
                                logging.error(f"Error deleting folder {folder_path}: {e}")
                            break
                        else:
                            # Add to the current folder's file set and the seen files
                            current_folder_files.add(file)
                            seen_files.add(file)
            else:
                # If no duplicate was found, log the files in the current folder
                logging.info(f"Unique files in {folder_path}: {current_folder_files}")

def process_extracted_folders_with_parser(sample_folder_path: Union[str, Path]) -> None:
    """
    Process each extracted folder with ClariusParser and save rf_no_tgc_raw_data_3d as numpy file.
    The numpy file will be named based on the source RF raw file (e.g., L15_large_rf.raw -> L15_large_rf_no_tgc.npy)
    
    Args:
        sample_folder_path (Union[str, Path]): Path to the sample directory containing extracted folders
    """
    from .parser import ClariusParser
    import numpy as np
    
    # Convert to Path object if string
    if isinstance(sample_folder_path, str):
        sample_folder_path = Path(sample_folder_path)
        
    logging.info(f"Processing extracted folders with ClariusParser in: {sample_folder_path}")
    
    # Find all extracted folders
    for folder in os.listdir(sample_folder_path):
        folder_path = os.path.join(sample_folder_path, folder)
        if os.path.isdir(folder_path) and folder.endswith("extracted"):
            logging.info(f"Processing folder: {folder_path}")
            
            try:
                # Find required files in the folder
                rf_raw_files = [f for f in os.listdir(folder_path) 
                              if f.endswith('rf.raw') and any(x in f for x in ["C3_large", "C3_small", "L15_large", "L15_small"])]
                env_tgc_yml_files = [f for f in os.listdir(folder_path) if f.endswith('env.tgc.yml')]
                rf_yml_files = [f for f in os.listdir(folder_path) if f.endswith('rf.yml')]
                
                if not (rf_raw_files and env_tgc_yml_files and rf_yml_files):
                    logging.warning(f"Missing required files in {folder_path}")
                    continue
                
                # Use the first matching file of each type
                rf_raw_path = os.path.join(folder_path, rf_raw_files[0])
                env_tgc_yml_path = os.path.join(folder_path, env_tgc_yml_files[0])
                rf_yml_path = os.path.join(folder_path, rf_yml_files[0])
                
                # Create parser instance
                parser = ClariusParser(
                    rf_raw_path=rf_raw_path,
                    env_tgc_yml_path=env_tgc_yml_path,
                    rf_yml_path=rf_yml_path,
                    visualize=False,
                    use_tgc=False
                )
                
                # Get the rf_no_tgc_raw_data_3d
                rf_data = parser.rf_no_tgc_raw_data_3d
                
                # Generate output filename based on the RF raw file name
                rf_raw_name = os.path.splitext(rf_raw_files[0])[0]  # Remove .raw extension
                output_filename = os.path.join(folder_path, f"{rf_raw_name}_no_tgc.npy")
                
                # Save as numpy file
                np.save(output_filename, rf_data)
                logging.info(f"Saved RF data to: {output_filename}")
                
            except Exception as e:
                logging.error(f"Error processing folder {folder_path}: {e}")
