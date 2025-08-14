# Copy this code into a new cell in your Jupyter notebook

### Delete Folders by Search String

import shutil
import os
import time
import stat
import subprocess
from pathlib import Path

def find_folders_by_name(root_path, search_string):
    """
    Recursively find all folders containing the search string in their names.
    
    Args:
        root_path (str or Path): The root path to search in
        search_string (str): The string to search for in folder names
        
    Returns:
        list: List of paths to folders containing the search string
    """
    matching_folders = []
    root = Path(root_path)
    
    if not root.exists():
        print(f"Error: Path '{root_path}' does not exist.")
        return matching_folders
    
    if not root.is_dir():
        print(f"Error: Path '{root_path}' is not a directory.")
        return matching_folders
    
    # Walk through all subdirectories
    for item in root.rglob('*'):
        if item.is_dir() and search_string.lower() in item.name.lower():
            matching_folders.append(item)
    
    return matching_folders

def change_file_attributes(path):
    """
    Change file attributes to make deletion easier.
    
    Args:
        path (Path): Path to file or directory
    """
    try:
        # Remove read-only attribute
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        
        # For directories, also change attributes of all contents
        if path.is_dir():
            for item in path.rglob('*'):
                try:
                    os.chmod(item, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                except:
                    pass
    except Exception as e:
        print(f"  Warning: Could not change attributes for {path}: {e}")

def force_delete_folder(folder_path, max_retries=3, delay=1, aggressive=False):
    """
    Attempt to delete a folder with retries and force options.
    
    Args:
        folder_path (Path): Path to the folder to delete
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between retries in seconds
        aggressive (bool): Use aggressive deletion methods
        
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Try normal deletion first
            shutil.rmtree(folder_path)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}: Permission denied, trying aggressive methods...")
                
                if aggressive:
                    # Try changing file attributes
                    change_file_attributes(folder_path)
                    
                    # Try using Windows command line
                    try:
                        result = subprocess.run(
                            ['cmd', '/c', 'rmdir', '/s', '/q', str(folder_path)],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            return True
                        else:
                            print(f"  Windows command failed: {result.stderr}")
                    except Exception as cmd_error:
                        print(f"  Windows command error: {cmd_error}")
                
                time.sleep(delay)
                continue
            else:
                print(f"  Final attempt failed: {e}")
                return False
        except Exception as e:
            print(f"  Error: {e}")
            return False
    
    return False

def delete_folders_by_name(target_path, search_string, dry_run=True, force_delete=False, skip_failed=True, aggressive=False):
    """
    Delete all folders containing the search string in their names.
    
    Args:
        target_path (str or Path): Path to search for folders
        search_string (str): The string to search for in folder names
        dry_run (bool): If True, only print what would be deleted without actually deleting
        force_delete (bool): If True, attempt to force delete with retries
        skip_failed (bool): If True, continue with other folders if one fails
        aggressive (bool): If True, use aggressive deletion methods (Windows commands, attribute changes)
        
    Returns:
        int: Number of folders deleted
    """
    print(f"Searching for folders containing '{search_string}' in: {target_path}")
    if dry_run:
        print("DRY RUN MODE - No folders will actually be deleted")
    if force_delete:
        print("FORCE DELETE MODE - Will retry failed deletions")
    if aggressive:
        print("AGGRESSIVE MODE - Will use Windows commands and attribute changes")
    print("-" * 50)
    
    # Find matching folders
    matching_folders = find_folders_by_name(target_path, search_string)
    
    if not matching_folders:
        print(f"No folders containing '{search_string}' found.")
        return 0
    
    # Display found folders
    print(f"Found {len(matching_folders)} folder(s) containing '{search_string}':")
    for folder in matching_folders:
        print(f"  - {folder}")
    print()
    
    # Ask for confirmation (unless dry run)
    if not dry_run:
        response = input(f"Are you sure you want to delete these {len(matching_folders)} folder(s)? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return 0
    
    # Delete folders
    deleted_count = 0
    failed_count = 0
    failed_folders = []
    
    for folder_path in matching_folders:
        try:
            if dry_run:
                print(f"[DRY RUN] Would delete: {folder_path}")
            else:
                print(f"Deleting: {folder_path}")
                
                if force_delete:
                    success = force_delete_folder(folder_path, aggressive=aggressive)
                    if success:
                        deleted_count += 1
                    else:
                        failed_count += 1
                        failed_folders.append(folder_path)
                        if not skip_failed:
                            print("Stopping due to failure and skip_failed=False")
                            break
                else:
                    shutil.rmtree(folder_path)
                    deleted_count += 1
                    
        except PermissionError as e:
            failed_count += 1
            failed_folders.append(folder_path)
            print(f"  Permission denied: {e}")
            if not skip_failed:
                print("Stopping due to permission error and skip_failed=False")
                break
        except Exception as e:
            failed_count += 1
            failed_folders.append(folder_path)
            print(f"  Error deleting {folder_path}: {e}")
            if not skip_failed:
                print("Stopping due to error and skip_failed=False")
                break
    
    # Summary
    print("-" * 50)
    if dry_run:
        print(f"DRY RUN: Would have deleted {deleted_count} folder(s)")
    else:
        print(f"Successfully deleted {deleted_count} folder(s)")
        if failed_count > 0:
            print(f"Failed to delete {failed_count} folder(s):")
            for folder in failed_folders:
                print(f"  - {folder}")
            print("\nSuggestions:")
            print("1. Try running with force_delete=True and aggressive=True")
            print("2. Close any applications that might be using these folders")
            print("3. Run as administrator if needed")
            print("4. Check file permissions")
            print("5. Try rebooting and then deleting")
            print("6. Use Windows Explorer to manually delete one folder first")
    
    return deleted_count

# Example usage - DRY RUN FIRST (recommended)
print("=== DRY RUN - Preview what would be deleted ===")
# You can change 'extracted' to any search string you want
delete_folders_by_name(r"C:\johanna_samples", "extracted", dry_run=False, force_delete=True, aggressive=True)

print("\n" + "="*60 + "\n")

# To actually delete, uncomment the line below and set dry_run=False
# delete_folders_by_name("data/samples", "extracted", dry_run=False)

# Example with different search string and force delete:
# delete_folders_by_name("data/samples", "raw", dry_run=False, force_delete=True)

# Example with force delete for access denied issues:
# delete_folders_by_name(r"C:\johanna_samples", "tar_extracted", dry_run=False, force_delete=True)

# Example with aggressive deletion (recommended for stubborn folders):
# delete_folders_by_name(r"C:\johanna_samples", "extracted", dry_run=False, force_delete=True, aggressive=True)
