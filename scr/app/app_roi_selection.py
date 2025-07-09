# Standard library imports for file and system operations
import os
import sys
import pydicom
import yaml

# Imports for image processing and scientific computing
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation and analysis
from scipy.signal import hilbert  # SciPy for signal processing tools
from pathlib import Path

# Imports for creating plots and integrating them into GUI applications
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from matplotlib.figure import Figure  # Matplotlib Figure object for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib canvas for PyQt5

# PyQt5 imports for building the graphical user interface
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFileDialog, QMessageBox,
    QListWidget, QComboBox, QSplitter, QSlider, QLabel
)

#################################################################################################

class ROI_selector_app(QMainWindow):
    
    def __init__(self):
        
        super().__init__()
        
        # Variables for GUI State and Configuration
        self.folders_path = None            # QListWidget for displaying file paths in the GUI
        self.selected_folder = None             # String for the path of the currently selected file
        self.sample_name = None           # Name of the currently selected data sample
        self.save_dir = None
        self.main_path = None

        self.frame_slider = None          # QSlider for navigating through frames
        self.checkbox_c3 = None           # QCheckBox for toggling C3 mode
        self.checkbox_l15 = None          # QCheckBox for toggling L15 mode
        self.first_fig = None             # Matplotlib figure for the first canvas (image display)
        self.second_fig = None            # Matplotlib figure for the second canvas (bar plots)
        self.third_fig = None
        self.first_canvas = None          # Matplotlib canvas for displaying the first figure
        self.second_canvas = None         # Matplotlib canvas for displaying the second figure
        self.third_canvas = None
        self.highlight_rect = None        # Rectangle used to highlight hovered bars in the bar plot
        self.bars = None                  # Bar objects in the second canvas for interaction

        # Variables for Data Management
        self.dcm_data_4d = None
        self.data_3d = None               # 3D numpy array for image or signal data
        self.data_2d = None               # 2D numpy array for the current frame's data
        self.envelope_2d = None           # 2D numpy array for the processed (enveloped) data of the current frame
        self.envelope_3d = None           # 3D numpy array for the processed (enveloped) data
        self.roi_with_anchor = None       # Tuple representing the currently selected ROI (region of interest)
        self.new_roi_with_anchor = None   # Tuple for the newly adjusted ROI after selection
        self.resized_new_roi_with_anchor = None
        self.selected_bar_index = None    # Index of the bar selected in the bar plot
        self.selected_bar_convolution_value = None
        self.x_histogram = None
        self.y_histogram = None
        self.frame = 0                    # Index of the currently displayed frame
        
        # Buttons
        self.large_roi = None
        self.small_roi = None
        self.large_roi_state = False
        self.small_roi_state = False

        # Variables for User Interaction and Modes
        self.c3 = False                   # Boolean flag for enabling/disabling C3 mode
        self.l15 = False                  # Boolean flag for enabling/disabling L15 mode
        self.is_selecting = False         # Boolean indicating if the user is in the process of selecting an ROI
        self.roi_start = None             # Coordinates marking the starting point of ROI selection

        self.__initUI()

    #################################################################################################

    def __initUI(self):
        # Set window properties
        self.setWindowTitle('Application')
        self.setGeometry(100, 100, 1600, 1000)

        # Create the main widget and overall layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Use QHBoxLayout for the overall layout

        # Create a vertical layout for the main content area
        content_layout = QVBoxLayout()

        # Create a title label at the top
        title_label = QLabel("- HybridEcho ROI selection app -")
        title_label.setAlignment(Qt.AlignCenter)  # Center align the text
        title_font = QFont("Arial", 18, QFont.Bold)  # Set font size and bold
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #8B0000;")  # Set title color
        content_layout.addWidget(title_label)  # Add the title to the content layout

        # Create a horizontal layout for the top-right buttons
        top_right_layout = QHBoxLayout()
        content_layout.addLayout(top_right_layout)

        # Spacer to align buttons to the right
        top_right_layout.addStretch()

        # Create buttons and add to the top-right layout
        self.large_roi = QPushButton("Large ROI")
        self.small_roi = QPushButton("Small ROI")

        self.large_roi.clicked.connect(lambda: self.toggle_buttons(self.large_roi, self.small_roi))
        self.small_roi.clicked.connect(lambda: self.toggle_buttons(self.small_roi, self.large_roi))

        top_right_layout.addWidget(self.large_roi)
        top_right_layout.addWidget(self.small_roi)

        # Create a horizontal layout for the rest of the UI below the title
        lower_layout = QHBoxLayout()
        content_layout.addLayout(lower_layout)

        # Configure the splitter for the main content area
        splitter = QSplitter(Qt.Horizontal)
        lower_layout.addWidget(splitter)

        # Set up the left panel
        left_panel = QWidget()
        splitter.addWidget(left_panel)
        left_layout = QVBoxLayout(left_panel)

        # Combo box for device selection
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Select device", "C3", "L15"])
        self.combo_box.currentIndexChanged.connect(self.handle_combo_selection)
        left_layout.addWidget(self.combo_box)

        # Load Data button
        load_data_button = QPushButton('Load Data')
        load_data_button.clicked.connect(self.select_folder)
        left_layout.addWidget(load_data_button)

        # List widget for file paths
        self.folders_path = QListWidget()
        self.folders_path.currentItemChanged.connect(self.load_data)
        left_layout.addWidget(self.folders_path)

        # Set up the right panel
        right_panel = QWidget()
        splitter.addWidget(right_panel)
        right_layout = QVBoxLayout(right_panel)

        # Slider for frame selection (horizontal)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Adjust maximum as needed
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_frame)
        right_layout.addWidget(self.frame_slider)

        # Create a layout for the first two figures side by side
        figures_layout = QHBoxLayout()

        # First figure canvas
        self.first_fig = Figure(figsize=(8, 12))
        self.first_canvas = FigureCanvas(self.first_fig)
        figures_layout.addWidget(self.first_canvas)
        self.first_canvas.mpl_connect('button_press_event', self.start_roi_selection)
        self.first_canvas.mpl_connect('motion_notify_event', self.update_roi_selection)
        self.first_canvas.mpl_connect('button_release_event', self.end_roi_selection)

        # Second figure canvas
        self.second_fig = Figure(figsize=(8, 12))
        self.second_canvas = FigureCanvas(self.second_fig)
        figures_layout.addWidget(self.second_canvas)

        # Add the figures layout (side by side) to the right layout
        right_layout.addLayout(figures_layout)

        # Third figure canvas (below the first two figures)
        self.third_fig = Figure(figsize=(8, 8))
        self.third_canvas = FigureCanvas(self.third_fig)
        right_layout.addWidget(self.third_canvas)

        # Buttons for saving and resetting ROIs
        button_layout = QHBoxLayout()
        right_layout.addLayout(button_layout)
        for button_text in ['Save ROIs', 'Reset']:
            button = QPushButton(button_text)
            button.clicked.connect(getattr(self, button_text.lower().replace(' ', '_')))
            button_layout.addWidget(button)

        # Add the content layout to the central widget
        main_layout.addLayout(content_layout)

        # Create and position the vertical slider on the far right
        self.vertical_slider = QSlider(Qt.Vertical)
        self.vertical_slider.setMinimum(0)
        self.vertical_slider.setMaximum(100)  # Adjust as needed
        self.vertical_slider.setValue(0)
        self.vertical_slider.setEnabled(False)  # Start as disabled
        self.vertical_slider.valueChanged.connect(self.update_roi_size)

        # Set a fixed height for the vertical slider to make it shorter
        self.vertical_slider.setFixedHeight(500)  # Adjust the height as needed

        # Center the vertical slider on the right
        slider_container = QVBoxLayout()
        slider_container.addStretch()  # Add space above the slider
        slider_container.addWidget(self.vertical_slider)
        slider_container.addStretch()  # Add space below the slider

        # Add the slider container to the main layout
        slider_widget = QWidget()
        slider_widget.setLayout(slider_container)
        main_layout.addWidget(slider_widget)

    #################################################################################################
    
    
    
    
    # data
    #################################################################################################

    def select_folder(self):
        # Open a dialog for the user to select a directory and store the selected directory path
        self.main_path = QFileDialog.getExistingDirectory(self, "Select Directory")

        # Clear existing items in the list widget
        self.folders_path.clear()

        # Check if a directory path was selected
        if self.main_path:
            # Initialize a list to collect paths of main folders (only top-level directories)
            folders_path = []

            # List only top-level directories within the selected folder
            for directory in os.listdir(self.main_path):
                # Construct full path
                folder_path = os.path.join(self.main_path, directory)
                # Check if it is a directory (not a file)
                if os.path.isdir(folder_path):
                    # Check for subfolders ending with 'extracted'
                    for sub_directory in os.listdir(folder_path):
                        subfolder_path = os.path.join(folder_path, sub_directory)
                        if os.path.isdir(subfolder_path) and sub_directory.endswith("extracted"):
                            # Look for .npy files inside the 'extracted' folder
                            for file in os.listdir(subfolder_path):
                                if file.endswith(".npy"):
                                    # Check for 'C3' or 'L15' in the filename
                                    if (self.c3 and 'C3' in file) or (self.l15 and 'L15' in file):
                                        folders_path.append(folder_path)
                                        break  # Stop checking other files in this folder if one matches
                            else:
                                continue
                            break  # Stop checking other subfolders if a match is found

            # If folders were found, update the file list widget
            if folders_path:
                self.folders_path.addItems(folders_path)
            else:
                # No main folders found, show a message
                QMessageBox.information(
                    self,
                    "No Folders Found",
                    "No top-level directories were found in the selected directory."
                )
        else:
            # No directory was selected, show a warning
            QMessageBox.warning(
                self,
                "No Directory Selected",
                "Please select a directory to load data."
            )

    #################################################################################################

    def update_folder(self):
        if self.main_path:
            # Clear existing items in the list widget
            self.folders_path.clear()

            # Initialize a list to collect paths of main folders
            valid_folders = []

            # List only top-level directories within the selected folder
            for top_level_directory in os.listdir(self.main_path):
                top_level_path = os.path.join(self.main_path, top_level_directory)

                # Check if it's a directory
                if not os.path.isdir(top_level_path):
                    continue

                # Check subfolders in the top-level directory
                for sub_directory in os.listdir(top_level_path):
                    subfolder_path = os.path.join(top_level_path, sub_directory)

                    # Check if subfolder ends with 'extracted' and is a directory
                    if not (os.path.isdir(subfolder_path) and sub_directory.endswith("extracted")):
                        continue

                    # Look for .npy files in the 'extracted' folder
                    for file in os.listdir(subfolder_path):
                        if file.endswith(".npy"):
                            # Check for 'C3' or 'L15' in the filename
                            if (self.c3 and 'C3' in file) or (self.l15 and 'L15' in file):
                                valid_folders.append(top_level_path)
                                break  # Stop checking other files in this folder
                    else:
                        continue
                    break  # Stop checking other subfolders if a match is found

            # Update the file list widget if any valid folders are found
            if valid_folders:
                self.folders_path.addItems(valid_folders)

    #################################################################################################

    def load_data(self):
        """Loads data from the selected folder and initializes required variables."""
        # Reset the current frame index
        self.frame = 0

        # Get the currently selected item from the file path selector
        selected_folder = self.folders_path.currentItem()

        if selected_folder:
            # Extract folder path and sample name
            self.selected_folder = selected_folder.text()
            self.sample_name = self.get_sample_name()

            print(f"Selected filepath: {self.selected_folder}")

            # Attempt to load .npy files
            try:
                self.read_numpy()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load .npy files. Error: {e}")

            # Attempt to load DICOM (.dcm) files
            try:
                self.read_dcm()
            except Exception as e:
                print(e)
                #QMessageBox.warning(self, "Error", f"Failed to load .dcm files. Error: {e}")
                self.display_dcm_empty()

    #################################################################################################
    
    def read_numpy(self):
        
        # Walk through the directory tree starting from the selected directory
        for root, dirs, files in os.walk(self.selected_folder):
            # Check if the current directory ends with 'extracted'
            if root.endswith('extracted'):
                # Iterate over each file in the directory
                for file in files:
                    # Check if the file ends with 'large_rf.raw.npy'
                    if file.endswith('_large_rf.raw.npy'):
                        # Check if C3 or L15 filtering is enabled
                        if (self.c3 and 'C3' in file) or (self.l15 and 'L15' in file):
                            # Append the full path of the file to the list
                            numpy_file_path = os.path.join(root, file)
                            print(numpy_file_path)

    
        # Load the selected .npy file
        self.data_3d = np.load(numpy_file_path)
        
        # Assuming frames are on the first 
        # axis, update the frame slider to the number of frames
        frame_count = self.data_3d.shape[2]
        self.frame_slider.setMaximum(frame_count - 1)
    
        self.data_2d = self.data_3d[:, :, 0]
        
        print("self.data_3d shape: ", self.data_3d.shape)
        print("self.data_2d shape: ", self.data_2d.shape)

        # Initialize a list to store transformed 2D envelope data for each frame
        envelope_2d_list = []
        
        # Loop over each frame to apply signal processing transformations
        for i in range(frame_count):
            
            data_2d = self.data_3d[:, :, i]
            # Apply the Hilbert transform along the signal axis to obtain the analytic signal
            analytic_signal = hilbert(data_2d, axis=1)
            # Compute the magnitude of the analytic signal to get the envelope
            envelope_2d = np.abs(analytic_signal)

            # Apply a logarithmic transformation for better visualization
            envelope_2d = 20 * np.log10(1 + envelope_2d)

            # Rotate and flip the envelope for correct orientation
            envelope_2d = self.rotate_and_flip(envelope_2d)

            # Store the processed envelope data
            envelope_2d_list.append(envelope_2d)

        # Convert the list of 2D envelopes into a 3D numpy array
        self.envelope_3d = np.array(envelope_2d_list)
        self.envelope_2d = self.envelope_3d[0, :, :]
        
        print("self.envelope_3d shape:", self.envelope_3d.shape)
        print("self.envelope_2d shape:", self.envelope_2d.shape)

        # Display the first frame of the processed data
        try:
            x, y, w, h = self.get_roi_info_from_small()
            self.display_envelope_with_roi_from_small(x, y, w, h)
        except:
            self.display_envelope(0)

    #################################################################################################

    def read_dcm(self):
        # Initialize a list to store the paths of matching DCM files
        dcm_files = []

        # Get the list of files in the main folder
        if self.selected_folder:
            for file in os.listdir(self.selected_folder):
                # Construct the full path of the file
                file_path = os.path.join(self.selected_folder, file)

                # Check if it is a file and has the '.dcm' extension
                if os.path.isfile(file_path) and file.endswith('.dcm'):
                    # Apply filtering based on C3 or L15
                    if (self.c3 and file.startswith('0')) or (self.l15 and file.startswith('1')):
                        # Append the full path of the file to the list
                        dcm_files.append(file_path)

        # Process or print the collected DCM file paths
        if dcm_files:
            for dcm_file in dcm_files:
                print(dcm_file)
      
        # Read the DICOM file using pydicom
        dcm = pydicom.dcmread(dcm_files[0])

        # Extract pixel data from the DICOM file
        pixel_array = dcm.pixel_array

        # Load the selected .npy file
        self.dcm_data_4d = np.array(pixel_array)
        
        print("dcm_data_4d shape :", self.dcm_data_4d.shape)
        
        # Display the first frame of the processed data
        self.display_dcm(0)
            
    #################################################################################################
    
    def toggle_c3(self, enabled):
        # Update the 'c3' attribute based on the combo box selection
        self.c3 = enabled
        
        # Print the current state of 'c3' (enabled or disabled)
        print(f"C3 is {'enabled' if self.c3 else 'disabled'}.")
        
    #################################################################################################

    def toggle_l15(self, enabled):
        # Update the 'l15' attribute based on the combo box selection
        self.l15 = enabled
        
        # Print the current state of 'l15' (enabled or disabled)
        print(f"L15 is {'enabled' if self.l15 else 'disabled'}.")
        
    #################################################################################################
    
    def toggle_buttons(self, selected_button, other_button):
        """Deactivate the selected button, activate the other button, and update their states."""
        # Toggle the states
        selected_button.setEnabled(False)  # Disable the selected button
        other_button.setEnabled(True)     # Enable the other button

        # Update the states
        if selected_button == self.large_roi:
            self.large_roi_state = True
            self.small_roi_state = False
        elif selected_button == self.small_roi:
            self.large_roi_state = False
            self.small_roi_state = True

        # Print the current states of the buttons
        print(f"large_roi_state is {self.large_roi_state}.")
        print(f"small_roi_state is {self.small_roi_state}.")

    #################################################################################################
    
    def handle_combo_selection(self):
        # Handle selection from combo box and toggle visibility or activation of related UI elements
        selected_item = self.combo_box.currentText()
        if selected_item == "C3":
            self.toggle_c3(True)
            self.toggle_l15(False)
            self.update_folder()

        elif selected_item == "L15":
            self.toggle_c3(False)
            self.toggle_l15(True)
            self.update_folder()

        else:  # Handle "Select Option" or unselected state
            self.toggle_c3(False)
            self.toggle_l15(False)
            
    #################################################################################################
    
    def get_sample_name(self):
        # Extract the last part of the folder path
        if self.selected_folder:
            return os.path.basename(self.selected_folder)
        else:
            return None  # Handle case where selected_folder is not set

    #################################################################################################
    
    def rotate_and_flip(self, array_2d):
        # Apply rotation and flip transformations sequentially
        return np.flipud(np.rot90(array_2d))
                    
    #################################################################################################

    def get_roi_info_from_small(self):
        # Initialize variables
        yaml_file_path = None
        numpy_file_path = None
        delay_samples = None
        initial_rx_element = None
        final_rx_element = None
        
        # Walk through the directory tree starting from the selected folder
        for root, dirs, files in os.walk(self.selected_folder):
            # Process only directories ending with 'extracted'
            if root.endswith('extracted'):
                for file in files:
                    # Identify and process YAML files
                    if file.endswith('_small_rf.yaml'):
                        if (self.c3 and 'C3' in file) or (self.l15 and 'L15' in file):
                            yaml_file_path = os.path.join(root, file)
                            print(f"YAML file found: {yaml_file_path}")
                            
                            # Load data from the YAML file
                            if os.path.exists(yaml_file_path):
                                with open(yaml_file_path, 'r') as yaml_file:
                                    rf_yaml_data = yaml.safe_load(yaml_file)
                                
                                # Extract delay samples
                                delay_samples = rf_yaml_data.get('delay samples', 0)
                                # Extract RX elements if 'lines' key exists
                                lines = rf_yaml_data.get('lines', [])
                                if lines:
                                    initial_rx_element = lines[0].get('rx element', 0)
                                    final_rx_element = lines[-1].get('rx element', 0)
                                    
                    # Identify and process Numpy files
                    elif file.endswith('_small_rf.raw.npy'):
                        if (self.c3 and 'C3' in file) or (self.l15 and 'L15' in file):
                            numpy_file_path = os.path.join(root, file)
                            print(f"Numpy file found: {numpy_file_path}")
        
        # Ensure all required files and variables are properly set
        if yaml_file_path is None or numpy_file_path is None:
            raise FileNotFoundError("Required YAML or Numpy file not found.")
        if delay_samples is None or initial_rx_element is None or final_rx_element is None:
            raise ValueError("Missing data in the YAML file.")
        
        # Load data from the Numpy file
        data_3d = np.load(numpy_file_path)
        
        # Calculate the ROI parameters
        h1_with_delay = delay_samples
        h2_with_delay = delay_samples + data_3d.shape[1]
        
        x = initial_rx_element
        y = h1_with_delay
        w = final_rx_element - initial_rx_element
        h = h2_with_delay - h1_with_delay
        
        return x, y, w, h

    #################################################################################################

    def activate_roi_slider(self):
        """Enable or disable the vertical slider based on self.new_roi_with_anchor."""
        self.vertical_slider.setEnabled(self.new_roi_with_anchor is not None)
        
        _, _, w, _ = self.new_roi_with_anchor
        
        self.vertical_slider.setMaximum(w//2 - 1)  # Adjust as needed
        self.vertical_slider.setMinimum(0)
        self.vertical_slider.setValue(0)
        
        self.resized_new_roi_with_anchor = self.new_roi_with_anchor

    #################################################################################################


    
    
    # display
    #################################################################################################
    
    def display_envelope(self, frame_index):
        # Verify the presence of a 3D envelope data structure
        if hasattr(self, 'envelope_3d') and self.envelope_3d is not None:
            
            # Clear any existing content from the figure to prepare for new image display
            self.first_fig.clear()
            
            # Get the current axes, or create them if they don't exist
            ax = self.first_fig.gca()
            
            # Extract the 2D image slice from the 3D array at the specified index
            self.envelope_2d = self.envelope_3d[frame_index, :, :]
            
            # Display the 2D image slice with a grayscale color map and adjust the aspect ratio
            ax.imshow(self.envelope_2d, cmap='gray', aspect='auto')
            
            ax.set_xlim(left=0)  # Setting the left x-limit to zero
                                
            # Set the title of the plot to include the sample name
            if hasattr(self, 'sample_name'):
                ax.set_title(f'Sample: {self.sample_name}', fontsize=16)
                       
            # Refresh the canvas to reflect changes
            self.first_canvas.draw()

    #################################################################################################

    def display_dcm(self, frame_index):
        # Verify the presence of a 3D envelope data structure
        if hasattr(self, 'dcm_data_4d') and self.dcm_data_4d is not None:
            
            # Clear any existing content from the figure to prepare for new image display
            self.second_fig.clear()
            
            # Get the current axes, or create them if they don't exist
            ax = self.second_fig.gca()
            
            # Extract the 2D image slice from the 3D array at the specified index
            self.dcm_data_2d = self.dcm_data_4d[frame_index, 150:650, :, 0]
            
            # Display the 2D image slice with a grayscale color map and adjust the aspect ratio
            ax.imshow(self.dcm_data_2d, cmap='gray', aspect='auto')
            
            ax.set_xlim(left=0)  # Setting the left x-limit to zero
                                
            # Set the title of the plot to include the sample name
            if hasattr(self, 'sample_name'):
                ax.set_title(f'Sample: {self.sample_name}', fontsize=16)
                       
            # Refresh the canvas to reflect changes
            self.second_canvas.draw()
            
    #################################################################################################
    
    def display_dcm_empty(self):
        
        # Clear any existing content from the figure to prepare for new image display
        self.second_fig.clear()
        
        # Get the current axes, or create them if they don't exist
        ax = self.second_fig.gca()
                    
        ax.set_xlim(left=0)  # Setting the left x-limit to zero
                                                    
        # Refresh the canvas to reflect changes
        self.second_canvas.draw()
            
    #################################################################################################

    def display_envelope_with_roi(self):
        # Retrieve the current frame index from the slider
        current_frame_index = self.frame_slider.value()
        
        # Display the envelope image using the current frame index
        self.display_envelope(current_frame_index)
        
        # Obtain the current axes
        ax = self.first_fig.gca()
        
        # If an ROI is specified, reapply the ROI rectangle
        if self.roi_with_anchor:
            x, y, w, h = self.roi_with_anchor  # Unpack the ROI coordinates and dimensions
            rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)  # Define the ROI rectangle
            ax.add_patch(rect)  # Add the ROI rectangle to the axes
        
        if hasattr(self, 'sample_name'):
            ax.set_title(f'Sample: {self.sample_name}', fontsize=16)
                       
        # Refresh the canvas to update the image display
        self.first_canvas.draw()
        
    #################################################################################################
    
    def display_envelope_with_roi_from_small(self, x, y, w, h):
               
        # Retrieve the current frame index from the slider
        current_frame_index = self.frame_slider.value()
        
        # Display the envelope image using the current frame index
        self.display_envelope(current_frame_index)
        
        # Obtain the current axes
        ax = self.first_fig.gca()
        
        rect = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=2)  # Define the ROI rectangle
        ax.add_patch(rect)  # Add the ROI rectangle to the axes
        
        if hasattr(self, 'sample_name'):
            ax.set_title(f'Sample: {self.sample_name}', fontsize=16)
                       
        # Refresh the canvas to update the image display
        self.first_canvas.draw()
        
    #################################################################################################

    def update_frame(self, frame):
        # Check if the 3D envelope data exists and update the display with the specified frame
        if hasattr(self, 'envelope_3d') and self.envelope_3d is not None:
            self.display_envelope(frame)
            
        if hasattr(self, 'dcm_data_4d') and self.dcm_data_4d is not None:
            self.display_dcm(frame)
        
        # Update the 2D signal data using the current frame index
        self.data_2d = self.data_3d[:, :, frame]
        self.envelope_2d = self.envelope_3d[:, :, frame]

        # Update the current frame index
        self.frame = frame
        
        # If an ROI is defined, update the display to include the ROI and generate associated plots
        if self.roi_with_anchor:
            self.display_envelope_with_roi()
            self.generate_bar_plot()

    #################################################################################################
    
    def update_roi_size(self, size_deviation):
                      
        x, y, w, h = self.new_roi_with_anchor
         
        if w > size_deviation * 2:
            self.resized_new_roi_with_anchor = (x + size_deviation, y, w - size_deviation * 2, h)
            self.plot_resized_roi_based_on_selected_bar()
            print("self.resized_new_roi_with_anchor: ", self.resized_new_roi_with_anchor)

    #################################################################################################

    
    
    # ROI selection 
    #################################################################################################

    def start_roi_selection(self, event):
        # Start ROI selection if the 2D envelope image is loaded and the event occurred within the axes
        if self.envelope_2d is not None and event.inaxes:
            self.is_selecting = True
            self.roi_start = (int(event.xdata), int(event.ydata))
            print(f"ROI selection started at {self.roi_start}")

    #################################################################################################

    def update_roi_selection(self, event):
        # Update ROI selection if it is currently active and the event is within the axes
        if self.is_selecting and event.inaxes:
            current_pos = (int(event.xdata), int(event.ydata))
            x0, y0 = self.roi_start
            x1, y1 = current_pos
            
            # Define the ROI using the start and current positions
            self.roi_with_anchor = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            
            print(f"ROI with anchor updated to: {self.roi_with_anchor}")

            # Update the display to include the current ROI
            self.display_envelope_with_roi()

    #################################################################################################

    def end_roi_selection(self, event):
        # Finalize ROI selection if it is currently active
        if self.is_selecting:
            self.is_selecting = False
            if self.roi_with_anchor:
                print(f"ROI selection finalized at {self.roi_with_anchor}")
                # Generate plots based on the finalized ROI
                self.generate_bar_plot()
            else:
                print("ROI selection ended without defining an ROI")

    #################################################################################################
    
    def roi_convertor(self, roi_with_anchor):
        # Convert a rectangle defined by an anchor point (x, y) and dimensions (width, depth) into boundaries (h1, h2, v1, v2)
        x, y, width, depth = roi_with_anchor
        
        h1 = y
        h2 = y + depth
        v1 = x
        v2 = x + width
            
        return (h1, h2, v1, v2)  # Return top-left to bottom-right boundaries
            
    #################################################################################################

    
    
    

    # result generation and demostration
    #################################################################################################

    def generate_bar_plot(self):
        
        # Check if the 2D envelope data and region of interest (ROI) are defined
        if self.envelope_2d is not None and self.roi_with_anchor:
              
            # Calculate tissue similarity and get the resultant plots and ROI data
            x_histogram, y_histogram = self.get_convolution_results_subsections(
                                                        roi = self.roi_convertor(self.roi_with_anchor),
                                                        data_2d = self.rotate_and_flip(self.data_2d))
            
            self.x_histogram, self.y_histogram = x_histogram, y_histogram
                        
            # Clear the current figure to ensure no overlay of multiple plots
            self.third_fig.clear()
            
            # Create a bar plot directly on the figure
            ax = self.third_fig.gca()  # Get the current axes, or create one if none exists
            
            self.bars = ax.bar(x_histogram, y_histogram, width=1, picker=True)

            # Initialize or reset the hover rectangle reference
            self.highlight_rect = None
            
            # Redraw the canvas to display changes
            self.third_canvas.draw()
            
            # Connect event handlers for hover and click actions on the bar plot
            self.third_canvas.mpl_connect('motion_notify_event', self.on_hover)
            self.third_canvas.mpl_connect('pick_event', self.on_bar_click)
            
    #################################################################################################
    
    def get_convolution_results(self, roi, data_2d):
        # Retrieve the area above the specified ROI from the data
        above_roi = self.get_above_roi(roi)

        # Convert data to float64 for precision in calculations
        data_2d = np.array(data_2d).astype(np.float64)

        # Extract dimensions and width of the area above the ROI
        h1, h2, v1, v2 = above_roi
        width = v2 - v1

        # Extract kernel from the area above the ROI
        extracted_above_roi_kernel = self.extract_kernel(data_2d, above_roi)

        histogram = []
        # Calculate convolution results for each position across the width of data
        for i in range(max(0, data_2d.shape[1] - width + 1)):
            floating_above_roi = data_2d[h1:h2, i:i + width]
            operation_result = floating_above_roi * extracted_above_roi_kernel
            histogram.append(np.sum(operation_result))

        # Normalize the histogram values to the range [0, 1]
        histogram = self.normalize_with_min_max(histogram, 0, 1)

        # Generate x-coordinates for the histogram plotting
        x_histogram = [i + width // 2 for i in range(len(histogram))]
        y_histogram = histogram

        return x_histogram, y_histogram
    
    #################################################################################################
    
    def get_convolution_results_subsections(self, roi, data_2d, sublet_width=20):
        # Retrieve the area above the specified ROI from the data
        above_roi = self.get_above_roi(roi)

        # Convert data to float64 for precision in calculations
        data_2d = np.array(data_2d).astype(np.float64)

        # Extract dimensions and width of the area above the ROI
        h1, h2, v1, v2 = above_roi
        width = v2 - v1

        histograms = []

        # If the width is less than or equal to the sublet_width
        if width <= sublet_width:
            # Extract kernel for the entire area
            extracted_kernel = self.extract_kernel(data_2d, above_roi)

            # Calculate convolution results for the entire area
            for i in range(data_2d.shape[1] - width + 1):
                floating_area = data_2d[h1:h2, i:i + width]
                operation_result = floating_area * extracted_kernel
                histograms.append(np.sum(operation_result))

        else:  
            # Calculate convolution results for the entire area
            for i in range(data_2d.shape[1] - width + 1):

                # Replace histogram with an average histogram derived from sublets
                sublet_histograms = []

                num_sublets = width // sublet_width  # Number of sublets within the width
                
                # Forward
                for sublet_idx in range(num_sublets):
                    sublet_start = v1 + sublet_idx * sublet_width
                    sublet_end = sublet_start + sublet_width
                    sublet_area = (h1, h2, sublet_start, sublet_end)
                    sublet_kernel = self.extract_kernel(data_2d, sublet_area)
                    floating_area = data_2d[h1:h2, i:i + sublet_width]
                    operation_result = floating_area * sublet_kernel
                    sublet_histograms.append(np.sum(operation_result))
                
                # Backward    
                for sublet_idx in range(num_sublets):
                    sublet_start = v2 - sublet_idx * sublet_width
                    sublet_end = sublet_start + sublet_width
                    sublet_area = (h1, h2, sublet_start, sublet_end)
                    sublet_kernel = self.extract_kernel(data_2d, sublet_area)
                    floating_area = data_2d[h1:h2, i:i + sublet_width]
                    operation_result = floating_area * sublet_kernel
                    sublet_histograms.append(np.sum(operation_result))
                
                avg_sublet_histograms = np.mean(sublet_histograms, axis=0)
                histograms.append(avg_sublet_histograms)
                
        normalized_histograms = self.normalize_with_min_max(histograms, 0, 1)
        normalized_histograms = [0 if value == 1 else value for value in normalized_histograms]

        # Generate x-coordinates for the histogram plotting
        x_histogram = [i + width // 2 for i in range(len(normalized_histograms))]
        y_histogram = normalized_histograms
        
        return x_histogram, y_histogram  
    
    #################################################################################################
    
    def get_above_roi(self, roi):
        # Calculate the region above the specified ROI in an image
        h1, h2, v1, v2 = roi
        return (0, h1, v1, v2)  # Return coordinates from top to the start of ROI

    #################################################################################################

    def normalize_with_min_max(self, data, min_val, max_val):
        # Normalize data to a new range [min_val, max_val] based on its original min and max values
        min_data = np.min(data)  # Find minimum in data
        max_data = np.max(data)  # Find maximum in data
        normalized_data = (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
        return normalized_data

    #################################################################################################
    
    def extract_kernel(self, envelope_2d, roi_coords):
        # Extracts a region from envelope_2D defined by coordinates (h1, h2, v1, v2)
        h1, h2, v1, v2 = roi_coords
        return envelope_2d[h1:h2, v1:v2]

    #################################################################################################

    def on_hover(self, event):
        
        # Check if the hover event is within the axes of the third figure
        if event.inaxes == self.third_fig.gca():
            # Remove the previously highlighted rectangle if it exists
            if self.highlight_rect:
                self.highlight_rect.remove()
                self.highlight_rect = None

            # Iterate over all bars in the plot to check if the hover is over one of them
            for bar in self.bars:
                # If the mouse is over a bar, draw a rectangle around it
                if bar.contains(event)[0]:
                    x = bar.get_x()
                    y = bar.get_y()
                    width = bar.get_width()
                    height = bar.get_height()
                    # Create a rectangle patch with a green outline and no fill
                    self.highlight_rect = self.third_fig.gca().add_patch(
                        plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='green', facecolor='none')
                    )
                    # Only one bar should be highlighted, so break after drawing the rectangle
                    break

            # Redraw the canvas to show the updated visuals
            self.third_canvas.draw()
            
    #################################################################################################
        
    def on_bar_click(self, event):
        
        # Check if the clicked object is a Rectangle (a bar in the bar chart)
        if isinstance(event.artist, plt.Rectangle):
            # Iterate over all bars (patches) in the first axes of the figure
            for bar in event.canvas.figure.get_axes()[0].patches:
                # Set the color of all bars to blue
                bar.set_facecolor('blue')
            
            # Highlight the clicked bar by setting its color to orange
            event.artist.set_facecolor('orange')
            
            # Determine the index of the selected bar based on its x-coordinate
            self.selected_bar_index = int(event.artist.get_x())
            
            # Redraw the third canvas to reflect changes
            self.third_canvas.draw()
            
            # Perform a computation or update based on the selected bar index
            self.compute_roi_based_on_selected_bar()
            
            # Call a function to visually highlight the newly set region
            self.plot_roi_based_on_selected_bar()
            
            self.activate_roi_slider()
            
            correct_index = self.selected_bar_index - self.x_histogram[0] + 1
            self.selected_bar_convolution_value = self.y_histogram[correct_index]
                        
    #################################################################################################    
    
    def compute_roi_based_on_selected_bar(self):
        
        # Check if the region of interest (ROI) with an anchor is already set
        if self.roi_with_anchor:
            # Unpack the current ROI parameters
            _, y, w, h = self.roi_with_anchor
                       
            # Calculate the new x-coordinate for the ROI centered on the selected bar index
            new_x = self.selected_bar_index - w // 2 + 1
            
            # Ensure the new x-coordinate does not go out of bounds
            new_x = max(0, min(new_x, self.envelope_2d.shape[1] - w))
            
            # Set the new ROI with the calculated x-coordinate and existing y, w, h
            self.new_roi_with_anchor = (new_x, y, w, h)
        
    #################################################################################################

    def plot_roi_based_on_selected_bar(self):
        
        # Display the envelope image with the region of interest overlay
        self.display_envelope_with_roi()
        
        # Get the current Axes instance on the figure
        ax = self.first_fig.gca()
        
        # Check if there is a new region of interest defined
        if self.new_roi_with_anchor:
            # Unpack the new ROI dimensions
            x, y, w, h = self.new_roi_with_anchor
            
            # Create a rectangle to represent the ROI
            rect = plt.Rectangle((x, y), w, h, edgecolor='#00FF00', facecolor='none', linewidth=2)
            
            # Add the rectangle to the axes
            ax.add_patch(rect)
            
            # Redraw the canvas to update the display
            self.first_canvas.draw()
            
            # Print the current and new ROI for debugging purposes
            print("self.roi_with_anchor: ", self.roi_with_anchor)
            print("self.new_roi_with_anchor", self.new_roi_with_anchor)
                
    #################################################################################################
    
    def plot_resized_roi_based_on_selected_bar(self):
        
        # Display the envelope image with the region of interest overlay
        self.display_envelope_with_roi()
        
        # Get the current Axes instance on the figure
        ax = self.first_fig.gca()
        
        # Check if there is a new region of interest defined
        if self.new_roi_with_anchor:
            # Unpack the new ROI dimensions
            x, y, w, h = self.resized_new_roi_with_anchor
            
            # Create a rectangle to represent the ROI
            rect = plt.Rectangle((x, y), w, h, edgecolor='#00FF00', facecolor='none', linewidth=2)
            
            # Add the rectangle to the axes
            ax.add_patch(rect)
            
            # Redraw the canvas to update the display
            self.first_canvas.draw()
            
            # Print the current and new ROI for debugging purposes
            print("self.roi_with_anchor: ", self.roi_with_anchor)
            print("self.new_roi_with_anchor", self.new_roi_with_anchor)
            
    #################################################################################################
    
    

    # save, reset
    #################################################################################################

    def save_rois(self):
        try:
            # Validate ROI states
            if not self.large_roi_state and not self.small_roi_state:
                QMessageBox.warning(self, "Save Session", "Error: Both ROI states are False. Nothing to save.")
                return  # Exit the function

            # Convert ROI data using a custom method
            roi = self.roi_convertor(self.roi_with_anchor)
            new_roi = self.roi_convertor(self.resized_new_roi_with_anchor)
            
            if self.l15 and not self.c3:
                device = "L15"
            elif not self.l15 and self.c3:
                device = "C3"
            else:
                device = "Unknown"

            # Prepare DataFrame to store ROI data
            df = pd.DataFrame({
                "sample_name": [self.sample_name],
                "device": [device],
                "frame": [self.frame],
                "h1": [roi[0]],
                "h2": [roi[1]],
                "v1": [roi[2]],
                "v2": [roi[3]],
                "h1_new": [new_roi[0]],
                "h2_new": [new_roi[1]],
                "v1_new": [new_roi[2]],
                "v2_new": [new_roi[3]],
                "selected_bar_convolution_value": [self.selected_bar_convolution_value]
            })

            if not self.save_dir:
                # Allow user to select a directory to save the Excel file
                self.save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Session")
            
            if self.save_dir:
                file_path = os.path.join(self.save_dir, f"{self.sample_name}.xlsx")

                # Determine which sheet to write to based on the ROI state
                sheet_name = "Large_ROI" if self.large_roi_state else "Small_ROI"

                if os.path.exists(file_path):
                    # If the file exists, append the new sheet
                    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # If the file does not exist, create a new file
                    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                QMessageBox.information(self, "Save Session", f"Data saved successfully to {file_path}")
            else:
                QMessageBox.warning(self, "Save Session", "Session save canceled. No directory selected.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {e}")

    #################################################################################################

    def reset(self):

        # Clear the figures to remove any plotted data or images
        self.first_fig.clear()
        self.second_fig.clear()
        self.third_fig.clear()
        
        # Redraw the canvases to update the GUI with cleared figures
        self.first_canvas.draw()
        self.second_canvas.draw()
        self.third_canvas.draw()
        
        # Clear the list widget that holds the file paths
        self.folders_path.clear()
        
        # Reset data variables to ensure there's no leftover data
        self.dcm_data_4d = None
        self.envelope_3d = None          
        self.envelope_2d = None
        self.data_3d = None
        self.data_2d = None
        self.roi_with_anchor = None
        self.new_roi_with_anchor = None
                
        # Reset the slider and selection indexes
        self.frame_slider.setValue(0)
        self.selected_bar_index = None
        
        # Reset other internal state variables as needed
        self.frame = 0
        self.sample_name = None

        # Confirm the reset to the user or for debugging
        print("Reset complete.")

    #################################################################################################






# Main
#################################################################################################

def main():
    app = QApplication(sys.argv)
    ex = ROI_selector_app()
    ex.show()
    sys.exit(app.exec_())
    
#################################################################################################

if __name__ == '__main__':
    main()
    
#################################################################################################