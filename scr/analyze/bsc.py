# Standard library imports
import logging
import os

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Local application/library imports
from src.data import Data

# Logging setup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.optimize import curve_fit

class BSC():

    def __init__(self,
                 folder_path,
                 device,
                 size,
                 signal_type,
                 ac_method,
                 alpha,
                 mode,
                 v1 = None,
                 v2 = None,
                 h1 = None,
                 h2 = None,
                 visualize = None,
                 stft_nperseg = None,
                 stft_noverlap = None) -> None:
        
        # Initialize instance variables
        self.folder_path = folder_path
        self.device = device
        self.size = size
        self.signal_type = signal_type
        self.ac_method = ac_method
        self.mode = mode
        self.alpha = alpha
        self.v1 = v1
        self.v2 = v2
        self.h1 = h1
        self.h2 = h2
        self.visualize = visualize

        self.nperseg = stft_nperseg
        self.noverlap = stft_noverlap
        
        self.stft_results = {}
        self.data_obj = None

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()
        elif self.mode == "single_CF": self.__run_single_CF()

    ###################################################################################
    
    def __run_single(self):
        
        self.prepare_stft_analysis_data_single(mode="1d")
        self.plot_stft_results()       
                                
    ###################################################################################
   
    def __run_multiple(self):
        
        self.prepare_stft_analysis_data_multiple(mode="1d")
        self.plot_stft_results()       
                                
    ###################################################################################
    
    def __run_single_CF(self):
        
        self.prepare_stft_analysis_data_CF()
        
    ###################################################################################
    
    def compute_stft(self, signal_1d, fs, window, nperseg, noverlap):
        logging.debug("Entering compute_stft method.")
        logging.debug(f"Input parameters - fs: {fs}, nperseg: {nperseg}, noverlap: {noverlap}")

        step = nperseg - noverlap
        freqs = np.fft.fftfreq(nperseg, 1/fs)
        time_slices = np.arange(0, len(signal_1d) - nperseg, step)
        stft_matrix = np.zeros((len(freqs), len(time_slices)), dtype=complex)

        logging.debug(f"Step size for slicing: {step}")
        logging.debug(f"Number of time slices: {len(time_slices)}")

        for i, t in enumerate(time_slices):
            logging.debug(f"Processing time slice index: {i}, time: {t/fs} seconds")
            x_segment = signal_1d[t:t+nperseg] * window
            stft_matrix[:, i] = np.fft.fft(x_segment)

        freqs = np.fft.fftshift(freqs)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
        
        logging.debug("STFT computation completed, shifting frequencies and STFT matrix.")
        logging.debug(f"Output frequencies shape: {freqs.shape}, time slices shape: {time_slices.shape}, STFT matrix shape: {stft_matrix.shape}")

        logging.debug("Exiting compute_stft method.")
        return freqs, time_slices / fs, stft_matrix

    ###################################################################################
    
    def gaussian_window(self, window_length_0d, fs_0d):
        logging.debug("Entering gaussian_window method.")
        logging.debug(f"Input parameters - window_length_0d: {window_length_0d}, fs_0d: {fs_0d}")
        
        # Convert the number of points to an integer
        num_points = int(window_length_0d * fs_0d)
        logging.debug(f"Number of points calculated: {num_points}")

        time_1d = np.linspace(0, window_length_0d, num_points)  # Generating appropriate tau range
        logging.debug(f"Time vector generated with length: {len(time_1d)}")

        sigma = window_length_0d / 6  # Standard deviation
        logging.debug(f"Standard deviation (sigma) calculated: {sigma}")
        
        t_0 = window_length_0d / 2  # Central point (mean)
        logging.debug(f"Central point (t_0) calculated: {t_0}")

        amp_1d = np.exp(-((time_1d - t_0)**2) / (2 * sigma**2))
        normalization = np.sqrt(np.sqrt(np.pi)) 
        amp_1d /= normalization
        logging.debug("Amplitude vector calculated and normalized.")

        logging.debug("Exiting gaussian_window method.")
        return time_1d, amp_1d

    ###################################################################################

    def stft_1d(self, signal_1d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        if normalize_signal:
            # Calculate signal energy and normalize if requested
            signal_energy = np.sum(np.abs(signal_1d) ** 2)
            logging.info(f"Signal energy calculated: {signal_energy}")
            
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, times, stft_result

    ###################################################################################
    
    def stft_2d(self, signal_2d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_2d method.")
        
        cumulative_amplitude = None
        total_iterations = signal_2d.shape[0]  # Total iterations now equal to the number of rows
        logging.debug(f"Total iterations set to: {total_iterations}")

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")

            signal_1d = signal_2d[i, :]
            freqs, times, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal=normalize_signal)

            # Initialize cumulative arrays on the first iteration
            if cumulative_amplitude is None:
                cumulative_amplitude = np.zeros_like(amplitude)
                logging.info("Initialized cumulative frequencies and amplitudes.")

            # Add the current frequencies and amplitudes to the cumulative sum
            cumulative_amplitude += amplitude
            logging.debug(f"Updated cumulative frequencies and amplitudes for index {i}.")

        # Calculate the average
        avg_amplitude = cumulative_amplitude / total_iterations
        logging.info("Calculated averaged frequencies and amplitudes.")

        logging.debug("Exiting stft_2d method.")
        return freqs, times, avg_amplitude
    
    ###################################################################################
    
    def stft_2d_CF(self, signal_2d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_2d method.")
        
        # Initialize dictionaries to store frequency, time, and amplitude arrays
        all_freqs = {}
        all_times = {}
        all_amplitudes = {}
        all_central_freq = {}

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")
            
            signal_1d = signal_2d[i, :]
            freqs, times, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal=normalize_signal)
            
            # Filter out zero frequency. Assume zero frequency is at index 0
            valid_indices = np.arange(1, len(freqs))
            
            # Find index of the maximum amplitude for each time point excluding zero frequency
            central_freq = np.array([freqs[valid_indices[amp[valid_indices].argmax()]] for amp in amplitude.T])

            # Store the results in dictionaries with index i as the key
            all_freqs[i] = freqs
            all_times[i] = times
            all_amplitudes[i] = amplitude
            all_central_freq[i] = central_freq

        logging.debug("Exiting stft_2d method.")

        return all_freqs, all_central_freq, all_times, all_amplitudes

    ###################################################################################
    
    def plot_stft(self, time, freqs, stft_result, log=False):
        # Log the start of the plotting process
        logging.info("Starting to plot STFT.")

        # Convert frequencies from Hz to MHz
        freqs_mhz = freqs / 1e6
        logging.debug(f"Frequencies converted to MHz: {freqs_mhz}")

        # Convert time from seconds to microseconds
        time_us = time * 1e6
        logging.debug(f"Time converted to µs: {time_us}")

        # Plot STFT with both positive and negative frequencies
        plt.figure(figsize=(12, 6))

        # Check for logarithm and avoid log(0) by adding a small constant
        if log:
            stft_result = np.log(np.abs(stft_result) + 1e-10)  # Avoid log(0)
            logging.info("Applied logarithmic scaling to STFT result.")

        # Create the plot using imshow
        plt.imshow(np.abs(stft_result), aspect='auto',
                extent=[time_us.min(), time_us.max(), freqs_mhz.min(), freqs_mhz.max()], origin='lower', cmap='jet')
        
        plt.title('STFT with Gaussian Window')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [µs]')
        plt.colorbar(label='Magnitude')

        # Show the plot
        plt.show()
        logging.info("STFT plot displayed successfully.")

    ###################################################################################
    
    def calculate_stft_time(self, time, stft_result):
        # Ensure stft_result has at least one dimension
        if stft_result.ndim < 2:
            raise ValueError("stft_result must have at least two dimensions.")

        # Determine the number of time bins in the STFT result
        num_time_bins = stft_result.shape[1]

        # Calculate the total number of points in the time array
        total_time_points = len(time)

        # Log the original time array length and STFT result size
        logging.info(f"Original time array length: {total_time_points}")
        logging.info(f"Number of time bins in STFT result: {num_time_bins}")

        # Check if we can trim the time array to the required size
        if total_time_points < num_time_bins:
            raise ValueError("The time array is smaller than the STFT result time bins.")

        # Calculate how many points to remove from each side to achieve the target size
        total_points_to_remove = total_time_points - num_time_bins

        # Ensure we remove points from both sides equally
        if total_points_to_remove % 2 == 0:
            points_to_remove_each_side = total_points_to_remove // 2
        else:
            points_to_remove_each_side = total_points_to_remove // 2

        # Trim the time array from both ends
        trimmed_time = time[points_to_remove_each_side:total_time_points - points_to_remove_each_side]
        
        # Log the trimming process
        logging.info(f"Trimming time array: removing {points_to_remove_each_side} points from each side.")

        # Ensure trimmed_time has the correct size
        if len(trimmed_time) != num_time_bins:
            # If it doesn't match, recalculate the range to ensure correct length
            start_index = (total_time_points - num_time_bins) // 2
            end_index = start_index + num_time_bins
            trimmed_time = time[start_index:end_index]
            logging.info(f"Adjusted trimmed time to ensure correct size: {len(trimmed_time)}")

        # Final check
        if len(trimmed_time) != num_time_bins:
            raise ValueError("Trimming resulted in a time array that does not match the STFT result size.")

        # Log the final length of the trimmed time array
        logging.info(f"Final trimmed time array length: {len(trimmed_time)}")

        return trimmed_time

    ###################################################################################
   
    def prepare_stft_analysis_data_single(self, mode):
        logging.debug("Initializing data object for STFT analysis.")

        data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.v1,
            v2=self.v2,
            h1=self.h1,
            h2=self.h2,
            visualize=self.visualize,
            alpha=self.alpha,
            shift_signal=True
        )
        
        logging.debug(f"Data object initialized with mode: {data_obj.mode}")
        
        self.stft_window_length = self.calculate_signal_duration(self.nperseg, data_obj.sampling_rate_MHz * 1e6)

        if mode == "1d":
            logging.debug("Starting 1D STFT analysis.")
            stft_freqs, stft_times, stft_analysis_coefficients = self.stft_1d(
                data_obj.trimmed_signal_1d,
                data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
            )
            logging.debug("1D STFT analysis completed.")

        elif mode == "2d":
            logging.debug("Starting 2D STFT analysis.")
            stft_freqs, stft_times, stft_analysis_coefficients = self.stft_2d(
                data_obj.trimmed_signal_2d,
                data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
            )
            logging.debug("2D STFT analysis completed.")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[data_obj.sample_name] = {
            "frequencies": stft_freqs,
            "signal_energy": np.sum(np.abs(stft_analysis_coefficients) ** 2, axis=1)
        }
        logging.debug(f"Results stored for sample {data_obj.sample_name}")

        logging.info("Completed processing all folders.")

    ###################################################################################
        
    def prepare_stft_analysis_data_multiple(self, mode):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}.")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                visualize=False,
                shift_signal=True
            )
                        
            if data_obj.sampling_rate_MHz:
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, data_obj.sampling_rate_MHz * 1e6)

                if mode == "1d":
                    
                    stft_freqs, stft_times, stft_analysis_coefficients = self.stft_1d(
                        data_obj.trimmed_signal_1d,
                        data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                    )

                elif mode == "2d":
                    
                    stft_freqs, stft_times, stft_analysis_coefficients = self.stft_2d(
                        data_obj.trimmed_signal_2d,
                        data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                    )

                # Store the result in the dictionary, using the folder path as the key
                self.stft_results[data_obj.sample_name] = {
                    "frequencies": stft_freqs,
                    "signal_energy": np.sum(np.abs(stft_analysis_coefficients) ** 2, axis=1),
                    "starting_depth": data_obj.trimmed_depth_1d[0]
                }

            logging.info("Completed processing all folders.")

    ###################################################################################
    
    def prepare_stft_analysis_data_CF(self):
        logging.debug("Initializing data object for STFT analysis.")

        data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.v1,
            v2=self.v2,
            h1=self.h1,
            h2=self.h2,
            visualize=self.visualize,
            alpha=self.alpha
        )
        
        self.data_obj = data_obj
        
        logging.debug(f"Data object initialized with mode: {data_obj.mode}")
        
        self.stft_window_length = self.calculate_signal_duration(self.nperseg, data_obj.sampling_rate_MHz * 1e6)

        logging.debug("Starting 2D STFT analysis.")
        all_stft_freqs, all_central_freq, all_stft_times, all_stft_analysis_coefficients = self.stft_2d_CF(
            data_obj.trimmed_signal_2d,
            data_obj.sampling_rate_MHz * 1e6,
            window_length=self.stft_window_length,
            normalize_signal=False
        )
        logging.debug("2D STFT analysis completed.")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[data_obj.sample_name] = {
            "frequencies": all_stft_freqs,
            "central_freq": all_central_freq,
            "times": all_stft_times,
            "stft_signal": all_stft_analysis_coefficients,
            "original_signal": data_obj.trimmed_signal_2d
        }
        logging.debug(f"Results stored for sample {data_obj.sample_name}")

        logging.info("Completed processing all folders.")
        
    ###################################################################################
    
    def plot_stft_results(self):
        if not self.stft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.stft_results.items():
            
            signal_energy = results['signal_energy']

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Amplitude length: {len(signal_energy)}")

            plt.plot(freqs, signal_energy, label=sample_name)

        plt.title('STFT Amplitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel('Signal Energy')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    ###################################################################################
    @classmethod        
    def post_statistic_analysis_stft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.stft_results
        results_obj_2 = obj_2.stft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")

    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################
    
    def calculate_signal_duration(self, data_points, sampling_frequency):
        
        return data_points / sampling_frequency
    
    ###################################################################################
    @staticmethod
    def rotate_flip(
                    two_dimension_array: np.ndarray) -> np.ndarray:

        # Rotate the input array counterclockwise by 90 degrees
        rotated_array = np.rot90(two_dimension_array)
        
        # Flip the rotated array horizontally
        rotated_flipped_array = np.flipud(rotated_array)
        
        return rotated_flipped_array

    ###################################################################################





class BSC_STFT_with_normaliazation():

    def __init__(self,
                 folder_path,
                 ROIs_path,
                 device,
                 ac_method,
                 alpha,
                 ROI_size,
                 mode,
                 visualize,
                 stft_nperseg,
                 stft_noverlap) -> None:
        
        # Initialize instance variables
        self.folder_path = folder_path
        self.ROIs_path = ROIs_path
        self.device = device
        self.size = "large"
        self.signal_type = "no_tgc"
        self.ac_method = ac_method
        self.mode = mode
        self.alpha = alpha
        self.ROI_size = ROI_size
        self.visualize = visualize
        self.nperseg = stft_nperseg
        self.noverlap = stft_noverlap
        
        self.stft_window_length = None        
        self.main_roi_v1 = None
        self.main_roi_v2 = None
        self.main_roi_h1 = None
        self.main_roi_h2 = None
        self.normalization_roi_v1 = None
        self.normalization_roi_v2 = None
        self.normalization_roi_h1 = None
        self.normalization_roi_h2 = None    
        
        self.excel_all_data = None
        self.large_roi_data = None
        self.small_roi_data = None
                
        self.stft_results = {}

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################
    
    def __run_single(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_single(mode="2d")
        self.plot_stft_results()       
                                
    ###################################################################################
   
    def __run_multiple(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_multiple(mode="2d")
        self.plot_stft_results()       
                                
    ###################################################################################

    def calculate_signal_duration(self, data_points, sampling_frequency):
        
        return data_points / sampling_frequency

    ###################################################################################
    
    def set_roi_df(self):
        # Dictionaries to hold data for each ROI size
        large_data_list = []
        small_data_list = []
        
        # Iterate through each file in the directory specified by self.ROIs_path
        for filename in os.listdir(self.ROIs_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Check for Excel file extensions
                # Construct full file path
                file_path = os.path.join(self.ROIs_path, filename)
                # Load data from specific tabs
                try:
                    large_data = pd.read_excel(file_path, sheet_name="Large_ROI")
                    small_data = pd.read_excel(file_path, sheet_name="Small_ROI")
                    large_data_list.append(large_data)
                    small_data_list.append(small_data)
                except ValueError as e:
                    logging.error(f"Error loading tabs from {file_path}: {e}")
                    raise ValueError(f"Missing expected tabs in {file_path}: {e}")

        # Combine data for each ROI size
        self.large_roi_data = pd.concat(large_data_list, ignore_index=True) if large_data_list else pd.DataFrame()
        self.small_roi_data = pd.concat(small_data_list, ignore_index=True) if small_data_list else pd.DataFrame()
        
    ###################################################################################

    def compute_stft(self, signal_1d, fs, window, nperseg, noverlap):
        logging.debug("Entering compute_stft method.")
        logging.debug(f"Input parameters - fs: {fs}, nperseg: {nperseg}, noverlap: {noverlap}")

        step = nperseg - noverlap
        freqs = np.fft.fftfreq(nperseg, 1/fs)
        time_slices = np.arange(0, len(signal_1d) - nperseg, step)
        stft_matrix = np.zeros((len(freqs), len(time_slices)), dtype=complex)

        logging.debug(f"Step size for slicing: {step}")
        logging.debug(f"Number of time slices: {len(time_slices)}")

        for i, t in enumerate(time_slices):
            logging.debug(f"Processing time slice index: {i}, time: {t/fs} seconds")
            x_segment = signal_1d[t:t+nperseg] * window
            stft_matrix[:, i] = np.fft.fft(x_segment)

        freqs = np.fft.fftshift(freqs)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
        
        logging.debug("STFT computation completed, shifting frequencies and STFT matrix.")
        logging.debug(f"Output frequencies shape: {freqs.shape}, time slices shape: {time_slices.shape}, STFT matrix shape: {stft_matrix.shape}")

        logging.debug("Exiting compute_stft method.")
        return freqs, time_slices / fs, stft_matrix

    ###################################################################################
    
    def gaussian_window(self, window_length_0d, fs_0d):
        logging.debug("Entering gaussian_window method.")
        logging.debug(f"Input parameters - window_length_0d: {window_length_0d}, fs_0d: {fs_0d}")
        
        # Convert the number of points to an integer
        num_points = int(window_length_0d * fs_0d)
        logging.debug(f"Number of points calculated: {num_points}")

        time_1d = np.linspace(0, window_length_0d, num_points)  # Generating appropriate tau range
        logging.debug(f"Time vector generated with length: {len(time_1d)}")

        sigma = window_length_0d / 6  # Standard deviation
        logging.debug(f"Standard deviation (sigma) calculated: {sigma}")
        
        t_0 = window_length_0d / 2  # Central point (mean)
        logging.debug(f"Central point (t_0) calculated: {t_0}")

        amp_1d = np.exp(-((time_1d - t_0)**2) / (2 * sigma**2))
        normalization = np.sqrt(np.sqrt(np.pi)) 
        amp_1d /= normalization
        logging.debug("Amplitude vector calculated and normalized.")

        logging.debug("Exiting gaussian_window method.")
        return time_1d, amp_1d

    ###################################################################################

    def stft_1d(self, signal_1d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        # Calculate signal energy and normalize if requested
        signal_energy = np.sum(np.abs(signal_1d) ** 2)
        logging.info(f"Signal energy calculated: {signal_energy}")

        if normalize_signal and signal_energy > 0:
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        #print("nperseg =", nperseg)
        
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, stft_result

    ###################################################################################

    def stft_2d(self, signal_2d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_2d method.")
        
        cumulative_amplitude = None
        total_iterations = signal_2d.shape[0]  # Total iterations now equal to the number of rows
        logging.debug(f"Total iterations set to: {total_iterations}")

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")

            signal_1d = signal_2d[i, :]
            freqs, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal=normalize_signal)

            # Initialize cumulative arrays on the first iteration
            if cumulative_amplitude is None:
                cumulative_amplitude = np.zeros_like(amplitude)
                logging.info("Initialized cumulative frequencies and amplitudes.")

            # Add the current frequencies and amplitudes to the cumulative sum
            cumulative_amplitude += amplitude
            logging.debug(f"Updated cumulative frequencies and amplitudes for index {i}.")

        # Calculate the average
        avg_amplitude = cumulative_amplitude / total_iterations
        logging.info("Calculated averaged frequencies and amplitudes.")

        logging.debug("Exiting stft_2d method.")
        return freqs, avg_amplitude

    ###################################################################################
    
    def plot_stft(self, time, freqs, stft_result, log=False):
        # Log the start of the plotting process
        logging.info("Starting to plot STFT.")

        # Convert frequencies from Hz to MHz
        freqs_mhz = freqs / 1e6
        logging.debug(f"Frequencies converted to MHz: {freqs_mhz}")

        # Convert time from seconds to microseconds
        time_us = time * 1e6
        logging.debug(f"Time converted to µs: {time_us}")

        # Plot STFT with both positive and negative frequencies
        plt.figure(figsize=(12, 6))

        # Check for logarithm and avoid log(0) by adding a small constant
        if log:
            stft_result = np.log(np.abs(stft_result) + 1e-10)  # Avoid log(0)
            logging.info("Applied logarithmic scaling to STFT result.")

        # Create the plot using imshow
        plt.imshow(np.abs(stft_result), aspect='auto',
                extent=[time_us.min(), time_us.max(), freqs_mhz.min(), freqs_mhz.max()], origin='lower', cmap='jet')
        
        plt.title('STFT with Gaussian Window')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [µs]')
        plt.colorbar(label='Magnitude')

        # Show the plot
        plt.show()
        logging.info("STFT plot displayed successfully.")

    ###################################################################################
    
    def calculate_stft_time(self, time, stft_result):
        # Ensure stft_result has at least one dimension
        if stft_result.ndim < 2:
            raise ValueError("stft_result must have at least two dimensions.")

        # Determine the number of time bins in the STFT result
        num_time_bins = stft_result.shape[1]

        # Calculate the total number of points in the time array
        total_time_points = len(time)

        # Log the original time array length and STFT result size
        logging.info(f"Original time array length: {total_time_points}")
        logging.info(f"Number of time bins in STFT result: {num_time_bins}")

        # Check if we can trim the time array to the required size
        if total_time_points < num_time_bins:
            raise ValueError("The time array is smaller than the STFT result time bins.")

        # Calculate how many points to remove from each side to achieve the target size
        total_points_to_remove = total_time_points - num_time_bins

        # Ensure we remove points from both sides equally
        if total_points_to_remove % 2 == 0:
            points_to_remove_each_side = total_points_to_remove // 2
        else:
            points_to_remove_each_side = total_points_to_remove // 2

        # Trim the time array from both ends
        trimmed_time = time[points_to_remove_each_side:total_time_points - points_to_remove_each_side]
        
        # Log the trimming process
        logging.info(f"Trimming time array: removing {points_to_remove_each_side} points from each side.")

        # Ensure trimmed_time has the correct size
        if len(trimmed_time) != num_time_bins:
            # If it doesn't match, recalculate the range to ensure correct length
            start_index = (total_time_points - num_time_bins) // 2
            end_index = start_index + num_time_bins
            trimmed_time = time[start_index:end_index]
            logging.info(f"Adjusted trimmed time to ensure correct size: {len(trimmed_time)}")

        # Final check
        if len(trimmed_time) != num_time_bins:
            raise ValueError("Trimming resulted in a time array that does not match the STFT result size.")

        # Log the final length of the trimmed time array
        logging.info(f"Final trimmed time array length: {len(trimmed_time)}")

        return trimmed_time

    ###################################################################################
   
    def prepare_stft_analysis_data_single(self, mode):
        logging.debug("Initializing data object for STFT analysis.")

        sample_name = self.get_sample_name_from_path(self.folder_path)
        self.set_roi_data_from_df(sample_name)
        
        # Log the mode of operation and sample name being processed
        logging.debug(f"Preparing STFT analysis data for mode: {mode}, sample: {sample_name}")

        main_roi_data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.main_roi_v1,
            v2=self.main_roi_v2,
            h1=self.main_roi_h1,
            h2=self.main_roi_h2,
            visualize=self.visualize,
            alpha=self.alpha,
            shift_signal=True

        )
        
        normalization_roi_data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.normalization_roi_v1,
            v2=self.normalization_roi_v2,
            h1=self.normalization_roi_h1,
            h2=self.normalization_roi_h2,
            visualize=self.visualize,
            alpha=self.alpha
        )
        
        # Log the creation of data objects for both main and normalization ROIs
        logging.debug(f"Data objects created for main and normalization ROIs of sample: {sample_name}")

        if mode == "1d":
            
            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
                )
            
            # Log the completion of STFT analysis for 1D mode
            logging.debug(f"STFT 1D analysis completed for sample: {sample_name}")

        elif mode == "2d":

            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=True
                )
            
            # Log the completion of STFT analysis for 2D mode
            logging.debug(f"STFT 2D analysis completed for sample: {sample_name}")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[main_roi_data_obj.sample_name] = {
            "frequencies": stft_freqs_main_roi,
            "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                            np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
        }
        logging.debug(f"Results stored for sample {main_roi_data_obj.sample_name}")

        logging.info("Completed processing for the provided mode.")

    ###################################################################################
        
    def prepare_stft_analysis_data_multiple(self, mode):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
        
            main_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
            
            normalization_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.normalization_roi_v1,
                v2=self.normalization_roi_v2,
                h1=self.normalization_roi_h1,
                h2=self.normalization_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
        
            if main_roi_data_obj.sampling_rate_MHz:
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, main_roi_data_obj.sampling_rate_MHz * 1e6)

                if mode == "1d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                        main_roi_data_obj.trimmed_signal_1d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                        normalization_roi_data_obj.trimmed_signal_1d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                        np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }

                elif mode == "2d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                        main_roi_data_obj.trimmed_signal_2d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                        normalization_roi_data_obj.trimmed_signal_2d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=True
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }
                    
            logging.info("Completed processing all folders.")

    ###################################################################################
        
    def plot_stft_results(self):
        if not self.stft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.stft_results.items():
            
            signal_energy = results['signal_energy']

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Amplitude length: {len(signal_energy)}")

            plt.plot(freqs, signal_energy, label=sample_name)

        plt.title('STFT Amplitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel('Signal Energy')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    ###################################################################################
    @classmethod        
    def post_statistic_analysis_stft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.stft_results
        results_obj_2 = obj_2.stft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")

    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################

    def get_sample_name_from_path(self, folder_path):
        # Assuming the sample name is the last part of the folder path
        sample_name = folder_path.split(os.sep)[-1]
        logging.debug(f"Extracted sample name: {sample_name} from path: {self.folder_path}")
        return sample_name
            
    ###################################################################################
    
    def set_roi_data_from_df(self, sample_name):
        logging.debug(f"Setting ROI data for sample: {sample_name} with ROI size: {self.ROI_size}")
        
        # Validate `ROI_size` argument
        if self.ROI_size not in ["Small_ROI", "Large_ROI"]:
            logging.error(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
            raise ValueError(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
        
        # Select the appropriate data based on ROI size
        data = self.large_roi_data if self.ROI_size == "Large_ROI" else self.small_roi_data
        
        # Ensure the required columns exist
        required_columns = ['sample_name', 'v1', 'v2', 'h1', 'h2', 'v1_new', 'v2_new', 'h1_new', 'h2_new']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns} in the {self.ROI_size} data")
            raise ValueError(f"Missing required columns in the {self.ROI_size} data: {missing_columns}")

        # Filter the DataFrame for rows where 'sample_name' matches
        sample_data = data[data['sample_name'] == sample_name]
        
        if not sample_data.empty:
            # Extract the first matching row
            row = sample_data.iloc[0]
            
            # Set attributes from the DataFrame row
            self.main_roi_v1 = row['v1']
            self.main_roi_v2 = row['v2']
            self.main_roi_h1 = row['h1']
            self.main_roi_h2 = row['h2']
            self.normalization_roi_v1 = row['v1_new']
            self.normalization_roi_v2 = row['v2_new']
            self.normalization_roi_h1 = row['h1_new']
            self.normalization_roi_h2 = row['h2_new']
            
            logging.info(f"ROI data successfully set for sample: {sample_name} using {self.ROI_size}")
        else:
            logging.error(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")
            raise ValueError(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")
        
    ###################################################################################
    
    
    
    
    
    
    
class BSC_FT_with_normaliazation():

    def __init__(self,
                 folder_path,
                 ROIs_path,
                 device,
                 ac_method,
                 alpha,
                 ROI_size,
                 visualize,
                 ) -> None:
        
        # Initialize instance variables
        self.folder_path = folder_path
        self.ROIs_path = ROIs_path
        self.device = device
        self.size = "large"
        self.signal_type = "no_tgc"
        self.ac_method = ac_method
        self.alpha = alpha
        self.ROI_size = ROI_size
        self.visualize = visualize

        self.main_roi_v1 = None
        self.main_roi_v2 = None
        self.main_roi_h1 = None
        self.main_roi_h2 = None
        self.normalization_roi_v1 = None
        self.normalization_roi_v2 = None
        self.normalization_roi_h1 = None
        self.normalization_roi_h2 = None    
        
        self.excel_all_data = None
        self.large_roi_data = None
        self.small_roi_data = None
                
        self.ft_results = {}

        # initialize
        self.__run()

    ###################################################################################
                                   
    def __run(self):
        self.set_roi_df()    
        self.prepare_ft_analysis_data_multiple()
        self.plot_ft_results('ft_magnitude_main_roi')      
        self.plot_ft_results('ft_magnitude_normalization_roi')       
        self.plot_ft_results('ft_magnitude_ratio')       
                                
    ###################################################################################
    
    def set_roi_df(self):
        # Dictionaries to hold data for each ROI size
        large_data_list = []
        small_data_list = []
        
        # Iterate through each file in the directory specified by self.ROIs_path
        for filename in os.listdir(self.ROIs_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Check for Excel file extensions
                # Construct full file path
                file_path = os.path.join(self.ROIs_path, filename)
                # Load data from specific tabs
                try:
                    large_data = pd.read_excel(file_path, sheet_name="Large_ROI")
                    small_data = pd.read_excel(file_path, sheet_name="Small_ROI")
                    large_data_list.append(large_data)
                    small_data_list.append(small_data)
                except ValueError as e:
                    logging.error(f"Error loading tabs from {file_path}: {e}")
                    raise ValueError(f"Missing expected tabs in {file_path}: {e}")

        # Combine data for each ROI size
        self.large_roi_data = pd.concat(large_data_list, ignore_index=True) if large_data_list else pd.DataFrame()
        self.small_roi_data = pd.concat(small_data_list, ignore_index=True) if small_data_list else pd.DataFrame()
        
    ###################################################################################
           
    def prepare_ft_analysis_data_multiple(self):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
        
            main_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
            
            normalization_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.normalization_roi_v1,
                v2=self.normalization_roi_v2,
                h1=self.normalization_roi_h1,
                h2=self.normalization_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
                )
        
            if main_roi_data_obj.sampling_rate_MHz:
                                   
                ft_freqs_main_roi, ft_magnitude_main_roi = self.ft_2d(
                    main_roi_data_obj.trimmed_signal_2d,
                    main_roi_data_obj.sampling_rate_MHz * 1e6,
                    normalize_signal=False
                    )
                
                _, ft_magnitude_normalization_roi = self.ft_2d(
                    normalization_roi_data_obj.trimmed_signal_2d,
                    normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                    normalize_signal=False
                    )
                    
                # Store the result in the dictionary, using the folder path as the key
                self.ft_results[main_roi_data_obj.sample_name] = {
                    "frequencies":                    ft_freqs_main_roi,
                    "ft_magnitude_main_roi":          np.abs(ft_magnitude_main_roi), 
                    "ft_magnitude_normalization_roi": np.abs(ft_magnitude_normalization_roi),
                    "ft_magnitude_ratio":         np.abs(ft_magnitude_main_roi) / np.abs(ft_magnitude_normalization_roi)
                }

            logging.info("Completed processing all folders.")
            
    ###################################################################################
    
    def get_sample_name_from_path(self, folder_path):
        # Assuming the sample name is the last part of the folder path
        sample_name = folder_path.split(os.sep)[-1]
        logging.debug(f"Extracted sample name: {sample_name} from path: {self.folder_path}")
        return sample_name
    
    ###################################################################################
    
    def set_roi_data_from_df(self, sample_name):
        logging.debug(f"Setting ROI data for sample: {sample_name} with ROI size: {self.ROI_size}")
        
        # Validate `ROI_size` argument
        if self.ROI_size not in ["Small_ROI", "Large_ROI"]:
            logging.error(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
            raise ValueError(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
        
        # Select the appropriate data based on ROI size
        data = self.large_roi_data if self.ROI_size == "Large_ROI" else self.small_roi_data
        
        # Ensure the required columns exist
        required_columns = ['sample_name', 'v1', 'v2', 'h1', 'h2', 'v1_new', 'v2_new', 'h1_new', 'h2_new']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns} in the {self.ROI_size} data")
            raise ValueError(f"Missing required columns in the {self.ROI_size} data: {missing_columns}")

        # Filter the DataFrame for rows where 'sample_name' matches
        sample_data = data[data['sample_name'] == sample_name]
        
        if not sample_data.empty:
            # Extract the first matching row
            row = sample_data.iloc[0]
            
            # Set attributes from the DataFrame row
            self.main_roi_v1 = row['v1']
            self.main_roi_v2 = row['v2']
            self.main_roi_h1 = row['h1']
            self.main_roi_h2 = row['h2']
            self.normalization_roi_v1 = row['v1_new']
            self.normalization_roi_v2 = row['v2_new']
            self.normalization_roi_h1 = row['h1_new']
            self.normalization_roi_h2 = row['h2_new']
            
            logging.info(f"ROI data successfully set for sample: {sample_name} using {self.ROI_size}")
        else:
            logging.error(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")
            raise ValueError(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")

    ###################################################################################

    def ft_2d(self, signal_2d, fs, normalize_signal):
        logging.debug("Entering stft_2d method.")
        
        cumulative_amplitude = None
        total_iterations = signal_2d.shape[0]  # Total iterations now equal to the number of rows
        logging.debug(f"Total iterations set to: {total_iterations}")

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")

            signal_1d = signal_2d[i, :]
            freqs, amplitude = self.ft_1d(signal_1d, fs, normalize_signal=normalize_signal)

            # Initialize cumulative arrays on the first iteration
            if cumulative_amplitude is None:
                cumulative_amplitude = np.zeros_like(amplitude)
                logging.info("Initialized cumulative frequencies and amplitudes.")

            # Add the current frequencies and amplitudes to the cumulative sum
            cumulative_amplitude += amplitude
            logging.debug(f"Updated cumulative frequencies and amplitudes for index {i}.")

        # Calculate the average
        avg_amplitude = cumulative_amplitude / total_iterations
        logging.info("Calculated averaged frequencies and amplitudes.")

        logging.debug("Exiting stft_2d method.")
        return freqs, avg_amplitude

    ###################################################################################
    
    def ft_1d(self, signal_1d, fs, normalize_signal):
        logging.debug("Entering stft_1d method.")
        
        if normalize_signal:
            # Calculate signal energy and normalize if requested
            signal_energy = np.sum(np.abs(signal_1d) ** 2)
            logging.info(f"Signal energy calculated: {signal_energy}")
        
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
       
        # Compute FT
        ft_freqs, ft_magnitude = self.compute_ft(signal_1d, fs)
        logging.info("STFT computation completed.")

        logging.debug("Exiting stft_1d method.")
        return ft_freqs, ft_magnitude
    
    ###################################################################################

    def compute_ft(self, signal_1d, fs, normalize=True):
                
        n = len(signal_1d)
        ft_spectrum = np.fft.fft(signal_1d)
        ft_magnitude = np.abs(ft_spectrum)
        
        if normalize:
            ft_magnitude /= n  # Normalize by the number of samples
        
        ft_freqs = np.fft.fftfreq(n, d=1/fs)
        half_n = n // 2
        return ft_freqs[:half_n], ft_magnitude[:half_n]
    
    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################
        
    def plot_ft_results(self, key):
        if not self.ft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.ft_results.items():
            
            signal_magnitude = results[key]

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Magnitude length: {len(signal_magnitude)}")

            plt.plot(freqs, signal_magnitude, label=sample_name)

        plt.title(f'FT Magnitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel(f'{key}')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()
        
    ###################################################################################
    @classmethod        
    def post_statistic_analysis_ft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.ft_results
        results_obj_2 = obj_2.ft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")
                
    ###################################################################################
    
    
    
    
    
    
    
    
    
class BSC_STFT_reference_vs_phantom():

    def __init__(self,
                 folder_path: str,
                 ROIs_path: str,
                 phantom_path: str,
                 **kwargs) -> None:
        
        # Initialize instance variables
        self.folder_path: str = folder_path
        self.ROIs_path: str = ROIs_path
        self.phantom_path: str = phantom_path
               
        self.device: str | None = kwargs.get("device", None)
        self.ac_method: str | None = kwargs.get("ac_method", None)
        self.alpha: float | None = kwargs.get("alpha", None)
        self.ROI_size: int | None = kwargs.get("ROI_size", None)
        self.mode: str | None = kwargs.get("mode", None)
        self.normalization: bool | None = kwargs.get("normalization", None)
        self.visualize: bool | None = kwargs.get("visualize", None)
        self.nperseg: int | None = kwargs.get("stft_nperseg", None)
        self.noverlap: int | None = kwargs.get("stft_noverlap", None)

        # Additional fixed attributes
        self.size: str = "large"
        self.signal_type: str = "no_tgc"
        
        self.stft_window_length = None        
        self.main_roi_v1 = None
        self.main_roi_v2 = None
        self.main_roi_h1 = None
        self.main_roi_h2 = None
        self.normalization_roi_v1 = None
        self.normalization_roi_v2 = None
        self.normalization_roi_h1 = None
        self.normalization_roi_h2 = None    
        
        self.excel_all_data = None
        self.large_roi_data = None
        self.small_roi_data = None
                
        self.stft_results = {}

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################
    
    def __run_single(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_single(mode="2d")
        self.plot_stft_results()       
                                
    ###################################################################################
   
    def __run_multiple(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_multiple(mode="2d")
        self.plot_stft_results()       
                                
    ###################################################################################
    
    def prepare_stft_analysis_data_CF(self):
        logging.debug("Initializing data object for STFT analysis.")

        data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.v1,
            v2=self.v2,
            h1=self.h1,
            h2=self.h2,
            visualize=self.visualize,
            alpha=self.alpha
        )
        
        self.data_obj = data_obj
        
        logging.debug(f"Data object initialized with mode: {data_obj.mode}")
        
        self.stft_window_length = self.calculate_signal_duration(self.nperseg, data_obj.sampling_rate_MHz * 1e6)

        logging.debug("Starting 2D STFT analysis.")
        all_stft_freqs, all_central_freq, all_stft_times, all_stft_analysis_coefficients = self.stft_2d_CF(
            data_obj.trimmed_signal_2d,
            data_obj.sampling_rate_MHz * 1e6,
            window_length=self.stft_window_length,
            normalize_signal=False
        )
        logging.debug("2D STFT analysis completed.")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[data_obj.sample_name] = {
            "frequencies": all_stft_freqs,
            "central_freq": all_central_freq,
            "times": all_stft_times,
            "stft_signal": all_stft_analysis_coefficients,
            "original_signal": data_obj.trimmed_signal_2d
        }
        logging.debug(f"Results stored for sample {data_obj.sample_name}")

        logging.info("Completed processing all folders.")
        
    ###################################################################################

    def calculate_signal_duration(self, data_points, sampling_frequency):
        
        return data_points / sampling_frequency

    ###################################################################################
    
    def set_roi_df(self):
        # Dictionaries to hold data for each ROI size
        large_data_list = []
        small_data_list = []
        
        # Iterate through each file in the directory specified by self.ROIs_path
        for filename in os.listdir(self.ROIs_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Check for Excel file extensions
                # Construct full file path
                file_path = os.path.join(self.ROIs_path, filename)
                # Load data from specific tabs
                try:
                    large_data = pd.read_excel(file_path, sheet_name="Large_ROI")
                    small_data = pd.read_excel(file_path, sheet_name="Small_ROI")
                    large_data_list.append(large_data)
                    small_data_list.append(small_data)
                except ValueError as e:
                    logging.error(f"Error loading tabs from {file_path}: {e}")
                    raise ValueError(f"Missing expected tabs in {file_path}: {e}")

        # Combine data for each ROI size
        self.large_roi_data = pd.concat(large_data_list, ignore_index=True) if large_data_list else pd.DataFrame()
        self.small_roi_data = pd.concat(small_data_list, ignore_index=True) if small_data_list else pd.DataFrame()
        
    ###################################################################################

    def compute_stft(self, signal_1d, fs, window, nperseg, noverlap):
        logging.debug("Entering compute_stft method.")
        logging.debug(f"Input parameters - fs: {fs}, nperseg: {nperseg}, noverlap: {noverlap}")

        step = nperseg - noverlap
        freqs = np.fft.fftfreq(nperseg, 1/fs)
        time_slices = np.arange(0, len(signal_1d) - nperseg, step)
        stft_matrix = np.zeros((len(freqs), len(time_slices)), dtype=complex)

        logging.debug(f"Step size for slicing: {step}")
        logging.debug(f"Number of time slices: {len(time_slices)}")

        for i, t in enumerate(time_slices):
            logging.debug(f"Processing time slice index: {i}, time: {t/fs} seconds")
            x_segment = signal_1d[t:t+nperseg] * window
            stft_matrix[:, i] = np.fft.fft(x_segment)

        freqs = np.fft.fftshift(freqs)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
        
        logging.debug("STFT computation completed, shifting frequencies and STFT matrix.")
        logging.debug(f"Output frequencies shape: {freqs.shape}, time slices shape: {time_slices.shape}, STFT matrix shape: {stft_matrix.shape}")

        logging.debug("Exiting compute_stft method.")
        return freqs, time_slices / fs, stft_matrix

    ###################################################################################
    
    def gaussian_window(self, window_length_0d, fs_0d):
        logging.debug("Entering gaussian_window method.")
        logging.debug(f"Input parameters - window_length_0d: {window_length_0d}, fs_0d: {fs_0d}")
        
        # Convert the number of points to an integer
        num_points = int(window_length_0d * fs_0d)
        logging.debug(f"Number of points calculated: {num_points}")

        time_1d = np.linspace(0, window_length_0d, num_points)  # Generating appropriate tau range
        logging.debug(f"Time vector generated with length: {len(time_1d)}")

        sigma = window_length_0d / 6  # Standard deviation
        logging.debug(f"Standard deviation (sigma) calculated: {sigma}")
        
        t_0 = window_length_0d / 2  # Central point (mean)
        logging.debug(f"Central point (t_0) calculated: {t_0}")

        amp_1d = np.exp(-((time_1d - t_0)**2) / (2 * sigma**2))
        normalization = np.sqrt(np.sqrt(np.pi)) 
        amp_1d /= normalization
        logging.debug("Amplitude vector calculated and normalized.")

        logging.debug("Exiting gaussian_window method.")
        return time_1d, amp_1d

    ###################################################################################

    def stft_1d(self, signal_1d, fs, window_length, normalize_signal):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        # Calculate signal energy and normalize if requested
        signal_energy = np.sum(np.abs(signal_1d) ** 2)
        logging.info(f"Signal energy calculated: {signal_energy}")

        if normalize_signal and signal_energy > 0:
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        #print("nperseg =", nperseg)
        
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, stft_result

    ###################################################################################

    def stft_2d(self, signal_2d, fs, window_length, normalize_signal, aggregation, mode="per_signal"):
        logging.debug("Entering stft_2d method.")

        if mode == "average_first":
            logging.debug("Averaging signals before STFT.")
            avg_signal = np.mean(signal_2d, axis=0)
            freqs, avg_amplitude = self.stft_1d(avg_signal, fs, window_length, normalize_signal)
            logging.info("Computed STFT on averaged signal.")
        elif mode == "per_signal":
            amplitude_list = []
            total_iterations = signal_2d.shape[0]
            logging.debug(f"Total iterations set to: {total_iterations}")

            for i in range(total_iterations):
                logging.debug(f"Processing signal index: {i}")
                signal_1d = signal_2d[i, :]
                freqs, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal)
                amplitude_list.append(amplitude)
                logging.debug(f"Appended amplitude for index {i}.")

            amplitude_stack = np.stack(amplitude_list, axis=0)

            if aggregation == "mean":
                avg_amplitude = np.mean(amplitude_stack, axis=0)
                logging.info("Calculated mean of amplitudes.")
            elif aggregation == "median":
                avg_amplitude = np.median(amplitude_stack, axis=0)
                logging.info("Calculated median of amplitudes.")
            elif aggregation == "without_aggregation":
                avg_amplitude = amplitude_stack
                logging.info("Returning amplitude stack without aggregation.")
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logging.debug("Exiting stft_2d method.")
        return freqs, avg_amplitude

    ###################################################################################
    
    def plot_stft(self, time, freqs, stft_result, log=False):
        # Log the start of the plotting process
        logging.info("Starting to plot STFT.")

        # Convert frequencies from Hz to MHz
        freqs_mhz = freqs / 1e6
        logging.debug(f"Frequencies converted to MHz: {freqs_mhz}")

        # Convert time from seconds to microseconds
        time_us = time * 1e6
        logging.debug(f"Time converted to µs: {time_us}")

        # Plot STFT with both positive and negative frequencies
        plt.figure(figsize=(12, 6))

        # Check for logarithm and avoid log(0) by adding a small constant
        if log:
            stft_result = np.log(np.abs(stft_result) + 1e-10)  # Avoid log(0)
            logging.info("Applied logarithmic scaling to STFT result.")

        # Create the plot using imshow
        plt.imshow(np.abs(stft_result), aspect='auto',
                extent=[time_us.min(), time_us.max(), freqs_mhz.min(), freqs_mhz.max()], origin='lower', cmap='jet')
        
        plt.title('STFT with Gaussian Window')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [µs]')
        plt.colorbar(label='Magnitude')

        # Show the plot
        plt.show()
        logging.info("STFT plot displayed successfully.")

    ###################################################################################
    
    def calculate_stft_time(self, time, stft_result):
        # Ensure stft_result has at least one dimension
        if stft_result.ndim < 2:
            raise ValueError("stft_result must have at least two dimensions.")

        # Determine the number of time bins in the STFT result
        num_time_bins = stft_result.shape[1]

        # Calculate the total number of points in the time array
        total_time_points = len(time)

        # Log the original time array length and STFT result size
        logging.info(f"Original time array length: {total_time_points}")
        logging.info(f"Number of time bins in STFT result: {num_time_bins}")

        # Check if we can trim the time array to the required size
        if total_time_points < num_time_bins:
            raise ValueError("The time array is smaller than the STFT result time bins.")

        # Calculate how many points to remove from each side to achieve the target size
        total_points_to_remove = total_time_points - num_time_bins

        # Ensure we remove points from both sides equally
        if total_points_to_remove % 2 == 0:
            points_to_remove_each_side = total_points_to_remove // 2
        else:
            points_to_remove_each_side = total_points_to_remove // 2

        # Trim the time array from both ends
        trimmed_time = time[points_to_remove_each_side:total_time_points - points_to_remove_each_side]
        
        # Log the trimming process
        logging.info(f"Trimming time array: removing {points_to_remove_each_side} points from each side.")

        # Ensure trimmed_time has the correct size
        if len(trimmed_time) != num_time_bins:
            # If it doesn't match, recalculate the range to ensure correct length
            start_index = (total_time_points - num_time_bins) // 2
            end_index = start_index + num_time_bins
            trimmed_time = time[start_index:end_index]
            logging.info(f"Adjusted trimmed time to ensure correct size: {len(trimmed_time)}")

        # Final check
        if len(trimmed_time) != num_time_bins:
            raise ValueError("Trimming resulted in a time array that does not match the STFT result size.")

        # Log the final length of the trimmed time array
        logging.info(f"Final trimmed time array length: {len(trimmed_time)}")

        return trimmed_time

    ###################################################################################
   
    def prepare_stft_analysis_data_single(self, mode):
        logging.debug("Initializing data object for STFT analysis.")

        sample_name = self.get_sample_name_from_path(self.folder_path)
        self.set_roi_data_from_df(sample_name)
        
        # Log the mode of operation and sample name being processed
        logging.debug(f"Preparing STFT analysis data for mode: {mode}, sample: {sample_name}")

        main_roi_data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.main_roi_v1,
            v2=self.main_roi_v2,
            h1=self.main_roi_h1,
            h2=self.main_roi_h2,
            visualize=self.visualize,
            alpha=self.alpha,
            shift_signal=True

        )
        
        if self.normalization == "reference":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.normalization_roi_v1,
                v2=self.normalization_roi_v2,
                h1=self.normalization_roi_h1,
                h2=self.normalization_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
            
        elif self.normalization == "phantom":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.phantom_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
        
        # Log the creation of data objects for both main and normalization ROIs
        logging.debug(f"Data objects created for main and normalization ROIs of sample: {sample_name}")

        if mode == "1d":
            
            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 1D mode
            logging.debug(f"STFT 1D analysis completed for sample: {sample_name}")

        elif mode == "2d":

            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 2D mode
            logging.debug(f"STFT 2D analysis completed for sample: {sample_name}")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[main_roi_data_obj.sample_name] = {
            "frequencies": stft_freqs_main_roi,
            "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                            np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
        }
        logging.debug(f"Results stored for sample {main_roi_data_obj.sample_name}")

        logging.info("Completed processing for the provided mode.")

    ###################################################################################
        
    def prepare_stft_analysis_data_multiple(self, mode):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
        
            main_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
                
            if self.normalization == "reference":
                            
                normalization_roi_data_obj = Data(
                    sample_folder_path=folder_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.normalization_roi_v1,
                    v2=self.normalization_roi_v2,
                    h1=self.normalization_roi_h1,
                    h2=self.normalization_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                    )      
                
            elif self.normalization == "phantom":
            
                normalization_roi_data_obj = Data(
                    sample_folder_path=self.phantom_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.main_roi_v1,
                    v2=self.main_roi_v2,
                    h1=self.main_roi_h1,
                    h2=self.main_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                )
                    
            if main_roi_data_obj.sampling_rate_MHz:
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, main_roi_data_obj.sampling_rate_MHz * 1e6)

                if mode == "1d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                        main_roi_data_obj.trimmed_signal_1d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                        normalization_roi_data_obj.trimmed_signal_1d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }

                elif mode == "2d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                        main_roi_data_obj.trimmed_signal_2d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="mean",
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                        normalization_roi_data_obj.trimmed_signal_2d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="mean",
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }
                    
            else:
                raise ValueError(f"Sampling rate is not defined for sample '{main_roi_data_obj.sample_name}'.")
            
        logging.info("Completed processing all folders.")

    ###################################################################################
        
    def plot_stft_results(self):
        if not self.stft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.stft_results.items():
            
            signal_energy = results['signal_energy']

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Amplitude length: {len(signal_energy)}")

            plt.plot(freqs, signal_energy, label=sample_name)

        plt.title('STFT Amplitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel('Signal Energy')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    ###################################################################################
    @classmethod        
    def post_statistic_analysis_stft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.stft_results
        results_obj_2 = obj_2.stft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")

    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################

    def get_sample_name_from_path(self, folder_path):
        # Assuming the sample name is the last part of the folder path
        sample_name = folder_path.split(os.sep)[-1]
        logging.debug(f"Extracted sample name: {sample_name} from path: {self.folder_path}")
        return sample_name
            
    ###################################################################################
    
    def set_roi_data_from_df(self, sample_name):
        logging.debug(f"Setting ROI data for sample: {sample_name} with ROI size: {self.ROI_size}")
        
        # Validate `ROI_size` argument
        if self.ROI_size not in ["Small_ROI", "Large_ROI"]:
            logging.error(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
            raise ValueError(f"Invalid ROI_size argument: {self.ROI_size}. Must be 'Small_ROI' or 'Large_ROI'.")
        
        # Select the appropriate data based on ROI size
        data = self.large_roi_data if self.ROI_size == "Large_ROI" else self.small_roi_data
        
        # Ensure the required columns exist
        required_columns = ['sample_name', 'v1', 'v2', 'h1', 'h2', 'v1_new', 'v2_new', 'h1_new', 'h2_new']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns} in the {self.ROI_size} data")
            raise ValueError(f"Missing required columns in the {self.ROI_size} data: {missing_columns}")

        # Filter the DataFrame for rows where 'sample_name' matches
        sample_data = data[data['sample_name'] == sample_name]
        
        if not sample_data.empty:
            # Extract the first matching row
            row = sample_data.iloc[0]
            
            # Set attributes from the DataFrame row
            self.main_roi_v1 = row['v1']
            self.main_roi_v2 = row['v2']
            self.main_roi_h1 = row['h1']
            self.main_roi_h2 = row['h2']
            self.normalization_roi_v1 = row['v1_new']
            self.normalization_roi_v2 = row['v2_new']
            self.normalization_roi_h1 = row['h1_new']
            self.normalization_roi_h2 = row['h2_new']
            
            logging.info(f"ROI data successfully set for sample: {sample_name} using {self.ROI_size}")
        else:
            logging.error(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")
            raise ValueError(f"No data available for sample name: {sample_name} in the {self.ROI_size} data")
        
    ###################################################################################
    
    
   
   
   
   
   
   
   
class BSC_STFT_IBD():

    def __init__(self,
                 folder_path: str,
                 ROIs_path: str,
                 **kwargs) -> None:
        
        # Initialize instance variables
        self.folder_path: str = folder_path
        self.ROIs_path: str = ROIs_path
               
        self.device: str | None = kwargs.get("device", None)
        self.ac_method: str | None = kwargs.get("ac_method", None)
        self.alpha: float | None = kwargs.get("alpha", None)
        self.mode: str | None = kwargs.get("mode", None)
        self.visualize: bool | None = kwargs.get("visualize", None)
        self.nperseg: int | None = kwargs.get("stft_nperseg", None)
        self.noverlap: int | None = kwargs.get("stft_noverlap", None)

        # Additional fixed attributes
        self.size: str = "large"
        self.signal_type: str = "no_tgc"
        
        self.stft_window_length = None        
        self.main_roi_v1 = None
        self.main_roi_v2 = None
        self.main_roi_h1 = None
        self.main_roi_h2 = None
        self.normalization_roi_v1 = None
        self.normalization_roi_v2 = None
        self.normalization_roi_h1 = None
        self.normalization_roi_h2 = None    
        
        self.excel_all_data = None
        self.roi_data = None
                
        self.stft_results = {}

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################
    
    def __run_single(self):
        pass
    
    ###################################################################################
   
    def __run_multiple(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_CF_multiple()
           
    ###################################################################################

    def calculate_signal_duration(self, data_points, sampling_frequency):
        
        return data_points / sampling_frequency

    ###################################################################################
    
    def set_roi_df(self):
        large_data_list = []
        small_data_list = []

        for filename in os.listdir(self.ROIs_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_path = os.path.join(self.ROIs_path, filename)
                try:
                    xl = pd.ExcelFile(file_path)
                    sheet_names = xl.sheet_names

                    if "Large_ROI" in sheet_names:
                        large_data = xl.parse("Large_ROI")
                        large_data["ROI_Type"] = "Large"
                        large_data_list.append(large_data)

                    if "Small_ROI" in sheet_names:
                        small_data = xl.parse("Small_ROI")
                        small_data["ROI_Type"] = "Small"
                        small_data_list.append(small_data)

                except Exception as e:
                    logging.error(f"Error loading Excel file {file_path}: {e}")

        # Combine both with an indicator column
        all_data = small_data_list + large_data_list
        self.roi_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    ###################################################################################

    def compute_stft(self, signal_1d, fs, window, nperseg, noverlap):
        logging.debug("Entering compute_stft method.")
        logging.debug(f"Input parameters - fs: {fs}, nperseg: {nperseg}, noverlap: {noverlap}")

        step = nperseg - noverlap
        freqs = np.fft.fftfreq(nperseg, 1/fs)
        time_slices = np.arange(0, len(signal_1d) - nperseg, step)
        stft_matrix = np.zeros((len(freqs), len(time_slices)), dtype=complex)

        logging.debug(f"Step size for slicing: {step}")
        logging.debug(f"Number of time slices: {len(time_slices)}")

        for i, t in enumerate(time_slices):
            logging.debug(f"Processing time slice index: {i}, time: {t/fs} seconds")
            x_segment = signal_1d[t:t+nperseg] * window
            stft_matrix[:, i] = np.fft.fft(x_segment)

        freqs = np.fft.fftshift(freqs)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
        
        logging.debug("STFT computation completed, shifting frequencies and STFT matrix.")
        logging.debug(f"Output frequencies shape: {freqs.shape}, time slices shape: {time_slices.shape}, STFT matrix shape: {stft_matrix.shape}")

        logging.debug("Exiting compute_stft method.")
        return freqs, time_slices / fs, stft_matrix

    ###################################################################################
    
    def gaussian_window(self, window_length_0d, fs_0d):
        logging.debug("Entering gaussian_window method.")
        logging.debug(f"Input parameters - window_length_0d: {window_length_0d}, fs_0d: {fs_0d}")
        
        # Convert the number of points to an integer
        num_points = int(window_length_0d * fs_0d)
        logging.debug(f"Number of points calculated: {num_points}")

        time_1d = np.linspace(0, window_length_0d, num_points)  # Generating appropriate tau range
        logging.debug(f"Time vector generated with length: {len(time_1d)}")

        sigma = window_length_0d / 6  # Standard deviation
        logging.debug(f"Standard deviation (sigma) calculated: {sigma}")
        
        t_0 = window_length_0d / 2  # Central point (mean)
        logging.debug(f"Central point (t_0) calculated: {t_0}")

        amp_1d = np.exp(-((time_1d - t_0)**2) / (2 * sigma**2))
        normalization = np.sqrt(np.sqrt(np.pi)) 
        amp_1d /= normalization
        logging.debug("Amplitude vector calculated and normalized.")

        logging.debug("Exiting gaussian_window method.")
        return time_1d, amp_1d

    ###################################################################################

    def stft_1d(self, signal_1d, fs, window_length, normalize_signal):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        # Calculate signal energy and normalize if requested
        signal_energy = np.sum(np.abs(signal_1d) ** 2)
        logging.info(f"Signal energy calculated: {signal_energy}")

        if normalize_signal and signal_energy > 0:
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        #print("nperseg =", nperseg)
        
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, stft_result

    ###################################################################################

    def stft_2d(self, signal_2d, fs, window_length, normalize_signal, aggregation, mode="per_signal"):

        logging.debug("Entering stft_2d method.")

        if mode == "average_first":
            logging.debug("Averaging signals before STFT.")
            avg_signal = np.mean(signal_2d, axis=0)
            freqs, avg_amplitude = self.stft_1d(avg_signal, fs, window_length, normalize_signal)
            logging.info("Computed STFT on averaged signal.")
        elif mode == "per_signal":
            amplitude_list = []
            total_iterations = signal_2d.shape[0]
            logging.debug(f"Total iterations set to: {total_iterations}")

            for i in range(total_iterations):
                logging.debug(f"Processing signal index: {i}")
                signal_1d = signal_2d[i, :]
                freqs, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal)
                amplitude_list.append(amplitude)
                logging.debug(f"Appended amplitude for index {i}.")

            amplitude_stack = np.stack(amplitude_list, axis=0)

            if aggregation == "mean":
                avg_amplitude = np.mean(amplitude_stack, axis=0)
                logging.info("Calculated mean of amplitudes.")
            elif aggregation == "median":
                avg_amplitude = np.median(amplitude_stack, axis=0)
                logging.info("Calculated median of amplitudes.")
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logging.debug("Exiting stft_2d method.")
        return freqs, avg_amplitude

    ###################################################################################
    
    def plot_stft(self, time, freqs, stft_result, log=False):
        # Log the start of the plotting process
        logging.info("Starting to plot STFT.")

        # Convert frequencies from Hz to MHz
        freqs_mhz = freqs / 1e6
        logging.debug(f"Frequencies converted to MHz: {freqs_mhz}")

        # Convert time from seconds to microseconds
        time_us = time * 1e6
        logging.debug(f"Time converted to µs: {time_us}")

        # Plot STFT with both positive and negative frequencies
        plt.figure(figsize=(12, 6))

        # Check for logarithm and avoid log(0) by adding a small constant
        if log:
            stft_result = np.log(np.abs(stft_result) + 1e-10)  # Avoid log(0)
            logging.info("Applied logarithmic scaling to STFT result.")

        # Create the plot using imshow
        plt.imshow(np.abs(stft_result), aspect='auto',
                extent=[time_us.min(), time_us.max(), freqs_mhz.min(), freqs_mhz.max()], origin='lower', cmap='jet')
        
        plt.title('STFT with Gaussian Window')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [µs]')
        plt.colorbar(label='Magnitude')

        # Show the plot
        plt.show()
        logging.info("STFT plot displayed successfully.")

    ###################################################################################
    
    def calculate_stft_time(self, time, stft_result):
        # Ensure stft_result has at least one dimension
        if stft_result.ndim < 2:
            raise ValueError("stft_result must have at least two dimensions.")

        # Determine the number of time bins in the STFT result
        num_time_bins = stft_result.shape[1]

        # Calculate the total number of points in the time array
        total_time_points = len(time)

        # Log the original time array length and STFT result size
        logging.info(f"Original time array length: {total_time_points}")
        logging.info(f"Number of time bins in STFT result: {num_time_bins}")

        # Check if we can trim the time array to the required size
        if total_time_points < num_time_bins:
            raise ValueError("The time array is smaller than the STFT result time bins.")

        # Calculate how many points to remove from each side to achieve the target size
        total_points_to_remove = total_time_points - num_time_bins

        # Ensure we remove points from both sides equally
        if total_points_to_remove % 2 == 0:
            points_to_remove_each_side = total_points_to_remove // 2
        else:
            points_to_remove_each_side = total_points_to_remove // 2

        # Trim the time array from both ends
        trimmed_time = time[points_to_remove_each_side:total_time_points - points_to_remove_each_side]
        
        # Log the trimming process
        logging.info(f"Trimming time array: removing {points_to_remove_each_side} points from each side.")

        # Ensure trimmed_time has the correct size
        if len(trimmed_time) != num_time_bins:
            # If it doesn't match, recalculate the range to ensure correct length
            start_index = (total_time_points - num_time_bins) // 2
            end_index = start_index + num_time_bins
            trimmed_time = time[start_index:end_index]
            logging.info(f"Adjusted trimmed time to ensure correct size: {len(trimmed_time)}")

        # Final check
        if len(trimmed_time) != num_time_bins:
            raise ValueError("Trimming resulted in a time array that does not match the STFT result size.")

        # Log the final length of the trimmed time array
        logging.info(f"Final trimmed time array length: {len(trimmed_time)}")

        return trimmed_time

    ###################################################################################
   
    def prepare_stft_analysis_data_single(self, mode):
        logging.debug("Initializing data object for STFT analysis.")

        sample_name = self.get_sample_name_from_path(self.folder_path)
        self.set_roi_data_from_df(sample_name)
        
        # Log the mode of operation and sample name being processed
        logging.debug(f"Preparing STFT analysis data for mode: {mode}, sample: {sample_name}")

        main_roi_data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.main_roi_v1,
            v2=self.main_roi_v2,
            h1=self.main_roi_h1,
            h2=self.main_roi_h2,
            visualize=self.visualize,
            alpha=self.alpha,
            shift_signal=True

        )
        
        if self.normalization == "reference":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.normalization_roi_v1,
                v2=self.normalization_roi_v2,
                h1=self.normalization_roi_h1,
                h2=self.normalization_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
            
        elif self.normalization == "phantom":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.phantom_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
        
        # Log the creation of data objects for both main and normalization ROIs
        logging.debug(f"Data objects created for main and normalization ROIs of sample: {sample_name}")

        if mode == "1d":
            
            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 1D mode
            logging.debug(f"STFT 1D analysis completed for sample: {sample_name}")

        elif mode == "2d":

            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 2D mode
            logging.debug(f"STFT 2D analysis completed for sample: {sample_name}")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[main_roi_data_obj.sample_name] = {
            "frequencies": stft_freqs_main_roi,
            "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                            np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
        }
        logging.debug(f"Results stored for sample {main_roi_data_obj.sample_name}")

        logging.info("Completed processing for the provided mode.")

    ###################################################################################
        
    def prepare_stft_analysis_data_multiple(self, mode):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
        
            main_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
                
            if self.normalization == "reference":
                            
                normalization_roi_data_obj = Data(
                    sample_folder_path=folder_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.normalization_roi_v1,
                    v2=self.normalization_roi_v2,
                    h1=self.normalization_roi_h1,
                    h2=self.normalization_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                    )      
                
            elif self.normalization == "phantom":
            
                normalization_roi_data_obj = Data(
                    sample_folder_path=self.phantom_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.main_roi_v1,
                    v2=self.main_roi_v2,
                    h1=self.main_roi_h1,
                    h2=self.main_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                )
                    
            if main_roi_data_obj.sampling_rate_MHz:
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, main_roi_data_obj.sampling_rate_MHz * 1e6)

                if mode == "1d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                        main_roi_data_obj.trimmed_signal_1d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                        normalization_roi_data_obj.trimmed_signal_1d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }

                elif mode == "2d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                        main_roi_data_obj.trimmed_signal_2d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="median",
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                        normalization_roi_data_obj.trimmed_signal_2d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="median",
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }
                    
            else:
                raise ValueError(f"Sampling rate is not defined for sample '{main_roi_data_obj.sample_name}'.")
            
        logging.info("Completed processing all folders.")

    ###################################################################################
        
    def plot_stft_results(self):
        if not self.stft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.stft_results.items():
            
            signal_energy = results['signal_energy']

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Amplitude length: {len(signal_energy)}")

            plt.plot(freqs, signal_energy, label=sample_name)

        plt.title('STFT Amplitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel('Signal Energy')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    ###################################################################################
    @classmethod        
    def post_statistic_analysis_stft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.stft_results
        results_obj_2 = obj_2.stft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")

    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################

    def get_sample_name_from_path(self, folder_path):
        # Assuming the sample name is the last part of the folder path
        sample_name = folder_path.split(os.sep)[-1]
        logging.debug(f"Extracted sample name: {sample_name} from path: {self.folder_path}")
        return sample_name
            
    ###################################################################################
    
    def set_roi_data_from_df(self, sample_name):
        logging.debug(f"Setting ROI data for sample: {sample_name}")
        
        data = self.roi_data

        if data is None or data.empty:
            logging.error("No ROI data available.")
            raise ValueError("No ROI data available.")

        # Required columns
        required_columns = ['sample_name', 'v1', 'v2', 'h1', 'h2', 'v1_new', 'v2_new', 'h1_new', 'h2_new', 'ROI_Type']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in ROI data: {missing_columns}")
            raise ValueError(f"Missing required columns in ROI data: {missing_columns}")

        # Try Large ROI first
        sample_data = data[(data['sample_name'] == sample_name) & (data['ROI_Type'] == 'Large')]

        # If no Large ROI found, try Small ROI
        if sample_data.empty:
            sample_data = data[(data['sample_name'] == sample_name) & (data['ROI_Type'] == 'Small')]

        if not sample_data.empty:
            row = sample_data.iloc[0]

            self.main_roi_v1 = row['v1']
            self.main_roi_v2 = row['v2']
            self.main_roi_h1 = row['h1']
            self.main_roi_h2 = row['h2']
            self.normalization_roi_v1 = row['v1_new']
            self.normalization_roi_v2 = row['v2_new']
            self.normalization_roi_h1 = row['h1_new']
            self.normalization_roi_h2 = row['h2_new']

            logging.info(f"ROI data successfully set for sample: {sample_name} (ROI_Type: {row['ROI_Type']})")
        else:
            logging.error(f"No data available for sample name: {sample_name}")
            raise ValueError(f"No data available for sample name: {sample_name}")

    ###################################################################################
    
    def prepare_stft_analysis_data_CF_multiple(self):
        
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.warning(f"Processing folder: {folder_path}")

            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
                    
            data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
            
            print(f"Processing folder 1: {folder_path}")

            if data_obj.trimmed_signal_2d is not None:
                
                print(f"Processing folder 2: {folder_path}")

                logging.debug(f"Data object initialized with mode: {data_obj.mode}")
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, data_obj.sampling_rate_MHz * 1e6)

                logging.debug("Starting 2D STFT analysis.")
                all_stft_freqs, all_central_freq, all_stft_times, all_stft_analysis_coefficients = self.stft_2d_CF(
                    data_obj.trimmed_signal_2d,
                    data_obj.sampling_rate_MHz * 1e6,
                    window_length=self.stft_window_length,
                    normalize_signal=False,
                    apply_gaussian=False
                )
                logging.debug("2D STFT analysis completed.")
            
                # Wrap in a list so it's indexable by signal_index later
                depth = [data_obj.trimmed_depth_1d]

                # Store the result in the dictionary
                self.stft_results[data_obj.sample_name] = {
                    #"frequencies": all_stft_freqs,                  # List of arrays per signal
                    "central_freq": all_central_freq,               # List of arrays per signal
                    #"times": all_stft_times,                        # List of arrays per signal
                    "depth": depth,                                 # List of depth arrays per signal
                    #"stft_signal": all_stft_analysis_coefficients,  # List of arrays per signal
                    #"original_signal": data_obj.trimmed_signal_2d   # shape: (186, 489)
                }

                logging.debug(f"Results stored for sample {data_obj.sample_name}")
                logging.info("Completed processing all folders.")
        
    ###################################################################################
    
    def stft_2d_CF(self, signal_2d, fs, window_length, normalize_signal=False, apply_gaussian=False, gaussian_sigma=1):
        logging.debug("Entering stft_2d_CF method.")
        
        freqs_list = []
        times_list = []
        amplitudes_list = []
        central_freq_list = []

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")
            
            signal_1d = signal_2d[i, :]
            freqs, times, amplitude = self.stft_1d_cf(
                signal_1d, fs, window_length, normalize_signal=normalize_signal
            )

            valid_indices = np.arange(1, len(freqs))

            if apply_gaussian:
                logging.debug(f"Applying Gaussian smoothing to STFT amplitude (freq axis) for signal index {i}")
                amplitude = gaussian_filter1d(amplitude, sigma=gaussian_sigma, axis=0)

            central_freq = np.array([
                freqs[valid_indices[amp[valid_indices].argmax()]] for amp in amplitude.T
            ])

            freqs_list.append(freqs)
            times_list.append(times)
            amplitudes_list.append(amplitude)
            central_freq_list.append(central_freq)
            
        # Konvertieren zu Arrays, um Mittelwert zu berechnen
        all_freqs = np.mean(freqs_list, axis=0)
        all_times = np.mean(times_list, axis=0)
        all_amplitudes = np.mean(amplitudes_list, axis=0)
        all_central_freq = np.mean(central_freq_list, axis=0)

        return all_freqs, all_central_freq, all_times, all_amplitudes

    ###################################################################################
    
    def stft_1d_cf(self, signal_1d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        if normalize_signal:
            # Calculate signal energy and normalize if requested
            signal_energy = np.sum(np.abs(signal_1d) ** 2)
            logging.info(f"Signal energy calculated: {signal_energy}")
            
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, times, stft_result
    
    ###################################################################################





class alpha_IBD():

    def __init__(self,
                 folder_path: str,
                 ROIs_path: str,
                 **kwargs) -> None:
        
        # Initialize instance variables
        self.folder_path: str = folder_path
        self.ROIs_path: str = ROIs_path
               
        self.device: str | None = kwargs.get("device", None)
        self.ac_method: str | None = kwargs.get("ac_method", None)
        self.alpha: float | None = kwargs.get("alpha", None)
        self.mode: str | None = kwargs.get("mode", None)
        self.visualize: bool | None = kwargs.get("visualize", None)
        self.nperseg: int | None = kwargs.get("stft_nperseg", None)
        self.noverlap: int | None = kwargs.get("stft_noverlap", None)

        # Additional fixed attributes
        self.size: str = "large"
        self.signal_type: str = "no_tgc"
        
        self.stft_window_length = None        
        self.main_roi_v1 = None
        self.main_roi_v2 = None
        self.main_roi_h1 = None
        self.main_roi_h2 = None
        self.normalization_roi_v1 = None
        self.normalization_roi_v2 = None
        self.normalization_roi_h1 = None
        self.normalization_roi_h2 = None    
        
        self.excel_all_data = None
        self.roi_data = None
                
        self.stft_results = {}

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################
    
    def __run_single(self):
        pass
    
    ###################################################################################
   
    def __run_multiple(self):
        
        self.set_roi_df()    
        self.prepare_stft_analysis_data_CF_multiple()
           
    ###################################################################################

    def calculate_signal_duration(self, data_points, sampling_frequency):
        
        return data_points / sampling_frequency

    ###################################################################################
    
    def set_roi_df(self):
        large_data_list = []
        small_data_list = []

        for filename in os.listdir(self.ROIs_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_path = os.path.join(self.ROIs_path, filename)
                try:
                    xl = pd.ExcelFile(file_path)
                    sheet_names = xl.sheet_names

                    if "Large_ROI" in sheet_names:
                        large_data = xl.parse("Large_ROI")
                        large_data["ROI_Type"] = "Large"
                        large_data_list.append(large_data)

                    if "Small_ROI" in sheet_names:
                        small_data = xl.parse("Small_ROI")
                        small_data["ROI_Type"] = "Small"
                        small_data_list.append(small_data)

                except Exception as e:
                    logging.error(f"Error loading Excel file {file_path}: {e}")

        # Combine both with an indicator column
        all_data = small_data_list + large_data_list
        self.roi_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    ###################################################################################

    def compute_stft(self, signal_1d, fs, window, nperseg, noverlap):
        logging.debug("Entering compute_stft method.")
        logging.debug(f"Input parameters - fs: {fs}, nperseg: {nperseg}, noverlap: {noverlap}")

        step = nperseg - noverlap
        freqs = np.fft.fftfreq(nperseg, 1/fs)
        time_slices = np.arange(0, len(signal_1d) - nperseg, step)
        stft_matrix = np.zeros((len(freqs), len(time_slices)), dtype=complex)

        logging.debug(f"Step size for slicing: {step}")
        logging.debug(f"Number of time slices: {len(time_slices)}")

        for i, t in enumerate(time_slices):
            logging.debug(f"Processing time slice index: {i}, time: {t/fs} seconds")
            x_segment = signal_1d[t:t+nperseg] * window
            stft_matrix[:, i] = np.fft.fft(x_segment)

        freqs = np.fft.fftshift(freqs)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
        
        logging.debug("STFT computation completed, shifting frequencies and STFT matrix.")
        logging.debug(f"Output frequencies shape: {freqs.shape}, time slices shape: {time_slices.shape}, STFT matrix shape: {stft_matrix.shape}")

        logging.debug("Exiting compute_stft method.")
        return freqs, time_slices / fs, stft_matrix

    ###################################################################################
    
    def gaussian_window(self, window_length_0d, fs_0d):
        logging.debug("Entering gaussian_window method.")
        logging.debug(f"Input parameters - window_length_0d: {window_length_0d}, fs_0d: {fs_0d}")
        
        # Convert the number of points to an integer
        num_points = int(window_length_0d * fs_0d)
        logging.debug(f"Number of points calculated: {num_points}")

        time_1d = np.linspace(0, window_length_0d, num_points)  # Generating appropriate tau range
        logging.debug(f"Time vector generated with length: {len(time_1d)}")

        sigma = window_length_0d / 6  # Standard deviation
        logging.debug(f"Standard deviation (sigma) calculated: {sigma}")
        
        t_0 = window_length_0d / 2  # Central point (mean)
        logging.debug(f"Central point (t_0) calculated: {t_0}")

        amp_1d = np.exp(-((time_1d - t_0)**2) / (2 * sigma**2))
        normalization = np.sqrt(np.sqrt(np.pi)) 
        amp_1d /= normalization
        logging.debug("Amplitude vector calculated and normalized.")

        logging.debug("Exiting gaussian_window method.")
        return time_1d, amp_1d

    ###################################################################################

    def stft_1d(self, signal_1d, fs, window_length, normalize_signal):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        # Calculate signal energy and normalize if requested
        signal_energy = np.sum(np.abs(signal_1d) ** 2)
        logging.info(f"Signal energy calculated: {signal_energy}")

        if normalize_signal and signal_energy > 0:
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        #print("nperseg =", nperseg)
        
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, stft_result

    ###################################################################################

    def stft_2d(self, signal_2d, fs, window_length, normalize_signal, aggregation, mode="per_signal"):

        logging.debug("Entering stft_2d method.")

        if mode == "average_first":
            logging.debug("Averaging signals before STFT.")
            avg_signal = np.mean(signal_2d, axis=0)
            freqs, avg_amplitude = self.stft_1d(avg_signal, fs, window_length, normalize_signal)
            logging.info("Computed STFT on averaged signal.")
        elif mode == "per_signal":
            amplitude_list = []
            total_iterations = signal_2d.shape[0]
            logging.debug(f"Total iterations set to: {total_iterations}")

            for i in range(total_iterations):
                logging.debug(f"Processing signal index: {i}")
                signal_1d = signal_2d[i, :]
                freqs, amplitude = self.stft_1d(signal_1d, fs, window_length, normalize_signal)
                amplitude_list.append(amplitude)
                logging.debug(f"Appended amplitude for index {i}.")

            amplitude_stack = np.stack(amplitude_list, axis=0)

            if aggregation == "mean":
                avg_amplitude = np.mean(amplitude_stack, axis=0)
                logging.info("Calculated mean of amplitudes.")
            elif aggregation == "median":
                avg_amplitude = np.median(amplitude_stack, axis=0)
                logging.info("Calculated median of amplitudes.")
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logging.debug("Exiting stft_2d method.")
        return freqs, avg_amplitude

    ###################################################################################
    
    def plot_stft(self, time, freqs, stft_result, log=False):
        # Log the start of the plotting process
        logging.info("Starting to plot STFT.")

        # Convert frequencies from Hz to MHz
        freqs_mhz = freqs / 1e6
        logging.debug(f"Frequencies converted to MHz: {freqs_mhz}")

        # Convert time from seconds to microseconds
        time_us = time * 1e6
        logging.debug(f"Time converted to µs: {time_us}")

        # Plot STFT with both positive and negative frequencies
        plt.figure(figsize=(12, 6))

        # Check for logarithm and avoid log(0) by adding a small constant
        if log:
            stft_result = np.log(np.abs(stft_result) + 1e-10)  # Avoid log(0)
            logging.info("Applied logarithmic scaling to STFT result.")

        # Create the plot using imshow
        plt.imshow(np.abs(stft_result), aspect='auto',
                extent=[time_us.min(), time_us.max(), freqs_mhz.min(), freqs_mhz.max()], origin='lower', cmap='jet')
        
        plt.title('STFT with Gaussian Window')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [µs]')
        plt.colorbar(label='Magnitude')

        # Show the plot
        plt.show()
        logging.info("STFT plot displayed successfully.")

    ###################################################################################
    
    def calculate_stft_time(self, time, stft_result):
        # Ensure stft_result has at least one dimension
        if stft_result.ndim < 2:
            raise ValueError("stft_result must have at least two dimensions.")

        # Determine the number of time bins in the STFT result
        num_time_bins = stft_result.shape[1]

        # Calculate the total number of points in the time array
        total_time_points = len(time)

        # Log the original time array length and STFT result size
        logging.info(f"Original time array length: {total_time_points}")
        logging.info(f"Number of time bins in STFT result: {num_time_bins}")

        # Check if we can trim the time array to the required size
        if total_time_points < num_time_bins:
            raise ValueError("The time array is smaller than the STFT result time bins.")

        # Calculate how many points to remove from each side to achieve the target size
        total_points_to_remove = total_time_points - num_time_bins

        # Ensure we remove points from both sides equally
        if total_points_to_remove % 2 == 0:
            points_to_remove_each_side = total_points_to_remove // 2
        else:
            points_to_remove_each_side = total_points_to_remove // 2

        # Trim the time array from both ends
        trimmed_time = time[points_to_remove_each_side:total_time_points - points_to_remove_each_side]
        
        # Log the trimming process
        logging.info(f"Trimming time array: removing {points_to_remove_each_side} points from each side.")

        # Ensure trimmed_time has the correct size
        if len(trimmed_time) != num_time_bins:
            # If it doesn't match, recalculate the range to ensure correct length
            start_index = (total_time_points - num_time_bins) // 2
            end_index = start_index + num_time_bins
            trimmed_time = time[start_index:end_index]
            logging.info(f"Adjusted trimmed time to ensure correct size: {len(trimmed_time)}")

        # Final check
        if len(trimmed_time) != num_time_bins:
            raise ValueError("Trimming resulted in a time array that does not match the STFT result size.")

        # Log the final length of the trimmed time array
        logging.info(f"Final trimmed time array length: {len(trimmed_time)}")

        return trimmed_time

    ###################################################################################
   
    def prepare_stft_analysis_data_single(self, mode):
        logging.debug("Initializing data object for STFT analysis.")

        sample_name = self.get_sample_name_from_path(self.folder_path)
        self.set_roi_data_from_df(sample_name)
        
        # Log the mode of operation and sample name being processed
        logging.debug(f"Preparing STFT analysis data for mode: {mode}, sample: {sample_name}")

        main_roi_data_obj = Data(
            sample_folder_path=self.folder_path,
            device=self.device,
            size=self.size,
            signal_type=self.signal_type,
            ac_method=self.ac_method,
            mode="read_data",
            v1=self.main_roi_v1,
            v2=self.main_roi_v2,
            h1=self.main_roi_h1,
            h2=self.main_roi_h2,
            visualize=self.visualize,
            alpha=self.alpha,
            shift_signal=True

        )
        
        if self.normalization == "reference":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.normalization_roi_v1,
                v2=self.normalization_roi_v2,
                h1=self.normalization_roi_h1,
                h2=self.normalization_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
            
        elif self.normalization == "phantom":
        
            normalization_roi_data_obj = Data(
                sample_folder_path=self.phantom_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha
            )
        
        # Log the creation of data objects for both main and normalization ROIs
        logging.debug(f"Data objects created for main and normalization ROIs of sample: {sample_name}")

        if mode == "1d":
            
            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 1D mode
            logging.debug(f"STFT 1D analysis completed for sample: {sample_name}")

        elif mode == "2d":

            stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                main_roi_data_obj.trimmed_signal_1d,
                main_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                normalization_roi_data_obj.trimmed_signal_1d,
                normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                window_length=self.stft_window_length,
                normalize_signal=False
                )
            
            # Log the completion of STFT analysis for 2D mode
            logging.debug(f"STFT 2D analysis completed for sample: {sample_name}")

        # Store the result in the dictionary, using the folder path as the key
        self.stft_results[main_roi_data_obj.sample_name] = {
            "frequencies": stft_freqs_main_roi,
            "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                            np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
        }
        logging.debug(f"Results stored for sample {main_roi_data_obj.sample_name}")

        logging.info("Completed processing for the provided mode.")

    ###################################################################################
        
    def prepare_stft_analysis_data_multiple(self, mode):
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.info(f"Processing folder: {folder_path}")
            
            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
        
            main_roi_data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
                
            if self.normalization == "reference":
                            
                normalization_roi_data_obj = Data(
                    sample_folder_path=folder_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.normalization_roi_v1,
                    v2=self.normalization_roi_v2,
                    h1=self.normalization_roi_h1,
                    h2=self.normalization_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                    )      
                
            elif self.normalization == "phantom":
            
                normalization_roi_data_obj = Data(
                    sample_folder_path=self.phantom_path,
                    device=self.device,
                    size=self.size,
                    signal_type=self.signal_type,
                    ac_method=self.ac_method,
                    mode="read_data",
                    v1=self.main_roi_v1,
                    v2=self.main_roi_v2,
                    h1=self.main_roi_h1,
                    h2=self.main_roi_h2,
                    visualize=self.visualize,
                    alpha=self.alpha,
                    shift_signal=True
                )
                    
            if main_roi_data_obj.sampling_rate_MHz:
                
                self.stft_window_length = self.calculate_signal_duration(self.nperseg, main_roi_data_obj.sampling_rate_MHz * 1e6)

                if mode == "1d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_1d(
                        main_roi_data_obj.trimmed_signal_1d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_1d(
                        normalization_roi_data_obj.trimmed_signal_1d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }

                elif mode == "2d":
                    
                    stft_freqs_main_roi, stft_analysis_coefficients_main_roi = self.stft_2d(
                        main_roi_data_obj.trimmed_signal_2d,
                        main_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="median",
                        )
                    
                    _, stft_analysis_coefficients_normalization_roi = self.stft_2d(
                        normalization_roi_data_obj.trimmed_signal_2d,
                        normalization_roi_data_obj.sampling_rate_MHz * 1e6,
                        window_length=self.stft_window_length,
                        normalize_signal=False,
                        aggregation="median",
                        )
                    
                    # Store the result in the dictionary, using the folder path as the key
                    self.stft_results[main_roi_data_obj.sample_name] = {
                        "frequencies": stft_freqs_main_roi,
                        "signal_energy": np.sum(np.abs(stft_analysis_coefficients_main_roi) ** 2, axis=1) /
                                         np.sum(np.abs(stft_analysis_coefficients_normalization_roi) ** 2, axis=1)
                    }
                    
            else:
                raise ValueError(f"Sampling rate is not defined for sample '{main_roi_data_obj.sample_name}'.")
            
        logging.info("Completed processing all folders.")

    ###################################################################################
        
    def plot_stft_results(self):
        if not self.stft_results:
            logging.warning("No STFT results available to plot.")
            return

        plt.figure(figsize=(12, 6))  # Adjust the figure size

        for sample_name, results in self.stft_results.items():
            
            signal_energy = results['signal_energy']

            freqs = results['frequencies'] / 1e6  # Convert frequencies to MHz

            # Debugging: Print sizes of frequencies and amplitude
            logging.info(f"Sample: {sample_name}, Frequencies length: {len(freqs)}, Amplitude length: {len(signal_energy)}")

            plt.plot(freqs, signal_energy, label=sample_name)

        plt.title('STFT Amplitude vs Frequencies for Samples')
        plt.xlabel('Frequency (MHz)')  # Updated label to MHz
        plt.ylabel('Signal Energy')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    ###################################################################################
    @classmethod        
    def post_statistic_analysis_stft(self, obj_1, obj_2, name_1, name_2):
        
        logging.debug("Starting post-analysis for stft results...")

        results_obj_1 = obj_1.stft_results
        results_obj_2 = obj_2.stft_results

        logging.debug("Extracting frequencies and amplitudes...")
        # Step 1: Get scales directly from the objects
        frequency_obj_1 = np.array([data["frequencies"] for data in results_obj_1.values()]) / 1e6
        frequency_obj_2 = np.array([data["frequencies"] for data in results_obj_2.values()]) / 1e6

        frequency_obj_1 = frequency_obj_1[0]
        frequency_obj_2 = frequency_obj_2[0]

        # Step 2: Collect amplitude data for each dataset
        all_amplitudes_obj_1 = np.array([data["signal_energy"] for data in results_obj_1.values()])
        all_amplitudes_obj_2 = np.array([data["signal_energy"] for data in results_obj_2.values()])

        logging.debug("Computing mean and standard deviation for each frequency...")
        # Compute mean and standard deviation for each frequency
        mean_amplitude_1 = np.mean(all_amplitudes_obj_1, axis=0)
        std_amplitude_1 = np.std(all_amplitudes_obj_1, axis=0)

        mean_amplitude_2 = np.mean(all_amplitudes_obj_2, axis=0)
        std_amplitude_2 = np.std(all_amplitudes_obj_2, axis=0)

        logging.debug("Performing t-tests across frequencies...")
        # Perform a t-test at each frequency to compare the means between the two datasets
        p_values = []
        for i in range(len(frequency_obj_1)):
            _, p_value = ttest_ind(all_amplitudes_obj_1[:, i], all_amplitudes_obj_2[:, i])
            p_values.append(p_value)

        p_values = np.array(p_values)

        logging.debug("Identifying maximum mean amplitude and corresponding frequency...")
        # Identify the maximum mean amplitude and corresponding frequency for each dataset
        max_index_1 = np.argmax(mean_amplitude_1)
        max_freq_1 = frequency_obj_1[max_index_1]

        max_index_2 = np.argmax(mean_amplitude_2)
        max_freq_2 = frequency_obj_2[max_index_2]

        # Get the number of samples for each case
        num_samples_1 = len(results_obj_1)
        num_samples_2 = len(results_obj_2)

        logging.debug("Plotting results...")
        # Plot the mean amplitude with error bars (standard deviation) for the two datasets on the same graph
        plt.figure(figsize=(12, 8))
        plt.title("STFT Analysis: Mean Amplitude Comparison", fontsize=16)

        plt.plot(frequency_obj_1, mean_amplitude_1, label=f'Mean Amplitude ({name_1}, n={num_samples_1})', color='blue', marker='o')
        plt.fill_between(frequency_obj_1, mean_amplitude_1 - std_amplitude_1, mean_amplitude_1 + std_amplitude_1, color='lightblue', alpha=0.5)

        plt.plot(frequency_obj_2, mean_amplitude_2, label=f'Mean Amplitude ({name_2}, n={num_samples_2})', color='red', marker='o')
        plt.fill_between(frequency_obj_2, mean_amplitude_2 - std_amplitude_2, mean_amplitude_2 + std_amplitude_2, color='lightcoral', alpha=0.5)

        plt.axvline(x=max_freq_1, color='blue', linestyle='--', label=f'Max Mean {name_1} ({max_freq_1:.2f} MHz)')
        plt.axvline(x=max_freq_2, color='red', linestyle='--', label=f'Max Mean {name_2} ({max_freq_2:.2f} MHz)')

        # Mark frequencies where the p-value is below the significance level (e.g., 0.05)
        significant_freqs = frequency_obj_1[p_values < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='green', linestyle=':')

        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Signal Energy', fontsize=14)
        plt.legend()
        plt.show()

        logging.debug("Analysis complete. Printing significant frequencies...")
        # Print the frequencies with significant differences and their corresponding p-values
        print("Frequencies with significant differences and their p-values:")
        for i, freq in enumerate(frequency_obj_1):
            if p_values[i] < 0.05:
                print(f"{name_1} vs {name_2} - Frequency: {freq:.2f} MHz, p-value: {p_values[i]:.4f}")

    ###################################################################################
    @staticmethod
    def make_signal_zero_mean(signal):
        # Calculate the mean of the signal
        mean_value = np.mean(signal)
        
        # Subtract the mean from each element of the signal to make it zero mean
        zero_mean_signal = signal - mean_value
        
        return zero_mean_signal
    
    ###################################################################################

    def get_sample_name_from_path(self, folder_path):
        # Assuming the sample name is the last part of the folder path
        sample_name = folder_path.split(os.sep)[-1]
        logging.debug(f"Extracted sample name: {sample_name} from path: {self.folder_path}")
        return sample_name
            
    ###################################################################################
    
    def set_roi_data_from_df(self, sample_name):
        logging.debug(f"Setting ROI data for sample: {sample_name}")
        
        data = self.roi_data

        if data is None or data.empty:
            logging.error("No ROI data available.")
            raise ValueError("No ROI data available.")

        # Required columns
        required_columns = ['sample_name', 'v1', 'v2', 'h1', 'h2', 'v1_new', 'v2_new', 'h1_new', 'h2_new', 'ROI_Type']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in ROI data: {missing_columns}")
            raise ValueError(f"Missing required columns in ROI data: {missing_columns}")

        # Try Large ROI first
        sample_data = data[(data['sample_name'] == sample_name) & (data['ROI_Type'] == 'Large')]

        # If no Large ROI found, try Small ROI
        if sample_data.empty:
            sample_data = data[(data['sample_name'] == sample_name) & (data['ROI_Type'] == 'Small')]

        if not sample_data.empty:
            row = sample_data.iloc[0]

            self.main_roi_v1 = row['v1']
            self.main_roi_v2 = row['v2']
            self.main_roi_h1 = row['h1']
            self.main_roi_h2 = row['h2']
            self.normalization_roi_v1 = row['v1_new']
            self.normalization_roi_v2 = row['v2_new']
            self.normalization_roi_h1 = row['h1_new']
            self.normalization_roi_h2 = row['h2_new']

            logging.info(f"ROI data successfully set for sample: {sample_name} (ROI_Type: {row['ROI_Type']})")
        else:
            logging.error(f"No data available for sample name: {sample_name}")
            raise ValueError(f"No data available for sample name: {sample_name}")

    ###################################################################################
    
    def prepare_stft_analysis_data_CF_multiple(self):
        
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]

        logging.info(f"Found {len(folder_paths)} folders to process in {self.folder_path}")

        for folder_path in folder_paths:
            logging.warning(f"Processing folder: {folder_path}")

            sample_name = self.get_sample_name_from_path(folder_path)
            self.set_roi_data_from_df(sample_name)
                    
            data_obj = Data(
                sample_folder_path=folder_path,
                device=self.device,
                size=self.size,
                signal_type=self.signal_type,
                ac_method=self.ac_method,
                mode="read_data",
                v1=self.main_roi_v1,
                v2=self.main_roi_v2,
                h1=self.main_roi_h1,
                h2=self.main_roi_h2,
                visualize=self.visualize,
                alpha=self.alpha,
                shift_signal=True
                )
            
            print(f"Processing folder 1: {folder_path}")

            if data_obj.trimmed_signal_2d is not None:
                
                print(f"Processing folder 2: {folder_path}")

                # ---- Modified usage for 2D signal array ----
                # signal_2d is your 2D array, each row is a 1D signal line
                signal_2d = data_obj.trimmed_signal_2d  # Make sure this exists
                fs = data_obj.sampling_rate_MHz * 1e6
                time_array = data_obj.trimmed_time_1d

                alpha_values = []

                for i, signal_1d in enumerate(signal_2d):
                    alpha = self.estimate_attenuation_coefficient(
                        signal=signal_1d,
                        fs=fs,
                        time_array=time_array,
                        visualize=False  # Set to True to visualize each line (optional)
                    )
                    alpha_values.append(alpha)
                average_alpha = np.mean(alpha_values)
                
                print(f"Average estimated attenuation coefficient: {34.72 * average_alpha:.4f} [dB/MHz/cm]")

                # Store the result in the dictionary
                self.stft_results[data_obj.sample_name] = {
                    "average_alpha": average_alpha,               # List of arrays per signal
                    "alpha": alpha,
                }

                logging.debug(f"Results stored for sample {data_obj.sample_name}")
                logging.info("Completed processing all folders.")
        
    ###################################################################################
    
    def stft_2d_CF(self, signal_2d, fs, window_length, normalize_signal=False, apply_gaussian=False, gaussian_sigma=1):
        logging.debug("Entering stft_2d_CF method.")
        
        freqs_list = []
        times_list = []
        amplitudes_list = []
        central_freq_list = []

        for i in range(signal_2d.shape[0]):
            logging.debug(f"Processing signal index: {i}")
            
            signal_1d = signal_2d[i, :]
            freqs, times, amplitude = self.stft_1d_cf(
                signal_1d, fs, window_length, normalize_signal=normalize_signal
            )

            valid_indices = np.arange(1, len(freqs))

            if apply_gaussian:
                logging.debug(f"Applying Gaussian smoothing to STFT amplitude (freq axis) for signal index {i}")
                amplitude = gaussian_filter1d(amplitude, sigma=gaussian_sigma, axis=0)

            central_freq = np.array([
                freqs[valid_indices[amp[valid_indices].argmax()]] for amp in amplitude.T
            ])

            freqs_list.append(freqs)
            times_list.append(times)
            amplitudes_list.append(amplitude)
            central_freq_list.append(central_freq)
            
        # Konvertieren zu Arrays, um Mittelwert zu berechnen
        all_freqs = np.mean(freqs_list, axis=0)
        all_times = np.mean(times_list, axis=0)
        all_amplitudes = np.mean(amplitudes_list, axis=0)
        all_central_freq = np.mean(central_freq_list, axis=0)

        return all_freqs, all_central_freq, all_times, all_amplitudes

    ###################################################################################
    
    def stft_1d_cf(self, signal_1d, fs, window_length, normalize_signal=False):
        logging.debug("Entering stft_1d method.")
        
        signal_1d = signal_1d.astype(np.float64)  # or np.float32, depending on your precision needs
        logging.debug("Converted signal to float64 for processing.")

        if normalize_signal:
            # Calculate signal energy and normalize if requested
            signal_energy = np.sum(np.abs(signal_1d) ** 2)
            logging.info(f"Signal energy calculated: {signal_energy}")
            
            signal_1d /= np.sqrt(signal_energy)
            logging.info("Signal normalized based on its energy.")
        else:
            logging.info("Normalization not applied; signal energy is zero or normalization not requested.")

        _, window_amp = self.gaussian_window(window_length, fs)
        nperseg = window_amp.shape[0]
        # noverlap = nperseg - 1
        logging.info(f"Window length (nperseg): {nperseg}, overlap (noverlap): {self.noverlap}")

        # Compute STFT
        freqs, times, stft_result = self.compute_stft(signal_1d, fs, window_amp, nperseg=nperseg, noverlap=self.noverlap)
        logging.info("STFT computation completed.")

        # Filter to keep only positive frequencies
        positive_freqs_mask = freqs >= 0
        freqs = freqs[positive_freqs_mask]
        stft_result = stft_result[positive_freqs_mask, :]
        logging.info(f"Filtered positive frequencies. Number of frequencies after filtering: {len(freqs)}")

        logging.debug("Exiting stft_1d method.")
        return freqs, times, stft_result
    
    ###################################################################################

    def gaussian(self, x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    ###################################################################################
    
    def estimate_attenuation_coefficient(self, signal, fs, time_array, visualize=True):
        SPEED_OF_SOUND = 1540  # m/s
        nperseg = self.nperseg
        overlap = self.noverlap

        frequencies, times, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=overlap)
        amplitudes = np.abs(Zxx)

        z_values = (SPEED_OF_SOUND * np.linspace(time_array[0], time_array[-1], len(times))) / 2 * 100  # in cm

        # Adjust depth array to match STFT time segments
        if len(z_values) > len(times):
            z_values = z_values[:len(times)]
        elif len(z_values) < len(times):
            z_values = np.interp(np.linspace(0, len(z_values)-1, len(times)), np.arange(len(z_values)), z_values)

        # Remove last point
        z_values = z_values[:-1]
        times = times[:-1]
        amplitudes = amplitudes[:, :-1]

        if visualize:
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(np.log(z_values), frequencies / 1e6, amplitudes, shading='gouraud', cmap='jet')
            plt.colorbar(label="Amplitude")
            plt.xlabel("Depth (cm)")
            plt.ylabel("Frequency (MHz)")
            plt.title("STFT Spectrogram of the Signal")
            plt.ylim([0, np.max(frequencies) / 1e6])
            plt.show()

        center_frequencies = []
        gaussian_fits = []

        for i in range(len(z_values)):
            spectrum = amplitudes[:, i]
            initial_guess = [np.max(spectrum), frequencies[np.argmax(spectrum)], 1e6]
            try:
                popt, _ = curve_fit(self.gaussian, frequencies, spectrum, p0=initial_guess)
                center_frequencies.append(popt[1])
                gaussian_fits.append(popt)
            except RuntimeError:
                center_frequencies.append(frequencies[np.argmax(spectrum)])
                gaussian_fits.append(None)

        center_frequencies = np.array(center_frequencies) / 1e6  # to MHz

        # Extract sigma_w from Gaussian fits (sigma = popt[2])
        sigma_ws = [fit[2] for fit in gaussian_fits if fit is not None]
        sigma_w = np.mean(sigma_ws) / 1e6
        
        f0 = center_frequencies[0]

        def linear_model(z, alpha):
            return f0 - 4 * sigma_w**2 * alpha * z

        popt, _ = curve_fit(linear_model, z_values, center_frequencies[:len(z_values)])
        estimated_alpha = popt[0]

        if visualize:
            plt.figure(figsize=(10, 6))
            plt.scatter(z_values, center_frequencies, label="Observed Center Frequencies", color="red", marker="o")
            plt.plot(z_values, linear_model(z_values, estimated_alpha),
                    label=f"Fitted Line (α = {estimated_alpha:.4f})", linestyle="--", color="blue")
            plt.xlabel("Depth (cm)")
            plt.ylabel("Center Frequency (MHz)")
            plt.title("Curve Fitting to Estimate Attenuation Coefficient")
            plt.legend()
            plt.grid()
            plt.show()

            for i, t in enumerate(times):
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))

                segment = signal[int(t * fs): int((t + nperseg / fs) * fs)]
                axes[0].plot(segment, color='blue')
                axes[0].set_title(f"Signal Segment at Depth {z_values[i]:.2f} cm")
                axes[0].set_xlabel("Sample Index")
                axes[0].set_ylabel("Amplitude")

                axes[1].plot(frequencies / 1e6, amplitudes[:, i], color='green', label="Observed Spectrum")
                if gaussian_fits[i] is not None:
                    fitted_curve = self.gaussian(frequencies, *gaussian_fits[i])
                    axes[1].plot(frequencies / 1e6, fitted_curve, color='red', linestyle="--", label="Gaussian Fit")
                    axes[1].axvline(center_frequencies[i], color='blue', linestyle=":", label=f"Peak @ {center_frequencies[i]:.2f} MHz")

                axes[1].set_title(f"Frequency Spectrum at Depth {z_values[i]:.2f} cm")
                axes[1].set_xlabel("Frequency (MHz)")
                axes[1].set_ylabel("Amplitude")
                axes[1].legend()

                plt.tight_layout()
                plt.show()

        return estimated_alpha

    ###################################################################################



