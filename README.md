# Backscatter Coefficient (BSC) Analysis Tool

A Python-based tool for processing and analyzing ultrasound data, specifically focused on Backscatter Coefficient (BSC) analysis from Clarius ultrasound devices. This tool provides functionality for processing raw ultrasound data BSC calculation.

## Features

- Process raw ultrasound data from Clarius devices (C3)
- Extract and handle RF (Radio Frequency) data
- Interactive ROI (Region of Interest) selection with GUI
- BSC calculation with multiple normalization methods:
  - Phantom-based normalization
  - Healthy liver tissue normalization
  - Constant alpha attenuation correction
  - Calculated alpha attenuation correction


## Project Structure

```
BSC/
├── data/                    # Data directory
│   ├── phantom/            # Phantom reference data
│   ├── ROIs/               # Region of Interest data files (.xlsx)
│   ├── results/            # BSC calculation results
│   └── samples/            # Ultrasound sample data
│       └── UKDCEUS*/       # Individual case directories
├── notebook/               # Jupyter notebooks
│   ├── BSC.ipynb          # BSC analysis notebook
│   ├── results.ipynb      # Results analysis notebook
│   └── unpacker.ipynb     # Data unpacking notebook
├── scr/                    # Source code
│   ├── app/
│   │   └── app_roi_selection.py  # ROI selection GUI
│   └── clarius/           # Clarius data processing
│       ├── lzop.py        # LZO compression handling
│       ├── main.py        # Main processing pipeline
│       ├── objects.py     # Data structures
│       ├── parser.py      # Data parser
│       └── transforms.py  # Data transformations
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd BSC
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Organization

### Required Data Structure

1. Sample Data (`data/samples/UKDCEUS*/`):
   - Raw ultrasound data files
   - Each case should have its own directory

2. ROI Data (`data/ROIs/`):
   - Excel files (.xlsx) containing ROI coordinates
   - File names should match sample directories

3. Phantom Data (`data/phantom/`):
   - Reference phantom data for normalization
   - Required for phantom-based normalization method

4. Results (`data/results/`):
   - Output directory for BSC calculations
   - Organized by case/sample name
   - Contains depth (cm), frequency (MHz), and power ratio data

## Usage

### 1. Data Unpacking

The data unpacking process extracts and processes raw ultrasound data from Clarius devices. This is the first step in the analysis pipeline.

```python
from scr.clarius.main import ClariusDataUnpacker

# Create an instance of the unpacker
unpacker = ClariusDataUnpacker()

# Unpack all samples in the data directory
# This will:
# 1. Clean up any existing extracted folders
# 2. Extract .tar files for each sample
# 3. Process and rename files based on device type
# 4. Generate necessary numpy arrays and delay samples
result = unpacker.unpack_data(path="data/samples")
```

The unpacker will:
- Extract `.tar` files containing raw ultrasound data
- Process RF (Radio Frequency) data files
- Generate and save processed numpy arrays
- Create delay sample information in Excel format
- Organize files in a structured format for BSC calculation

### 2. BSC Calculation

The BSC (Backscatter Coefficient) calculation processes the unpacked data using one of four normalization methods.

```python
from scr.clarius.main import BSC

# Initialize BSC with required parameters
bsc = BSC(
    # Data paths
    samples_folder_path="data/samples",    # Path to unpacked samples
    roi_folder_path="data/ROIs",          # Path to ROI Excel files
    result_folder_path="data/results",     # Where to save results
    
    # Analysis parameters
    normalization_method="normalized_with_phantom",  # Choose normalization method
    window="hann",     # STFT window type
    nperseg=64,       # Number of data points in STFT segment
    noverlap=32,      # Overlap between segments
    alpha=0.5         # Attenuation coefficient (dB/cm/MHz) for constant alpha method
)
```

The BSC calculation:
1. Loads unpacked sample data
2. Applies ROI selection from Excel files
3. Performs STFT (Short-Time Fourier Transform) analysis
4. Calculates power spectra and ratios
5. Saves results as CSV files:
   - `power_ratio.csv`: Spectral power ratios
   - `frequencies_MHz.csv`: Frequency values in MHz
   - `depths_cm.csv`: Depth measurements in cm

#### Normalization Methods:

1. **Phantom-based Normalization** (`normalized_with_phantom`):
   - Uses reference phantom data for normalization
   - Compensates for system-dependent effects
   - Requires phantom data in `data/phantom/` directory

2. **Healthy Liver Normalization** (`normalized_with_healthy_liver`):
   - Uses healthy liver tissue as reference
   - Normalizes against normal tissue regions

3. **Constant Alpha Correction** (`normalized_with_constant_alpha`):
   - Applies fixed attenuation coefficient (alpha)
   - Uses STFT-based attenuation compensation
   - Alpha specified in dB/cm/MHz
   - Requires `alpha` parameter in BSC initialization

4. **Calculated Alpha Correction** (`normalized_with_calculated_alpha`):
   - Automatically calculates attenuation coefficient from data
   - Uses central frequency (2.5 MHz) for alpha estimation
   - Applies frequency-dependent correction
