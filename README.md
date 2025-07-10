# Backscatter Coefficient (BSC) Analysis Tool

A Python-based tool for processing and analyzing ultrasound data, specifically focused on Backscatter Coefficient (BSC) analysis from Clarius ultrasound devices. This tool provides functionality for processing raw ultrasound data BSC calculation.

## Features

- Process raw ultrasound data from Clarius devices (C3)
- Extract and handle RF (Radio Frequency) data
- Interactive ROI (Region of Interest) selection with GUI
- BSC calculation with multiple normalization methods:
  - Phantom-based normalization
  - Healthy liver tissue normalization


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

The BSC (Backscatter Coefficient) calculation processes the unpacked data using either phantom-based or healthy liver tissue normalization.

```python
from scr.clarius.main import BSC

# Initialize BSC with required parameters
bsc = BSC(
    # Data paths
    samples_folder_path="data/samples",    # Path to unpacked samples
    roi_folder_path="data/ROIs",          # Path to ROI Excel files
    result_folder_path="data/results",     # Where to save results
    
    # Analysis parameters
    normalization_method="normalized_with_phantom",  # or "normalized_with_healthy_liver"
    window="hann",     # STFT window type
    nperseg=64,       # Number of data points in STFT segment
    noverlap=32       # Overlap between segments
)
```

The BSC calculation:
1. Loads unpacked sample data
2. Applies ROI selection from Excel files
3. Performs STFT (Short-Time Fourier Transform) analysis
4. Calculates power spectra and ratios
5. Saves results as CSV files:
   - `power_ratio.csv`: Spectral power ratios
   - `frequencies.csv`: Frequency values
   - `depths.csv`: Depth measurements

#### Normalization Methods:
- `normalized_with_phantom`: Uses phantom reference data
- `normalized_with_healthy_liver`: Uses healthy liver tissue as reference

#### Key Parameters:
- `window`: STFT window type (e.g., "hann")
- `nperseg`: Length of each STFT segment
- `noverlap`: Number of overlapping points between segments

## Output Data

The BSC calculation produces three CSV files per sample:
1. `power_ratio.csv`: Power spectrum ratios
2. `frequencies.csv`: Frequency values
3. `depths.csv`: Depth values

For questions, please open an issue in the GitHub repository. 