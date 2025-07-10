# Backscatter Coefficient (BSC) Analysis Tool

A Python-based tool for processing and analyzing ultrasound data, specifically focused on Backscatter Coefficient (BSC) analysis from Clarius ultrasound devices. This tool provides functionality for processing raw ultrasound data, ROI selection, and BSC calculation.

## Features

- Process raw ultrasound data from Clarius devices (C3 and L15 probes)
- Extract and handle RF (Radio Frequency) data
- Interactive ROI (Region of Interest) selection with GUI
- BSC calculation with multiple normalization methods:
  - Phantom-based normalization
  - Healthy liver tissue normalization
- Support for both large and small field-of-view acquisitions
- Automated data processing pipeline for multiple samples
- STFT (Short-Time Fourier Transform) based spectral analysis
- Comprehensive data visualization tools

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

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt:
  - numpy>=1.21.0: Numerical computations
  - pandas>=1.3.0: Data manipulation
  - pydicom>=2.3.0: DICOM file handling
  - matplotlib>=3.4.0: Data visualization
  - PyQt5>=5.15.0: GUI for ROI selection
  - scipy>=1.7.0: Signal processing
  - And others (see requirements.txt)

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
   - Supports both C3 and L15 probe data

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

### 1. Data Processing Pipeline

```python
from scr.clarius.main import BSC

# Initialize BSC with parameters
bsc = BSC(
    samples_folder_path="data/samples",  # Path to samples
    roi_folder_path="data/ROIs",        # Path to ROI files
    result_folder_path="data/results",   # Output path
    normalization_method="normalized_with_phantom",  # or "normalized_with_healthy_liver"
    window="hann",       # STFT window type
    nperseg=64,         # STFT segment length
    noverlap=32         # STFT overlap
)
```

### 2. ROI Selection

```python
from scr.app.app_roi_selection import ROI_selector_app
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
roi_selector = ROI_selector_app()
roi_selector.show()
sys.exit(app.exec_())
```

### 3. Jupyter Notebooks

The project includes two Jupyter notebooks:
- `BSC.ipynb`: For running BSC analysis
- `unpacker.ipynb`: For data extraction and preprocessing

## Key Parameters

### BSC Calculation
- `normalization_method`: Choose between phantom or healthy liver normalization
- `window`: STFT window type (e.g., "hann", "hamming")
- `nperseg`: Number of points per STFT segment
- `noverlap`: Number of overlapping points between segments

### Probe Parameters
- C3 Probe:
  - Sampling frequency: 15MHz
  - Frequency band: 1-6MHz
  - Center frequency: 2.5MHz

## Output Data

The BSC calculation produces three CSV files per sample:
1. `power_ratio.csv`: Power spectrum ratios
2. `frequencies.csv`: Frequency values
3. `depths.csv`: Depth values

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[Add your license information here]

## Support

For questions and support, please open an issue in the GitHub repository. 