# Ultrasound BSC Analysis

A Python-based tool for processing and analyzing ultrasound data, specifically focused on Backscatter Coefficient (BSC) analysis from Clarius ultrasound devices.

## Features

- Process raw ultrasound data from Clarius devices (C3 and L15 probes)
- Extract and handle RF (Radio Frequency) data
- Apply TGC (Time Gain Compensation) corrections
- ROI (Region of Interest) selection and analysis
- BSC calculation and analysis
- Support for both large and small field-of-view acquisitions

## Project Structure

```
BSC/
├── data/                    # Data directory (not included in repository)
│   ├── ROIs/               # Region of Interest data files
│   └── samples/            # Ultrasound sample data
│       └── UKDCEUS*/      # Individual case directories
├── notebook/               # Jupyter notebooks for analysis
│   └── parser.ipynb       # Data parsing notebook
├── scr/                    # Source code
│   ├── analyze/           # Analysis modules
│   │   └── bsc.py        # BSC calculation
│   ├── app/              # Application code
│   │   └── app_roi_selection.py  # ROI selection interface
│   └── clarius/          # Clarius data handling
│       ├── lzop.py       # LZO compression handling
│       ├── main.py       # Main processing pipeline
│       ├── objects.py    # Data objects
│       ├── parser.py     # Data parser
│       └── transforms.py # Data transformations
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd BSC
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\\Scripts\\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare data directory:
The `data/` directory structure is maintained in the repository, but you'll need to add your own data files:
- Place ROI Excel files in `data/ROIs/`
- Place ultrasound sample data in `data/samples/UKDCEUS*/`

## Data Structure

### Sample Data Organization
Each case directory (UKDCEUS*) contains:
- DICOM files (*.dcm)
- Raw data archives (raw_*.tar)
- Extracted data folders containing:
  - RF data files (*.raw)
  - Envelope data (*.env.raw)
  - Configuration files (*.yml)
  - TGC data (*.tgc.yml)
  - Processed NumPy arrays (*.npy)

### Supported Probes
- C3 (Curvilinear): Large and small field of view
- L15 (Linear): Large and small field of view

## Usage

1. **Data Processing Pipeline**
```python
from scr.clarius.main import unpack_clarius_data

# Process a single sample
unpacker = unpack_clarius_data("data/samples/UKDCEUS030", extraction_mode="single_sample")

# Process multiple samples
unpacker = unpack_clarius_data("data/samples", extraction_mode="multiple_samples")
```

2. **ROI Selection**
```python
from scr.app.app_roi_selection import run_roi_selection

run_roi_selection()
```

3. **BSC Analysis**
```python
from scr.analyze.bsc import calculate_bsc

# Calculate BSC for selected ROI
bsc_results = calculate_bsc(data_path, roi_path)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[Add your license information here]

## Contact

[Add your contact information here] 