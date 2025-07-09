# Backscatter Coefficient (BSC) Analysis

This project processes Clarius ultrasound data files to calculate and analyze Backscatter Coefficients. It handles RF data from Clarius ultrasound devices, specifically designed to work with C3 and L15 probes.

## Features

- Extract and process Clarius ultrasound data files (.tar format)
- Handle multiple probe types (C3 and L15) and FOV sizes (large and small)
- Process RF data with and without TGC (Time Gain Compensation)
- Save processed data as numpy arrays for further analysis
- ROI selection and analysis capabilities

## Project Structure

```
BSC/
├── data/
│   ├── ROIs/          # Excel files containing ROI information
│   └── samples/       # Raw ultrasound data samples
├── notebook/          # Jupyter notebooks for analysis
├── scr/
│   ├── analyze/       # Analysis modules
│   ├── app/           # GUI applications
│   └── clarius/       # Clarius data processing modules
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BSC.git
cd BSC
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Processing Ultrasound Data

1. Place your Clarius ultrasound data files in the `data/samples/` directory
2. Run the main processing script:
```python
from scr.clarius.main import process_sample
process_sample("path/to/sample/folder")
```

### ROI Selection

Use the ROI selection app to define regions of interest:
```python
from scr.app.app_roi_selection import run_app
run_app()
```

## Data Structure

The project expects Clarius ultrasound data in the following format:
- `.tar` files containing raw RF data
- Extracted folders containing:
  - RF data files (`*_rf.raw`)
  - Environment files (`*_env.raw`)
  - Configuration files (`*.yml`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license] 