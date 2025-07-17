# Backscatter Coefficient (BSC) Analysis Tool

This project provides tools for processing and analyzing ultrasound data to calculate Backscatter Coefficients (BSC) using various normalization methods. It's specifically designed to work with Clarius ultrasound data and supports multiple processing approaches.

## Features

- **Data Unpacking**: Handles Clarius ultrasound data extraction from tar files
- **ROI Selection**: Supports Region of Interest (ROI) selection and processing with two size options:
  - Large ROI: Uses 'Large_ROI' sheet from Excel files
  - Small ROI: Uses 'Small_ROI' sheet from Excel files
- **Multiple Normalization Methods**:
  - Phantom-based normalization
  - Healthy liver tissue normalization
  - Constant alpha attenuation correction
  - Calculated alpha attenuation correction
- **3D Data Processing**: Handles full 3D ultrasound data (lines × samples × frames)

## Project Structure

```
BSC/
├── data/
│   ├── phantom/           # Reference phantom data
│   ├── results/           # Output results
│   ├── ROIs/             # ROI definition files
│   └── samples/          # Input ultrasound data
├── notebook/             # Jupyter notebooks for analysis
├── scr/                  # Source code
│   ├── app/             # Application code
│   └── clarius/         # Clarius data processing
└── requirements.txt      # Project dependencies
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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your ultrasound data in the `data/samples/` directory
2. Define ROIs in Excel files in the `data/ROIs/` directory:
   - Each Excel file must contain two sheets: 'Large_ROI' and 'Small_ROI'
   - Name Excel files to match sample directories
3. Run the BSC calculation using the provided notebook or Python script

### Example Code

```python
from scr.clarius.main import BSC

# Initialize BSC with required parameters
bsc = BSC(
    samples_folder_path="data/samples",
    roi_folder_path="data/ROIs",
    result_folder_path="data/results",
    normalization_method="normalized_with_constant_alpha",
    window="hann",
    nperseg=64,
    noverlap=32,
    alpha=0.5,  # dB/cm/MHz
    roi_size="large"  # Use "large" for Large_ROI or "small" for Small_ROI
)
```

### Available Normalization Methods

1. **Phantom Normalization** (`normalized_with_phantom`):
   - Uses reference phantom data for normalization
   - Requires phantom data in `data/phantom/` directory

2. **Healthy Liver Normalization** (`normalized_with_healthy_liver`):
   - Uses healthy liver tissue as reference
   - Requires ROI selection for both normal and abnormal tissue

3. **Constant Alpha Correction** (`normalized_with_constant_alpha`):
   - Applies attenuation correction using a fixed alpha value
   - Suitable when tissue properties are well-known

4. **Calculated Alpha Correction** (`normalized_with_calculated_alpha`):
   - Calculates attenuation coefficient from the data
   - Uses central frequency (2.5 MHz) for coefficient calculation

```