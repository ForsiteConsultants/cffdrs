
# cffdrs Package

The `cffdrs` package provides tools for calculating Canadian Forest Fire Danger Rating System (CFFDRS) weather indices and 
fire behavior, and for generating test rasters for Fire Behavior Prediction (FBP) analysis. 

The package consists of three modules:

1. `cffbps.py`: Functions for calculating the Fire Behavior Prediction System (FBP) indices.
2. `cffwis.py`: Functions for calculating the Fire Weather Index System (FWI) indices.
3. `generate_test_fbp_rasters.py`: A script for generating test raster datasets for FBP calculations.

## Installation

1. Clone the repository or download the package files.
2. Ensure that you have Python 3.x installed.
3. Install required dependencies:
   ```bash
   pip install numpy
   ```

## Modules

### 1. `cffbps.py` - Fire Behavior Prediction System (FBP) Calculations

This module includes functions for calculating various FBP metrics, which help assess wildfire behavior under different conditions.

#### Functions:
- **`calcFC(fuel, wind, slope, aspect)`**: Calculates the Fire Crown potential based on fuel, wind, slope, and aspect.
- **`calcSP(sp_fmc, ffmc)`**: Computes Spread Potential based on the fine fuel moisture code and other factors.
- **`calcIC(intensity, slope)`**: Computes Initial Spread Index (ISI) based on intensity and slope.
  
#### Usage Example:
```python
from cffdrs.cffbps import calcFC, calcSP, calcIC

fc_value = calcFC(fuel="C1", wind=20, slope=15, aspect=180)
sp_value = calcSP(sp_fmc=85, ffmc=92)
ic_value = calcIC(intensity=30, slope=10)
```

### 2. `cffwis.py` - Fire Weather Index System (FWI) Calculations

This module contains functions to calculate FWI values based on environmental factors. FWI indices are used for wildfire risk assessment.

#### Functions:
- **`dailyDMC(dmc0, temp, precip, rh, month)`**: Calculates the Duff Moisture Code (DMC).
- **`dailyDC(dc0, temp, precip, month)`**: Calculates the Drought Code (DC).
- **`dailyISI(wind, ffmc)`**: Calculates the Initial Spread Index (ISI).
- **`dailyFWI(isi, bui)`**: Calculates the Fire Weather Index (FWI).
- **`dailyDSR(fwi)`**: Calculates the Daily Severity Rating (DSR).

#### Usage Example:
```python
from cffdrs.cffwis import dailyDMC, dailyDC, dailyISI, dailyFWI, dailyDSR

dmc_value = dailyDMC(dmc0=6, temp=20, precip=5, rh=40, month=6)
dc_value = dailyDC(dc0=15, temp=25, precip=2, month=7)
isi_value = dailyISI(wind=30, ffmc=85)
fwi_value = dailyFWI(isi=isi_value, bui=50)
dsr_value = dailyDSR(fwi=fwi_value)
```

### 3. `generate_test_fbp_rasters.py` - Generate Test Raster Data for FBP

This script generates test raster datasets for FBP analysis, useful for simulating fire behavior under different conditions. It requires an input folder with a `FuelType.tif` file.

#### Usage Example:
To generate test FBP rasters:
1. Set the parameters in the script (e.g., `wx_date`, `lat`, `long`, etc.).
2. Run the script:
   ```bash
   python generate_test_fbp_rasters.py
   ```

Example function call within the script:
```python
from cffdrs.generate_test_fbp_rasters import gen_test_data

gen_test_data(wx_date=20160516, lat=62.245533, long=-133.840363,
              elevation=1180, slope=8, aspect=60, ws=24, wd=266,
              ffmc=92, bui=31, pc=0, pdf=0, gfl=0, gcf=60)
```

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
