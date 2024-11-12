
# Canadian Forest Fire Danger Rating System (CFFDRS)

This project contains implementations for the Canadian Forest Fire Danger Rating System based on the 
Canadian Forest Fire Behavior Prediction System (CFFBPS) and Weather Index System (CFFWIS). 
The following instructions and function references help in utilizing the key scripts effectively.

## Repository Contents

- **cffbps.py**: Main module implementing the `FBP` class for fire behavior modeling.
- **cffwis.py**: Provides functions to calculate fire weather indices such as FFMC, DMC, DC, and ISI.
- **generate_test_fbp_rasters.py**: Script to generate test rasters based on user-provided parameters.

## Modules Overview

### 1. `cffbps.py` - Core Fire Behavior Prediction

#### Key Class: `FBP`
The `FBP` class provides various methods to model fire behavior based on specified parameters. Initialize this class with parameters such as fuel type, weather conditions, and fuel moisture codes.

##### Key Functions
- **`invertWindAspect()`**: Inverts the wind direction and aspect by 180°.
- **`calcSF()`**: Calculates the slope factor.
- **`calcISZ()`**: Calculates the initial spread index with no wind/no slope effects.
- **`calcFMC()`**: Computes the foliar moisture content (FMC) and foliar moisture effect (FME).
- **`calcROS()`**: Models the fire rate of spread.
- **`calcSFC()`**: Calculates forest floor consumption, woody fuel consumption, and total surface fuel consumption.
- **`getCBH_CFL()`**: Retrieves the default canopy base height (CBH) and canopy fuel load (CFL) for a specified fuel type.
- **`calcCSFI()`**: Calculates the critical surface fire intensity.
- **`calcRSO()`**: Computes the critical surface fire rate of spread.
- **`calcCFB()`**: Determines crown fraction burned.
- **`calcFireType()`**: Calculates the fire type (surface, intermittent crown, or active crown).
- **`calcCFC()`**: Determines the crown fuel consumed.
- **`calcC6hfros()`**: Calculates the crown and total head fire rate of spread for the C6 fuel type.
- **`calcTFC()`**: Computes the total fuel consumed.
- **`calcHFI()`**: Calculates the head fire intensity.
- **`runFBP()`**: Automatically runs the fire behavior model using all the methods above.

#### Usage Example
```python
from cffbps import FBP

# Initialize the FBP class with required parameters
fbp_instance = FBP(
    fuel_type=1, 
    wx_date=20240516, 
    lat=62.245533, 
    long=-133.840363, 
    elevation=1180,
    slope=8, 
    aspect=60, 
    ws=24, 
    wd=266, 
    ffmc=92, 
    bui=31, 
    pc=50, 
    pdf=35, 
    gfl=0.35, 
    gcf=80, 
    out_request=['fire_type', 'hfros', 'hfi']
)

# Run the fire behavior model and retrieve outputs
results = fbp_instance.runFBP()
```

### 2. `cffwis.py` - Fire Weather Index System

This module provides functions for calculating various weather indices in the Canadian Fire Weather Index (FWI) system, based on moisture and weather data.

#### Key Functions
- **`hourlyFFMC()`**: Computes hourly Fine Fuel Moisture Code (FFMC) values.
- **`dailyFFMC()`**: Calculates daily FFMC values.
- **`dailyDMC()`**: Determines the daily Drought Code (DMC).
- **`dailyDC()`**: Calculates the daily Drought Code (DC).
- **`dailyISI()`**: Calculates the Initial Spread Index (ISI).
- **`dailyBUI()`**: Computes the Build Up Index (BUI).
- **`dailyFWI()`**: Calculates the Fire Weather Index (FWI).
- **`dailyDSR()`**: Determines the Daily Severity Rating (DSR).
- **`startupDC()`**: Computes the Drought Code at the start of a season after overwintering.

#### Usage Example
```python
from cffwis import dailyFFMC, dailyISI, dailyBUI

# Calculate FFMC, ISI, and BUI based on weather inputs
ffmc = dailyFFMC(ffmc0=85, temp=15, rh=50, wind=10, precip=0)
isi = dailyISI(wind=10, ffmc=ffmc)
bui = dailyBUI(dmc=20, dc=30)
```

### 3. `generate_test_fbp_rasters.py` - Test Raster Data Generation

The `generate_test_fbp_rasters.py` script generates test raster data based on specified parameters. This is useful for validating and visualizing the model output in a spatial format.

#### Key Function
- **`gen_test_data()`**: Generates test raster files for various parameters, saved in the `Test_Data/Inputs` directory.

#### Usage Example
```python
from generate_test_fbp_rasters import gen_test_data

# Generate test raster data with specified parameters
gen_test_data(
    wx_date=20160516, 
    lat=62.245533, 
    long=-133.840363, 
    elevation=1180, 
    slope=8, 
    aspect=60, 
    ws=24, 
    wd=266, 
    ffmc=92, 
    bui=31
)
```

### Running Tests and Multiprocessing
You can test various functions in `cffbps.py` with the `_testFBP` function, which supports numeric, array, raster, and raster multiprocessing testing modes.

```python
from cffbps import _testFBP

# Test various modes
_testFBP(
    test_functions=['all'], 
    wx_date=20240516, 
    lat=62.245533, 
    long=-133.840363, 
    elevation=1180,
    slope=8, 
    aspect=60, 
    ws=24, 
    wd=266, 
    ffmc=92, 
    bui=31, 
    pc=50, 
    pdf=35, 
    gfl=0.35, 
    gcf=80, 
    out_request=['fire_type', 'hfros', 'hfi']
)
```

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
