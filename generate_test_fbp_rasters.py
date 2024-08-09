# -*- coding: utf-8 -*-
"""
Created on Thur Aug 8 17:30:00 2024

@author: Gregory A. Greene
"""

import os
import ProcessRasters as pr
from numpy import float64 as f64

# Get input folder
input_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Inputs')

# Get input dataset paths
fuel_type_path = os.path.join(input_folder, 'FuelType.tif')

# Get input data
fuel_type_ras = pr.getRaster(fuel_type_path)
fuel_type_profile = fuel_type_ras.profile
fuel_type_array = fuel_type_ras.read()

# Generate output dataset dictionary
data_dict = {
    'LAT': 62.245544,
    'LONG': -133.839203,
    'ELV': 1176,
    'GS': 7,
    'Aspect': 53,
    'WS': 24,
    'WD': 266,
    'FFMC': 92,
    'BUI': 31,
    'Dj': 137,
    'PC': 50,
    'PDF': 50,
    'GFL': 0.35,
    'cc': 80
}

for dset in list(data_dict.keys()):
    dset_array = fuel_type_array * 0 + data_dict.get(dset)
    # Save output datasets
    pr.arrayToRaster(array=dset_array,
                     out_file=os.path.join(input_folder, f'{dset}.tif'),
                     ras_profile=fuel_type_profile,
                     data_type=f64)
