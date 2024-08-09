# -*- coding: utf-8 -*-
"""
Created on Thur Aug 8 17:30:00 2024

@author: Gregory A. Greene
"""

import os
import ProcessRasters as pr
from numpy import float64 as f64
from datetime import datetime as dt
from typing import Union, Optional

# Get input folder
input_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Inputs')

# Get input dataset paths
fuel_type_path = os.path.join(input_folder, 'FuelType.tif')

# Get input data
fuel_type_ras = pr.getRaster(fuel_type_path)
fuel_type_profile = fuel_type_ras.profile
fuel_type_array = fuel_type_ras.read()


def gen_test_data(wx_date: int,
                  lat: Union[float, int],
                  long: Union[float, int],
                  elevation: Union[float, int],
                  slope: Union[float, int],
                  aspect: Union[float, int],
                  ws: Union[float, int],
                  wd: Union[float, int],
                  ffmc: Union[float, int],
                  bui: Union[float, int],
                  pc: Optional[Union[float, int]] = 50,
                  pdf: Optional[Union[float, int]] = 35,
                  gfl: Optional[Union[float, int]] = 0.35,
                  gcf: Optional[Union[float, int]] = 80):
    # Generate output dataset dictionary
    data_dict = {
        'Dj': dt.strptime(str(wx_date), '%Y%m%d%H').timetuple().tm_yday,
        'LAT': lat,
        'LONG': long,
        'ELV': elevation,
        'GS': slope,
        'Aspect': aspect,
        'WS': ws,
        'WD': wd,
        'FFMC': ffmc,
        'BUI': bui,
        'PC': pc,
        'PDF': pdf,
        'GFL': gfl,
        'cc': gcf
    }

    for dset in list(data_dict.keys()):
        dset_array = fuel_type_array * 0 + data_dict.get(dset)
        # Save output datasets
        pr.arrayToRaster(array=dset_array,
                         out_file=os.path.join(input_folder, f'{dset}.tif'),
                         ras_profile=fuel_type_profile,
                         data_type=f64)


if __name__ == '__main__':
    _wx_date = 20160516
    _lat = 62.245533
    _long = -133.840363
    _elevation = 1180
    _slope = 8
    _aspect = 60
    _ws = 24
    _wd = 266
    _ffmc = 92
    _bui = 31
    _pc = 0
    _pdf = 0
    _gfl = 0
    _gcf = 60

    # Generate FBP rasters
    gen_test_data(wx_date=_wx_date, lat=_lat, long=_long,
                  elevation=_elevation, slope=_slope, aspect=_aspect,
                  ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
                  pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf)
