# -*- coding: utf-8 -*-
"""
cffpbs.py modified to use CuPy for GPU acceleration.
Created on Thur Dec 26 20:45:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene']

import os
from typing import Union, Optional, Literal
from operator import itemgetter
import numpy as np
import cupy as cp
import rasterio as rio
from datetime import datetime as dt

import ProcessRasters

# Define lookup tables as in the original code
fbpFTCode_NumToAlpha_LUT = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
    7: 'C7', 8: 'D1', 9: 'D2', 10: 'M1', 11: 'M2', 12: 'M3',
    13: 'M4', 14: 'O1a', 15: 'O1b', 16: 'S1', 17: 'S2', 18: 'S3',
    19: 'NF', 20: 'WA'
}

fbpFTCode_AlphaToNum_LUT = {
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6,
    'C7': 7, 'D1': 8, 'D2': 9, 'M1': 10, 'M2': 11, 'M3': 12,
    'M4': 13, 'O1a': 14, 'O1b': 15, 'S1': 16, 'S2': 17, 'S3': 18,
    'NF': 19, 'WA': 20
}


def convert_grid_codes(fuel_type_array: cp.ndarray) -> cp.ndarray:
    """
    Convert grid code values from the cffdrs R package fuel type codes
    to the codes used in this module.

    :param fuel_type_array: CFFBPS fuel type array (CuPy ndarray)
    :return: Converted CuPy ndarray with remapped fuel type codes
    """
    return cp.where(fuel_type_array == 19, 20,
                    cp.where(fuel_type_array == 13, 19,
                             cp.where(fuel_type_array == 12, 13,
                                      cp.where(fuel_type_array == 11, 12,
                                               cp.where(fuel_type_array == 10, 11,
                                                        cp.where(fuel_type_array == 9, 10,
                                                                 fuel_type_array)))))).astype(cp.int8)


def getSeasonGrassCuring(season: str,
                         province: str,
                         subregion: str = None) -> int:
    """
    Function returns a default grass curing code based on season, province, and subregion
    :param season: annual season ("spring", "summer", "fall", "winter")
    :param province: province being assessed ("AB", "BC")
    :param subregion: British Columbia subregions ("southeast", "other")
    :return: grass curing percent (%); "None" is returned if season or province are invalid
    """
    if province == 'AB':
        # Default seasonal grass curing values for Alberta
        # These curing rates are recommended by Neal McLoughlin to align with Alberta Wildfire knowledge and practices
        gc_dict = {
            'spring': 75,
            'summer': 40,
            'fall': 60,
            'winter': 100
        }
    elif province == 'BC':
        if subregion == 'southeast':
            # Default seasonal grass curing values for southeastern British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 90,
                'fall': 90,
                'winter': 100
            }
        else:
            # Default seasonal grass curing values for British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 60,
                'fall': 85,
                'winter': 100
            }
    else:
        gc_dict = {}

    return gc_dict.get(season.lower(), None)


# Replace NumPy arrays with CuPy arrays within the FBP class and associated methods
class FBP:
    """
    Class to model fire behavior with the Canadian Forest Fire Behavior Prediction System.
    Modified to use CuPy for GPU acceleration.
    """

    def __init__(self):
        # Initialize CFFBPS input parameters
        self.fuel_type = cp.array([0], dtype=cp.float32)
        self.wx_date = cp.array([0], dtype=cp.float32)
        self.lat = cp.array([0], dtype=cp.float32)
        self.long = cp.array([0], dtype=cp.float32)
        self.elevation = cp.array([0], dtype=cp.float32)
        self.slope = cp.array([0], dtype=cp.float32)
        self.aspect = cp.array([0], dtype=cp.float32)
        self.ws = cp.array([0], dtype=cp.float32)
        self.wd = cp.array([0], dtype=cp.float32)
        self.ffmc = cp.array([0], dtype=cp.float32)
        self.bui = cp.array([0], dtype=cp.float32)
        self.pc = cp.array([0], dtype=cp.float32)
        self.pdf = cp.array([0], dtype=cp.float32)
        self.gfl = cp.array([0], dtype=cp.float32)
        self.gcf = cp.array([0], dtype=cp.float32)
        self.out_request = cp.array([0], dtype=cp.float32)
        self.convert_fuel_type_codes = cp.array([0], dtype=cp.float32)
        self.return_array_as = cp.array([0], dtype=cp.float32)

        # Internal tracking
        self.return_array = cp.array([0], dtype=cp.float32)
        self.ref_array = cp.array([0], dtype=cp.float32)
        self.initialized = False

        # Initialize multiprocessing block variable
        self.block = cp.array([0], dtype=cp.float32)

        # Initialize unique fuel types list
        self.ftypes = cp.array([0], dtype=cp.float32)

        # Initialize weather parameters
        self.isi = cp.array([0], dtype=cp.float32)
        self.m = cp.array([0], dtype=cp.float32)
        self.fF = cp.array([0], dtype=cp.float32)
        self.fW = cp.array([0], dtype=cp.float32)

        # Initialize slope effect parameters
        self.a = cp.array([0], dtype=cp.float32)
        self.b = cp.array([0], dtype=cp.float32)
        self.c = cp.array([0], dtype=cp.float32)
        self.rsz = cp.array([0], dtype=cp.float32)
        self.isz = cp.array([0], dtype=cp.float32)
        self.sf = cp.array([0], dtype=cp.float32)
        self.rsf = cp.array([0], dtype=cp.float32)
        self.isf = cp.array([0], dtype=cp.float32)
        self.rsi = cp.array([0], dtype=cp.float32)
        self.wse1 = cp.array([0], dtype=cp.float32)
        self.wse2 = cp.array([0], dtype=cp.float32)
        self.wse = cp.array([0], dtype=cp.float32)
        self.wsx = cp.array([0], dtype=cp.float32)
        self.wsy = cp.array([0], dtype=cp.float32)
        self.wsv = cp.array([0], dtype=cp.float32)
        self.raz = cp.array([0], dtype=cp.float32)

        # Initialize BUI effect parameters
        self.q = cp.array([0], dtype=cp.float32)
        self.bui0 = cp.array([0], dtype=cp.float32)
        self.be = cp.array([0], dtype=cp.float32)
        self.be_max = cp.array([0], dtype=cp.float32)

        # Initialize surface parameters
        self.cf = cp.array([0], dtype=cp.float32)
        self.ffc = cp.array([0], dtype=cp.float32)
        self.wfc = cp.array([0], dtype=cp.float32)
        self.sfc = cp.array([0], dtype=cp.float32)
        self.rss = cp.array([0], dtype=cp.float32)

        # Initialize foliar moisture content parameters
        self.latn = cp.array([0], dtype=cp.float32)
        self.dj = cp.array([0], dtype=cp.float32)
        self.d0 = cp.array([0], dtype=cp.float32)
        self.nd = cp.array([0], dtype=cp.float32)
        self.fmc = cp.array([0], dtype=cp.float32)
        self.fme = cp.array([0], dtype=cp.float32)

        # Initialize crown and total fuel consumed parameters
        self.cbh = cp.array([0], dtype=cp.float32)
        self.csfi = cp.array([0], dtype=cp.float32)
        self.rso = cp.array([0], dtype=cp.float32)
        self.rsc = cp.array([0], dtype=cp.float32)
        self.cfb = cp.array([0], dtype=cp.float32)
        self.cfl = cp.array([0], dtype=cp.float32)
        self.cfc = cp.array([0], dtype=cp.float32)
        self.tfc = cp.array([0], dtype=cp.float32)

        # Initialize the back fire rate of spread parameters
        self.bfW = cp.array([0], dtype=cp.float32)
        self.brsi = cp.array([0], dtype=cp.float32)
        self.bisi = cp.array([0], dtype=cp.float32)
        self.bros = cp.array([0], dtype=cp.float32)

        # Initialize default CFFBPS output parameters
        self.fire_type = cp.array([0], dtype=cp.int8)
        self.hfros = cp.array([0], dtype=cp.float32)
        self.hfi = cp.array([0], dtype=cp.float32)

        # Initialize C-6 rate of spread parameters
        self.sfros = cp.array([0], dtype=cp.float32)
        self.cfros = cp.array([0], dtype=cp.float32)

        # Initialize point ignition acceleration parameter
        self.accel_param = cp.array([0], dtype=cp.float32)

        # ### Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = ['cbh', 'fmc']
        self.rsoVarList = ['csfi', 'sfc']
        self.cfbVarList = ['cfros', 'rso']
        self.cfcVarList = ['cfb', 'cfl']
        self.cfiVarList = ['cfros', 'cfc']

        # Array of open fuel type codes
        self.open_fuel_types = cp.array([1, 7, 9, 14, 15, 16, 17, 18])

        # Array of non-crowning fuel type codes
        self.non_crowning_fuels = cp.array([8, 9, 14, 15, 16, 17, 18])

        # CFFBPS Canopy Base Height & Canopy Fuel Load Lookup Table (cbh, cfl, ht)
        self.fbpCBH_CFL_HT_LUT = {
            1: (2, 0.75, 10),
            2: (3, 0.8, 7),
            3: (8, 1.15, 18),
            4: (4, 1.2, 10),
            5: (18, 1.2, 25),
            6: (7, 1.8, 14),
            7: (10, 0.5, 20),
            8: (0, 0, 0),
            9: (0, 0, 0),
            10: (6, 0.8, 13),
            11: (6, 0.8, 13),
            12: (6, 0.8, 8),
            13: (6, 0.8, 8),
            14: (0, 0, 0),
            15: (0, 0, 0),
            16: (0, 0, 0),
            17: (0, 0, 0),
            18: (0, 0, 0)
        }

        # CFFBPS Surface Fire Rate of Spread Parameters (a, b, c, q, BUI0, be_max)
        self.rosParams = {
            1: (90, 0.0649, 4.5, 0.9, 72, 1.076),
            2: (110, 0.0282, 1.5, 0.7, 64, 1.321),
            3: (110, 0.0444, 3, 0.75, 62, 1.261),
            4: (110, 0.0293, 1.5, 0.8, 66, 1.184),
            5: (30, 0.0697, 4, 0.8, 56, 1.220),
            6: (30, 0.08, 3, 0.8, 62, 1.197),
            7: (45, 0.0305, 2, 0.85, 106, 1.134),
            8: (30, 0.0232, 1.6, 0.9, 32, 1.179),
            9: (30, 0.0232, 1.6, 0.9, 32, 1.179),
            10: (None, None, None, None, 0.8, 50, 1.250),
            11: (None, None, None, None, 0.8, 50, 1.250),
            12: (120, 0.0572, 1.4, 0.8, 50, 1.250),
            13: (100, 0.0404, 1.48, 0.8, 50, 1.250),
            14: (190, 0.0310, 1.4, 1, None, 1),
            15: (250, 0.0350, 1.7, 1, None, 1),
            16: (75, 0.0297, 1.3, 0.75, 38, 1.460),
            17: (40, 0.0438, 1.7, 0.75, 63, 1.256),
            18: (55, 0.0829, 3.2, 0.75, 31, 1.590)
        }

        return

    def _checkArray(self) -> None:
        """
        Check if any of the core input parameters are CuPy arrays and standardize reference shapes.

        :return: None
        """
        input_list = [
            self.fuel_type, self.lat, self.long,
            self.elevation, self.slope, self.aspect,
            self.ws, self.wd, self.ffmc, self.bui,
            self.pc, self.pdf, self.gfl, self.gcf
        ]

        if any(isinstance(data, cp.ndarray) for data in input_list):
            self.return_array = True
            array_indices = [i for i, data in enumerate(input_list) if isinstance(data, cp.ndarray)]
            arrays = itemgetter(*array_indices)(input_list)
            if isinstance(arrays, cp.ndarray):
                arrays = [arrays]
            shapes = {arr.shape for arr in arrays}
            if len(shapes) > 1:
                raise ValueError(f'Array dimensions mismatch: {shapes}')
            first_array = arrays[0]
            mask = cp.isnan(first_array)
            self.ref_array = cp.where(mask,
                                      cp.full_like(first_array, cp.nan, dtype=cp.float32),
                                      cp.zeros_like(first_array, dtype=cp.float32))
        else:
            self.return_array = False
            val = self.fuel_type
            if isinstance(val, str):
                val = fbpFTCode_AlphaToNum_LUT.get(val)
            elif isinstance(val, np.ndarray) and val.dtype.kind in ('U', 'S'):
                val = np.vectorize(fbpFTCode_AlphaToNum_LUT.get)(val).astype(np.uint16)
            mask = cp.isnan(cp.array([val], dtype=cp.float32))
            self.ref_array = cp.where(mask,
                                      cp.full_like(mask, cp.nan, dtype=cp.float32),
                                      cp.array([val], dtype=cp.float32))
        return

    def _verifyInputs(self) -> None:
        """
        Validate and convert all input parameters into CuPy arrays where needed.

        :return: None
        """
        if isinstance(self.fuel_type, str):
            self.fuel_type = cp.asarray([fbpFTCode_AlphaToNum_LUT.get(self.fuel_type)], dtype=cp.float32)
        elif isinstance(self.fuel_type, np.ndarray) and self.fuel_type.dtype.kind in ('U', 'S'):
            convert = np.vectorize(fbpFTCode_AlphaToNum_LUT.get)
            self.fuel_type = cp.asarray(convert(self.fuel_type), dtype=cp.int8)
        elif isinstance(self.fuel_type, cp.ndarray) and self.fuel_type.dtype.kind in ('U', 'S'):
            convert = cp.vectorize(fbpFTCode_AlphaToNum_LUT.get)
            self.fuel_type = convert(self.fuel_type).astype(cp.int8)
        elif not isinstance(self.fuel_type, cp.ndarray):
            self.fuel_type = cp.asarray([self.fuel_type], dtype=cp.float32)

        if self.convert_fuel_type_codes:
            self.fuel_type = convert_grid_codes(self.fuel_type)

        self.ftypes = cp.array([int(ft) for ft in cp.asnumpy(cp.unique(self.fuel_type)) if ft in self.rosParams])

        try:
            dt.strptime(str(self.wx_date), '%Y%m%d')
        except Exception:
            raise ValueError('wx_date must be in YYYYMMDD format')

        for attr in ['lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui']:
            val = getattr(self, attr)
            setattr(self, attr, cp.asarray(val, dtype=cp.float32))

        for attr, default in [('pc', 50), ('pdf', 35), ('gfl', 0.35), ('gcf', 80)]:
            val = getattr(self, attr)
            if val is None:
                val = default
            setattr(self, attr, cp.asarray(val, dtype=cp.float32))

        return

    def _init_array(self, fill_value=0, dtype=cp.float32) -> cp.ndarray:
        return cp.full(self.ref_array.shape, fill_value, dtype=dtype)

    def initialize(self,
                   fuel_type: Union[int, str, cp.ndarray] = None,
                   wx_date: int = None,
                   lat: Union[float, int, cp.ndarray] = None,
                   long: Union[float, int, cp.ndarray] = None,
                   elevation: Union[float, int, cp.ndarray] = None,
                   slope: Union[float, int, cp.ndarray] = None,
                   aspect: Union[float, int, cp.ndarray] = None,
                   ws: Union[float, int, cp.ndarray] = None,
                   wd: Union[float, int, cp.ndarray] = None,
                   ffmc: Union[float, int, cp.ndarray] = None,
                   bui: Union[float, int, cp.ndarray] = None,
                   pc: Optional[Union[float, int, cp.ndarray]] = 50,
                   pdf: Optional[Union[float, int, cp.ndarray]] = 35,
                   gfl: Optional[Union[float, int, cp.ndarray]] = 0.35,
                   gcf: Optional[Union[float, int, cp.ndarray]] = 80,
                   out_request: Optional[Union[list, tuple]] = None,
                   convert_fuel_type_codes: Optional[bool] = False,
                   return_array_as: Literal['numpy', 'cupy'] = 'numpy') -> None:
        """
        Initialize the FBP object with the provided parameters.

        :param fuel_type: CFFBPS fuel type (numeric code: 1-20)
            Model 1: C-1 fuel type ROS model
            Model 2: C-2 fuel type ROS model
            Model 3: C-3 fuel type ROS model
            Model 4: C-4 fuel type ROS model
            Model 5: C-5 fuel type ROS model
            Model 6: C-6 fuel type ROS model
            Model 7: C-7 fuel type ROS model
            Model 8: D-1 fuel type ROS model
            Model 9: D-2 fuel type ROS model
            Model 10: M-1 fuel type ROS model
            Model 11: M-2 fuel type ROS model
            Model 12: M-3 fuel type ROS model
            Model 13: M-4 fuel type ROS model
            Model 14: O-1a fuel type ROS model
            Model 15: O-1b fuel type ROS model
            Model 16: S-1 fuel type ROS model
            Model 17: S-2 fuel type ROS model
            Model 18: S-3 fuel type ROS model
            Model 19: Non-fuel (NF)
            Model 20: Water (WA)
        :param wx_date: Date of weather observation (used for fmc calculation) (YYYYMMDD)
        :param lat: Latitude of area being modelled (Decimal Degrees, floating point)
        :param long: Longitude of area being modelled (Decimal Degrees, floating point)
        :param elevation: Elevation of area being modelled (m)
        :param slope: Ground slope angle/tilt of area being modelled (%)
        :param aspect: Ground slope aspect/azimuth of area being modelled (degrees)
        :param ws: Wind speed (km/h @ 10m height)
        :param wd: Wind direction (degrees, direction wind is coming from)
        :param ffmc: CFFWIS Fine Fuel Moisture Code
        :param bui: CFFWIS Buildup Index
        :param pc: Percent conifer (%, value from 0-100)
        :param pdf: Percent dead fir (%, value from 0-100)
        :param gfl: Grass fuel load (kg/m^2)
        :param gcf: Grass curing factor (%, value from 0-100)
        :param out_request: Tuple or list of CFFBPS output variables
            # Default output variables
            fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
            hfros = Head fire rate of spread (m/min)
            hfi = head fire intensity (kW/m)

            # Weather variables
            ws = Observed wind speed (km/h)
            wd = Wind azimuth/direction (degrees)
            m = Moisture content equivalent of the FFMC (%, value from 0-100+)
            fF = Fine fuel moisture function in the ISI
            fW = Wind function in the ISI
            isi = Final ISI, accounting for wind and slope

            # Slope + wind effect variables
            a = Rate of spread equation coefficient
            b = Rate of spread equation coefficient
            c = Rate of spread equation coefficient
            RSZ = Surface spread rate with zero wind on level terrain
            SF = Slope factor
            RSF = spread rate with zero wind, upslope
            ISF = ISI, with zero wind upslope
            RSI = Initial spread rate without BUI effect
            WSE1 = Original slope equivalent wind speed value
            WSE2 = New slope equivalent sind speed value for cases where WSE1 > 40 (capped at max of 112.45)
            WSE = Slope equivalent wind speed
            WSX = Net vectorized wind speed in the x-direction
            WSY = Net vectorized wind speed in the y-direction
            WSV = (aka: slope-adjusted wind speed) Net vectorized wind speed (km/h)
            RAZ = (aka: slope-adjusted wind direction) Net vectorized wind direction (degrees)

            # BUI effect variables
            q = Proportion of maximum rate of spread at BUI equal to 50
            bui0 = Average BUI for each fuel type
            BE = Buildup effect on spread rate
            be_max = Maximum allowable BE value

            # Surface fuel variables
            ffc = Estimated forest floor consumption
            wfc = Estimated woody fuel consumption
            sfc = Estimated total surface fuel consumption

            # Foliar moisture content variables
            latn = Normalized latitude
            d0 = Julian date of minimum foliar moisture content
            nd = number of days between modelled fire date and d0
            fmc = foliar moisture content
            fme = foliar moisture effect

            # Critical crown fire threshold variables
            csfi = critical intensity (kW/m)
            rso = critical rate of spread (m/min)

            # Crown fuel parameters
            cbh = Height to live crown base (m)
            cfb = Crown fraction burned (proportion, value ranging from 0-1)
            cfl = Crown fuel load (kg/m^2)
            cfc = Crown fuel consumed
        :param convert_fuel_type_codes: Convert from CFS cffdrs R fuel type grid codes
            to the grid codes used in this module
        :param return_array_as: If the results are arrays, the type of array to return as. Options: 'numpy', 'cupy'
        """
        self.fuel_type = fuel_type
        self.wx_date = wx_date
        self.lat = lat
        self.long = long
        self.elevation = elevation
        self.slope = slope
        self.aspect = aspect
        self.ws = ws
        self.wd = wd
        self.ffmc = ffmc
        self.bui = bui
        self.pc = pc
        self.pdf = pdf
        self.gfl = gfl
        self.gcf = gcf
        self.out_request = out_request
        self.convert_fuel_type_codes = convert_fuel_type_codes
        self.return_array_as = return_array_as

        self._checkArray()
        self._verifyInputs()

        # Initialize all model parameter arrays
        self.isi = self._init_array()
        self.m = self._init_array()
        self.fF = self._init_array()
        self.fW = self._init_array()
        self.a = self._init_array()
        self.b = self._init_array()
        self.c = self._init_array()
        self.rsz = self._init_array()
        self.isz = self._init_array()
        self.sf = self._init_array()
        self.rsf = self._init_array()
        self.isf = self._init_array()
        self.rsi = self._init_array()
        self.wse1 = self._init_array()
        self.wse2 = self._init_array()
        self.wse = self._init_array()
        self.wsx = self._init_array()
        self.wsy = self._init_array()
        self.wsv = self._init_array()
        self.raz = self._init_array()
        self.q = self._init_array()
        self.bui0 = self._init_array()
        self.be = self._init_array()
        self.be_max = self._init_array()
        self.cf = self._init_array()
        self.ffc = self._init_array(cp.nan)
        self.wfc = self._init_array(cp.nan)
        self.sfc = self._init_array(cp.nan)
        self.rss = self._init_array()
        self.latn = self._init_array()
        self.dj = self._init_array()
        self.d0 = self._init_array()
        self.nd = self._init_array()
        self.fmc = self._init_array()
        self.fme = self._init_array()
        self.cbh = self._init_array()
        self.csfi = self._init_array()
        self.rso = self._init_array()
        self.rsc = self._init_array()
        self.cfb = self._init_array()
        self.cfl = self._init_array()
        self.cfc = self._init_array()
        self.tfc = self._init_array()
        self.bfW = self._init_array()
        self.brsi = self._init_array()
        self.bisi = self._init_array()
        self.bros = self._init_array()
        self.fire_type = self._init_array(0, dtype=cp.int8)
        self.hfros = self._init_array()
        self.hfi = self._init_array()
        self.sfros = self._init_array()
        self.cfros = self._init_array()
        self.accel_param = self._init_array()

        # List of required parameters
        required_params = [
            'fuel_type', 'wx_date', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui'
        ]

        # Check for missing required parameters
        missing_params = [param for param in required_params if getattr(self, param) is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Set initialized to True
        self.initialized = True
        return

    def invertWindAspect(self):
        """
        Function to invert/flip wind direction and aspect by 180 degrees using CuPy
        :return: None
        """
        # Invert wind direction by 180 degrees (i.e., direction wind is heading to)
        self.wd = cp.where(self.wd > 180, self.wd - 180, self.wd + 180)

        # Invert aspect by 180 degrees (i.e., up slope direction)
        self.aspect = cp.where(self.aspect > 180, self.aspect - 180, self.aspect + 180)

        return

    def calcSF(self) -> None:
        """
        Function to calculate the slope factor using CuPy
        :return: None
        """
        # Calculate the slope factor with CuPy
        self.sf = cp.where(
            self.slope < 70,
            cp.exp(3.533 * cp.power((self.slope / 100), 1.2)),
            cp.full_like(self.slope, 10, dtype=cp.int8)
        )
        return

    def calcISZ(self) -> None:
        """
        Function to calculate the initial spread index with no wind/no slope effects.
        """
        self.m = (250 * (59.5 / 101) * (101 - self.ffmc)) / (59.5 + self.ffmc)
        self.fF = (91.9 * cp.exp(-0.1386 * self.m)) * (1 + (cp.power(self.m, 5.31) / (4.93 * cp.power(10, 7))))
        self.isz = 0.208 * self.fF

    def calcFMC(self,
                lat: Optional[float] = None,
                long: Optional[float] = None,
                elevation: Optional[float] = None,
                wx_date: Optional[int] = None) -> None:
        """
        Function to calculate foliar moisture content (FMC) and foliar moisture effect (FME) using CuPy.

        :return: None
        """
        if lat is not None:
            self.lat = cp.asarray(lat, dtype=cp.float32)
        if long is not None:
            self.long = cp.asarray(long, dtype=cp.float32)
        if elevation is not None:
            self.elevation = cp.asarray(elevation, dtype=cp.float32)
        if wx_date is not None:
            self.wx_date = wx_date

            # Normalize latitude
        abs_long = cp.abs(self.long)
        use_elev = (self.elevation > 0)
        self.latn = cp.where(
            use_elev,
            43 + (33.7 * cp.exp(-0.0351 * (150 - abs_long))),
            46 + (23.4 * cp.exp(-0.036 * (150 - abs_long)))
        )

        # Julian date
        dj_value = dt.strptime(str(self.wx_date), '%Y%m%d').timetuple().tm_yday
        self.dj = cp.full_like(self.latn, dj_value, dtype=cp.int16)

        # D0 calculation
        self.d0 = cp.round(cp.where(
            use_elev,
            142.1 * (self.lat / self.latn) + (0.0172 * self.elevation),
            151 * (self.lat / self.latn)
        ), 0)

        # ND (difference between dates)
        self.nd = cp.abs(self.dj - self.d0)

        # FMC calculation with fused condition
        nd_lt_30 = self.nd < 30
        nd_lt_50 = (self.nd >= 30) & (self.nd < 50)

        self.fmc = cp.where(
            nd_lt_30,
            85 + (0.0189 * self.nd ** 2),
            cp.where(
                nd_lt_50,
                32.9 + (3.17 * self.nd) - (0.0288 * self.nd ** 2),
                cp.full_like(self.nd, 120, dtype=cp.float32)
            )
        )

        # FME calculation
        self.fme = 1000 * cp.power(1.5 - (0.00275 * self.fmc), 4) / (460 + (25.9 * self.fmc))
        return

    def _calcISI_slopeWind(self, ftype: int) -> None:
        """
        Calculate slope-equivalent wind speed and wind-adjusted ISI components.
        Shared logic used across most fuel types.
        """
        # Slope equivalent wind speed (WSE1)
        self.wse1 = cp.where(
            self.fuel_type == ftype,
            (1 / 0.05039) * cp.log(self.isf / (0.208 * self.fF)),
            self.wse1
        )

        # Alternate slope equivalent wind speed (WSE2)
        self.wse2 = cp.where(
            self.fuel_type == ftype,
            cp.where(
                self.isf < (0.999 * 2.496 * self.fF),
                28 - (1 / 0.0818) * cp.log(1 - (self.isf / (2.496 * self.fF))),
                112.45
            ),
            self.wse2
        )

        # Final slope equivalent wind speed (WSE)
        self.wse = cp.where(
            self.fuel_type == ftype,
            cp.where(self.wse1 <= 40, self.wse1, self.wse2),
            self.wse
        )

        # Combined slope + wind vector
        self.wsx = cp.where(
            self.fuel_type == ftype,
            (self.ws * cp.sin(cp.radians(self.wd))) + (self.wse * cp.sin(cp.radians(self.aspect))),
            self.wsx
        )
        self.wsy = cp.where(
            self.fuel_type == ftype,
            (self.ws * cp.cos(cp.radians(self.wd))) + (self.wse * cp.cos(cp.radians(self.aspect))),
            self.wsy
        )
        self.wsv = cp.where(
            self.fuel_type == ftype,
            cp.sqrt(self.wsx ** 2 + self.wsy ** 2),
            self.wsv
        )

        # Resultant wind direction (RAZ)
        arccos_val = cp.degrees(cp.arccos(cp.clip(self.wsy / cp.maximum(self.wsv, 1e-6), -1, 1)))
        self.raz = cp.where(
            self.fuel_type == ftype,
            cp.where(self.wsx < 0, 360 - arccos_val, arccos_val),
            self.raz
        )

        # Wind function for ISI
        self.fW = cp.where(
            self.fuel_type == ftype,
            cp.where(
                self.wsv > 40,
                12 * (1 - cp.exp(-0.0818 * (self.wsv - 28))),
                cp.exp(0.05039 * self.wsv)
            ),
            self.fW
        )

        # Final ISI
        self.isi = cp.where(
            self.fuel_type == ftype,
            0.208 * self.fW * self.fF,
            self.isi
        )

        # Backing fire wind effect
        self.bfW = cp.where(
            self.fuel_type == ftype,
            cp.exp(-0.05039 * self.wsv),
            self.bfW
        )

        self.bisi = self.bfW * self.fF * 0.208
        return

    def calcISI_RSI_BE(self, ftype: int) -> None:
        """
        Function to calculate the slope-/wind-adjusted Initial Spread Index (ISI),
        rate of spread (RSI), and the BUI buildup effect (BE) using CuPy.

        :param ftype: The numeric FBP fuel type code.
        :return: None
        """
        # Ensure all self attributes are CuPy arrays
        self.fuel_type = cp.asarray(self.fuel_type)
        self.isf = cp.asarray(self.isf)
        self.fF = cp.asarray(self.fF)
        self.ws = cp.asarray(self.ws)
        self.wd = cp.asarray(self.wd)
        self.aspect = cp.asarray(self.aspect)

        # ### CFFBPS ROS models
        if ftype not in [10, 11]:
            # Get fuel type-specific fixed rate of spread parameters
            self.a, self.b, self.c, self.q, self.bui0, self.be_max = self.rosParams[ftype]

            if ftype in [14, 15]:
                # ## Process O-1a/b fuel types...
                # Grass curing coefficient
                self.cf = cp.where(
                    self.fuel_type == ftype,
                    cp.where(
                        self.gcf < 58.8,
                        0.005 * (cp.exp(0.061 * self.gcf) - 1),
                        0.176 + (0.02 * (self.gcf - 58.8))
                    ),
                    self.cf
                )

                # No-slope/no-wind ROS
                self.rsz = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.isz), self.c) * self.cf,
                    self.rsz
                )

                # Slope-adjusted ROS
                self.rsf = cp.where(
                    self.fuel_type == ftype,
                    self.rsz * self.sf,
                    self.rsf
                )

                # Slope-influenced ISI estimate
                isf_numer = 1 - cp.power(self.rsf / (self.cf * self.a), 1 / self.c)
                isf_safe = cp.where(isf_numer >= 0.01, isf_numer, 0.01)
                self.isf = cp.where(
                    self.fuel_type == ftype,
                    cp.log(isf_safe) / -self.b,
                    self.isf
                )

                # Wind and ISI adjustments
                self._calcISI_slopeWind(ftype)

                # Final head fire ROS
                self.rsi = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.isi), self.c) * self.cf,
                    self.rsi
                )

                # Backing fire ROS
                self.brsi = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.bisi), self.c) * self.cf,
                    self.brsi
                )

                # BE = 1 for grass fuels
                self.be = cp.where(self.fuel_type == ftype, 1.0, self.be)

            elif ftype in [12, 13]:
                # ## Process M-3/4 fuel types...
                lu_params = self.rosParams[8] if ftype == 12 else self.rosParams[9]
                a_d1, b_d1, c_d1, *_ = lu_params

                # D1 component ROS and ISF
                rsz_d1 = cp.where(
                    self.fuel_type == ftype,
                    a_d1 * cp.power(1 - cp.exp(-b_d1 * self.isz), c_d1),
                    0.0
                )
                rsf_d1 = rsz_d1 * self.sf
                isf_d1 = cp.where(
                    self.fuel_type == ftype,
                    cp.where(
                        (1 - cp.power(rsf_d1 / a_d1, 1 / c_d1)) >= 0.01,
                        cp.log(1 - cp.power(rsf_d1 / a_d1, 1 / c_d1)) / -b_d1,
                        cp.log(0.01) / -b_d1
                    ),
                    0.0
                )

                # Mixedwood (primary) ROS and ISF
                self.a, self.b, self.c, self.q, self.bui0, self.be_max = self.rosParams[ftype]
                self.rsz = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.isz), self.c),
                    self.rsz
                )
                self.rsf = cp.where(
                    self.fuel_type == ftype,
                    self.rsz * self.sf,
                    self.rsf
                )

                isf_mixed = cp.where(
                    self.fuel_type == ftype,
                    cp.where(
                        (1 - cp.power(self.rsf / self.a, 1 / self.c)) >= 0.01,
                        cp.log(1 - cp.power(self.rsf / self.a, 1 / self.c)) / -self.b,
                        cp.log(0.01) / -self.b
                    ),
                    0.0
                )

                # Blend ISF based on percent dead fir
                self.isf = cp.where(
                    self.fuel_type == ftype,
                    (self.pdf / 100) * isf_mixed + (1 - self.pdf / 100) * isf_d1,
                    self.isf
                )

                # Wind and ISI adjustments
                self._calcISI_slopeWind(ftype)

                # RSI (head fire) for D1 component
                rsi_d1 = a_d1 * cp.power(1 - cp.exp(-b_d1 * self.isi), c_d1)
                brsi_d1 = a_d1 * cp.power(1 - cp.exp(-b_d1 * self.bisi), c_d1)

                # Final RSI blend
                if ftype == 13:
                    # M4
                    self.rsi = cp.where(
                        self.fuel_type == ftype,
                        (self.pdf / 100) * self.a * cp.power(1 - cp.exp(-self.b * self.isi), self.c) +
                        (1 - self.pdf / 100) * rsi_d1,
                        self.rsi
                    )
                    self.brsi = cp.where(
                        self.fuel_type == ftype,
                        (self.pdf / 100) * self.a * cp.power(1 - cp.exp(-self.b * self.bisi), self.c) +
                        (1 - self.pdf / 100) * brsi_d1,
                        self.brsi
                    )
                else:
                    # M3 (uses only 20% D1 influence)
                    self.rsi = cp.where(
                        self.fuel_type == ftype,
                        (self.pdf / 100) * self.a * cp.power(1 - cp.exp(-self.b * self.isi), self.c) +
                        0.2 * (1 - self.pdf / 100) * rsi_d1,
                        self.rsi
                    )
                    self.brsi = cp.where(
                        self.fuel_type == ftype,
                        (self.pdf / 100) * self.a * cp.power(1 - cp.exp(-self.b * self.bisi), self.c) +
                        0.2 * (1 - self.pdf / 100) * brsi_d1,
                        self.brsi
                    )

                # Buildup Effect
                self.be = cp.where(
                    self.fuel_type == ftype,
                    cp.where(
                        self.bui == 0,
                        0.0,
                        cp.exp(50 * cp.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))
                    ),
                    self.be
                )

            else:
                # ## Process all other fuel types...
                # Calculate no slope/no wind rate of spread
                self.rsz = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.isz), self.c),
                    self.rsz
                )

                # Calculate rate of spread with slope effect
                self.rsf = cp.where(
                    self.fuel_type == ftype,
                    self.rsz * self.sf,
                    self.rsf
                )

                # Calculate initial spread index with slope effect
                isf_numer = cp.where(
                    self.fuel_type == ftype,
                    1 - cp.power(self.rsf / self.a, 1 / self.c),
                    0.0
                )
                np.seterr(divide='ignore')  # Suppress divide warnings for log
                isf_safe = cp.clip(isf_numer, 0.01, None)
                self.isf = cp.where(
                    self.fuel_type == ftype,
                    cp.log(isf_safe) / -self.b,
                    self.isf
                )
                np.seterr(divide='warn')  # Restore divide warnings

                # Calculate slope effects on wind and ISI
                self._calcISI_slopeWind(ftype)

                # Calculate head fire rate of spread with slope and wind effects
                self.rsi = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.isi), self.c),
                    self.rsi
                )

                # Calculate back fire rate of spread with slope and wind effects
                self.brsi = cp.where(
                    self.fuel_type == ftype,
                    self.a * cp.power(1 - cp.exp(-self.b * self.bisi), self.c),
                    self.brsi
                )

                # Calculate Buildup Effect (BE)
                self.be = cp.where(
                    self.fuel_type == ftype,
                    cp.where(
                        self.bui == 0,
                        0.0,
                        cp.exp(50 * cp.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))
                    ),
                    self.be
                )

        else:
            # ## Process M-1/2 fuel types...
            *_, self.q, self.bui0, self.be_max = self.rosParams[ftype]

            # Calculate no slope/no wind rate of spread
            # Get C2 RSZ and ISF
            a_c2, b_c2, c_c2, q_c2, *_ = self.rosParams[2]

            # Calculate C2 RSZ
            rsz_c2 = cp.where(
                self.fuel_type == ftype,
                a_c2 * cp.power(1 - cp.exp(-b_c2 * self.isz), c_c2),
                0.0
            )

            # Calculate C2 RSF
            rsf_c2 = cp.where(
                self.fuel_type == ftype,
                rsz_c2 * self.sf,
                0.0
            )

            # Calculate C2 ISF
            isf_c2 = cp.where(
                self.fuel_type == ftype,
                cp.where(
                    (1 - cp.power(rsf_c2 / a_c2, 1 / c_c2)) >= 0.01,
                    cp.log(1 - cp.power(rsf_c2 / a_c2, 1 / c_c2)) / -b_c2,
                    cp.log(0.01) / -b_c2
                ),
                0.0
            )

            # Get parameters for D1/D2 fuel types
            if ftype == 10:
                # Get D1 parameters
                a_d1_2, b_d1_2, c_d1_2, q_d1_2, *_ = self.rosParams[8]
            else:
                # Get D2 parameters (technically the same as D1)
                a_d1_2, b_d1_2, c_d1_2, q_d1_2, *_ = self.rosParams[9]

            # Calculate D1 RSZ
            rsz_d1_2 = cp.where(
                self.fuel_type == ftype,
                a_d1_2 * cp.power(1 - cp.exp(-b_d1_2 * self.isz), c_d1_2),
                0.0
            )

            # Calculate D1 RSF
            rsf_d1_2 = cp.where(
                self.fuel_type == ftype,
                rsz_d1_2 * self.sf,
                0.0
            )

            # Calculate D1 ISF
            isf_d1_2 = cp.where(
                self.fuel_type == ftype,
                cp.where(
                    (1 - cp.power(rsf_d1_2 / a_d1_2, 1 / c_d1_2)) >= 0.01,
                    cp.log(1 - cp.power(rsf_d1_2 / a_d1_2, 1 / c_d1_2)) / -b_d1_2,
                    cp.log(0.01) / -b_d1_2
                ),
                0.0
            )

            # Calculate initial spread index with slope effects
            self.isf = cp.where(
                self.fuel_type == ftype,
                (self.pc / 100) * isf_c2 + (1 - self.pc / 100) * isf_d1_2,
                self.isf
            )

            # Calculate slope effects on wind and ISI
            self._calcISI_slopeWind(ftype)

            # Calculate head and back fire rates of spread with slope and wind effects for C2 and D1
            # RSI (C2 & D1 head spread)
            rsi_c2 = a_c2 * cp.power(1 - cp.exp(-b_c2 * self.isi), c_c2)
            rsi_d1 = a_d1_2 * cp.power(1 - cp.exp(-b_d1_2 * self.isi), c_d1_2)

            # Back fire RSI
            brsi_c2 = a_c2 * cp.power(1 - cp.exp(-b_c2 * self.bisi), c_c2)
            brsi_d1 = a_d1_2 * cp.power(1 - cp.exp(-b_d1_2 * self.bisi), c_d1_2)

            # Calculate rate of spread with slope and wind effects (RSI)
            if ftype == 11:
                # M2: Adjust for D2 fuel type = 20% D1 influence
                self.rsi = cp.where(
                    self.fuel_type == ftype,
                    (self.pc / 100) * rsi_c2 + 0.2 * (1 - self.pc / 100) * rsi_d1,
                    self.rsi
                )
                self.brsi = cp.where(
                    self.fuel_type == ftype,
                    (self.pc / 100) * brsi_c2 + 0.2 * (1 - self.pc / 100) * brsi_d1,
                    self.brsi
                )
            else:
                # M1: standard blend
                self.rsi = cp.where(
                    self.fuel_type == ftype,
                    (self.pc / 100) * rsi_c2 + (1 - self.pc / 100) * rsi_d1,
                    self.rsi
                )
                self.brsi = cp.where(
                    self.fuel_type == ftype,
                    (self.pc / 100) * brsi_c2 + (1 - self.pc / 100) * brsi_d1,
                    self.brsi
                )

            # Calculate Buildup Effect (BE)
            self.be = cp.where(
                self.fuel_type == ftype,
                cp.where(
                    self.bui == 0,
                    0.0,
                    cp.exp(50 * cp.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))
                ),
                self.be
            )

        # Ensure BE does not exceed be_max
        self.be = cp.where(
            self.fuel_type == ftype,
            cp.minimum(self.be, self.be_max),
            self.be
        )

        return

    def calcROS(self, ftype: int) -> None:
        """
        Function to model the fire rate of spread (m/min) using CuPy.
        For C6, this is the surface fire heading and backing rate of spread.
        For all other fuel types, this is the overall heading and backing fire rate of spread.

        :param ftype: The numeric FBP fuel type code.
        :return: None
        """
        # Calculate Final ROS
        if ftype == 6:
            # C6 fuel type
            # Head Surface Fire ROS
            self.sfros = cp.where(
                self.fuel_type == ftype,
                self.rsi * self.be,
                self.sfros)

            # Back Surface Fire ROS
            self.bros = cp.where(
                self.fuel_type == ftype,
                self.brsi * self.be,
                self.bros)
        else:
            # All other fuel types
            # Head Fire ROS
            self.hfros = cp.where(
                self.fuel_type == ftype,
                self.rsi * self.be,
                self.hfros)

            # Back Fire ROS
            self.bros = cp.where(
                self.fuel_type == ftype,
                self.brsi * self.be,
                self.bros)

            if ftype == 9:
                # D2: Suppress spread under low BUI conditions
                # Head Fire ROS
                self.hfros = cp.where(
                    self.fuel_type == ftype,
                    cp.where(self.bui < 70,
                             cp.zeros_like(self.fuel_type),
                             self.hfros * 0.2),
                    self.hfros)

                # Back Fire ROS
                self.bros = cp.where(
                    self.fuel_type == ftype,
                    cp.where(self.bui < 70,
                             cp.zeros_like(self.fuel_type),
                             self.bros * 0.2),
                    self.bros)

        return

    def calcSFC(self, ftype: int) -> None:
        """
        Function to calculate forest floor consumption (FFC), woody fuel consumption (WFC),
        and total surface fuel consumption (SFC) using CuPy.

        :param ftype: The numeric FBP fuel type code.
        :return: None
        """
        if ftype == 1:
            ffc = cp.nan
            wfc = cp.nan
            sfc = cp.where(self.ffmc > 84,
                           0.75 + 0.75 * cp.sqrt(1 - cp.exp(-0.23 * (self.ffmc - 84))),
                           0.75 - 0.75 * cp.sqrt(1 - cp.exp(0.23 * (self.ffmc - 84))))
        elif ftype == 2:
            ffc = cp.nan
            wfc = cp.nan
            sfc = 5 * (1 - cp.exp(-0.0115 * self.bui))
        elif ftype in [3, 4]:
            ffc = cp.nan
            wfc = cp.nan
            sfc = 5 * cp.power(1 - cp.exp(-0.0164 * self.bui), 2.24)
        elif ftype in [5, 6]:
            ffc = cp.nan
            wfc = cp.nan
            sfc = 5 * cp.power(1 - cp.exp(-0.0149 * self.bui), 2.48)
        elif ftype == 7:
            ffc = 2 * (1 - cp.exp(-0.104 * (self.ffmc - 70)))
            ffc = cp.where(ffc < 0, cp.zeros_like(self.ffc), ffc)
            wfc = 1.5 * (1 - cp.exp(-0.0201 * self.bui))
            sfc = ffc + wfc
        elif ftype in [8, 9]:
            ffc = cp.nan
            wfc = cp.nan
            sfc = 1.5 * (1 - cp.exp(-0.0183 * self.bui))
        elif ftype in [10, 11]:
            c2_sfc = 5 * (1 - cp.exp(-0.0115 * self.bui))
            d1_sfc = 1.5 * (1 - cp.exp(-0.0183 * self.bui))
            ffc = cp.nan
            wfc = cp.nan
            sfc = ((self.pc / 100) * c2_sfc) + (((100 - self.pc) / 100) * d1_sfc)
        elif ftype in [12, 13]:
            ffc = cp.nan
            wfc = cp.nan
            sfc = 5 * (1 - cp.exp(-0.0115 * self.bui))
        elif ftype in [14, 15]:
            ffc = cp.nan
            wfc = cp.nan
            sfc = self.gfl
        elif ftype == 16:
            ffc = 4 * (1 - cp.exp(-0.025 * self.bui))
            wfc = 4 * (1 - cp.exp(-0.034 * self.bui))
            sfc = ffc + wfc
        elif ftype == 17:
            ffc = 10 * (1 - cp.exp(-0.013 * self.bui))
            wfc = 6 * (1 - cp.exp(-0.06 * self.bui))
            sfc = ffc + wfc
        elif ftype == 18:
            ffc = 12 * (1 - cp.exp(-0.0166 * self.bui))
            wfc = 20 * (1 - cp.exp(-0.021 * self.bui))
            sfc = ffc + wfc
        elif ftype == 19 or ftype == 20:
            ffc = cp.nan
            wfc = cp.nan
            sfc = cp.nan
        else:
            ffc = cp.nan
            wfc = cp.nan
            sfc = cp.nan

        # Assign FFC
        self.ffc = cp.where(self.fuel_type == ftype, ffc, self.ffc)
        # Assign WFC
        self.wfc = cp.where(self.fuel_type == ftype, wfc, self.wfc)
        # Assign SFC
        self.sfc = cp.where(self.fuel_type == ftype, sfc, self.sfc)

        return

    def getCBH_CFL(self, ftype: int) -> None:
        """
        Function to get the default CFFBPS canopy base height (CBH) and
        canopy fuel load (CFL) values for a specified fuel type using CuPy.

        :param ftype: The numeric FBP fuel type code.
        :return: None
        """
        # Get canopy base height (CBH) for fuel type
        cbh_val = self.fbpCBH_CFL_HT_LUT.get(ftype, (cp.nan, cp.nan))[0]
        self.cbh = cp.where(self.fuel_type == ftype, cbh_val, self.cbh)

        # Get canopy fuel load (CFL) for fuel type
        cfl_val = self.fbpCBH_CFL_HT_LUT.get(ftype, (cp.nan, cp.nan))[1]
        self.cfl = cp.where(self.fuel_type == ftype, cfl_val, self.cfl)

        return

    def calcCSFI(self) -> None:
        """
        Function to calculate the critical surface fire intensity (CSFI) using CuPy.

        :return: None
        """
        # Calculate critical surface fire intensity (CSFI)
        self.csfi = cp.where(
            self.fuel_type < 14,
            cp.power(0.01 * self.cbh * (460 + (25.9 * self.fmc)), 1.5),
            0.0
        )
        return

    def calcRSO(self) -> None:
        """
        Function to calculate the critical surface fire rate of spread (RSO) using CuPy.

        :return: None
        """
        # Calculate critical surface fire rate of spread (RSO)
        self.rso = cp.where(
            self.sfc > 0,
            self.csfi / (300.0 * self.sfc),
            0.0
        )
        return

    def calcCFB(self) -> None:
        """
        Function to calculate crown fraction burned using CuPy.
        Equation per Forestry Canada Fire Danger Group (1992).

        :return: None
        """
        # Masks
        is_c6 = self.fuel_type == 6
        non_crowning = cp.isin(self.fuel_type, self.non_crowning_fuels)
        is_other = cp.isin(self.fuel_type, self.ftypes) & ~is_c6 & ~non_crowning

        # Precompute exponent input values
        delta_sfros_c6 = self.sfros - self.rso
        delta_hfros_other = self.hfros - self.rso

        # Compute CFB
        cfb_c6 = cp.where(delta_sfros_c6 < -3086, 0.0, 1 - cp.exp(-0.23 * delta_sfros_c6))
        cfb_other = cp.where(delta_hfros_other < -3086, 0.0, 1 - cp.exp(-0.23 * delta_hfros_other))

        # Initialize output
        self.cfb = cp.zeros_like(self.fuel_type, dtype=cp.float32)
        self.cfb = cp.where(is_c6, cfb_c6, self.cfb)
        self.cfb = cp.where(is_other, cfb_other, self.cfb)

        # Ensure values range between 0 and 1
        self.cfb = cp.clip(cp.nan_to_num(self.cfb, nan=0.0), 0, 1)

        # Clean up memory
        del is_c6, is_other, delta_sfros_c6, delta_hfros_other, cfb_c6, cfb_other

        return

    def calcAccelParam(self) -> None:
        """
        Function to calculate acceleration parameter for a fire starting from a point ignition source.

        :return: None
        """
        # Mask for open fuel types that use a fixed acceleration parameter (0.115)
        fixed_mask = cp.isin(self.fuel_type, self.open_fuel_types)

        # Mask for closed fuel types that require computation
        variable_mask = cp.isin(self.fuel_type, self.ftypes) & ~fixed_mask

        # Compute acceleration parameter for open fuel types
        self.accel_param = cp.where(
            fixed_mask,
            cp.full_like(fixed_mask, 0.115, dtype=cp.float32),
            self.accel_param
        )

        # Compute acceleration parameter for closed fuel types
        self.accel_param = cp.where(
            variable_mask,
            0.115 - 18.8 * cp.power(self.cfb, 2.5) * cp.exp(-8 * self.cfb),
            self.accel_param
        )

        # Clean up memory
        del fixed_mask, variable_mask

        return

    def calcFireType(self) -> None:
        """
        Function to calculate fire type using CuPy.
            - 1: surface
            - 2: intermittent crown
            - 3: active crown
        Applies only to fuels with numeric codes < 19.

        :return: None
        """
        self.fire_type = cp.where(
            self.fuel_type < 19,
            cp.where(
                self.cfb <= 0.1,
                # Surface fire
                1,
                cp.where(
                    (self.cfb > 0.1) & (self.cfb < 0.9),
                    # Intermittent crown fire
                    2,
                    cp.where(
                        self.cfb >= 0.9,
                        # Active crown fire
                        3,
                        # No fire type
                        0,
                    ),
                ),
            ),
            cp.zeros_like(self.fuel_type),
        ).astype(cp.int8)

        return

    def calcCFC(self) -> None:
        """
        Function to calculate crown fuel consumed (kg/m^2) using CuPy.

        :return: None
        """
        self.cfc = cp.where(
            (self.fuel_type == 10) | (self.fuel_type == 11),
            self.cfb * self.cfl * self.pc / 100,
            cp.where(
                (self.fuel_type == 12) | (self.fuel_type == 13),
                self.cfb * self.cfl * self.pdf / 100,
                self.cfb * self.cfl
            )
        )

        return

    def calcC6hfros(self) -> None:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type using CuPy.

        :returns: None
        """
        self.cfros = cp.where(
            self.fuel_type == 6,
            cp.where(
                self.cfc == 0,
                cp.zeros_like(self.fuel_type),
                60 * cp.power(1 - cp.exp(-0.0497 * self.isi), 1) * (self.fme / 0.778237),
            ),
            self.cfros
        )

        self.hfros = cp.where(
            self.fuel_type == 6,
            self.sfros + (self.cfb * (self.cfros - self.sfros)),
            self.hfros
        )

        return

    def calcTFC(self) -> None:
        """
        Function to calculate total fuel consumed (kg/m^2) using CuPy.

        :return: None
        """
        self.tfc = self.sfc + self.cfc

        return

    def calcHFI(self) -> None:
        """
        Function to calculate head fire intensity.
        """
        self.hfi = 300 * self.hfros * self.tfc

        return

    def setParams(self, set_dict: dict) -> None:
        """
        Function to set FBP parameters to specific values using CuPy.

        :param set_dict: Dictionary of FBP parameter names and the values to assign to the FBP class object
        :return: None
        """
        # Iterate through the set dictionary and assign values
        for key, value in set_dict.items():
            if hasattr(self, key):  # Check if the class has the attribute
                if isinstance(value, cp.ndarray):  # Check if value is already a CuPy array
                    # Mask NaN values using CuPy
                    setattr(self, key, cp.where(cp.isnan(value), cp.nan, value))
                else:
                    # Convert scalar or list to a CuPy array and mask NaN values
                    value_array = (cp.asarray(value)
                                   if isinstance(value, (list, tuple, np.ndarray))
                                   else cp.asarray([value]))
                    setattr(self, key, cp.where(cp.isnan(value_array), cp.nan, value_array))
        return

    def getParams(self, out_request: list[str]) -> Union[list, str, None]:
        """
        Function to output requested dataset parameters from the FBP class using CuPy.

        :param out_request: List of requested FBP parameters.
        :return: List of requested outputs.
        """
        # Dictionary of CFFBPS parameters
        fbp_params = {
            # Default output variables
            'fire_type': self.fire_type,  # Type of fire (surface, intermittent crown, active crown)
            'hfros': self.hfros,  # Head fire rate of spread (m/min)
            'hfi': self.hfi,  # Head fire intensity (kW/m)

            # Weather variables
            'ws': self.ws,  # Observed wind speed (km/h)
            'wd': self.wd,  # Wind azimuth/direction (degrees)
            'm': self.m,  # Moisture content equivalent of the FFMC (%, value from 0-100+)
            'fF': self.fF,  # Fine fuel moisture function in the ISI
            'fW': self.fW,  # Wind function in the ISI
            'isi': self.isi,  # Final ISI, accounting for wind and slope

            # Slope + wind effect variables
            'a': self.a,  # Rate of spread equation coefficient
            'b': self.b,  # Rate of spread equation coefficient
            'c': self.c,  # Rate of spread equation coefficient
            'rsz': self.rsz,  # Surface spread rate with zero wind on level terrain
            'sf': self.sf,  # Slope factor
            'rsf': self.rsf,  # Spread rate with zero wind, upslope
            'isf': self.isf,  # ISI, with zero wind upslope
            'rsi': self.rsi,  # Initial spread rate without BUI effect
            'wse1': self.wse1,  # Original slope equivalent wind speed value for cases where WSE1 <= 40
            'wse2': self.wse2,  # New slope equivalent wind speed value for cases where WSE1 > 40
            'wse': self.wse,  # Slope equivalent wind speed
            'wsx': self.wsx,  # Net vectorized wind speed in the x-direction
            'wsy': self.wsy,  # Net vectorized wind speed in the y-direction
            'wsv': self.wsv,  # Net vectorized wind speed
            'raz': self.raz,  # Net vectorized wind direction

            # BUI effect variables
            'q': self.q,  # Proportion of maximum rate of spread at BUI equal to 50
            'bui0': self.bui0,  # Average BUI for each fuel type
            'be': self.be,  # Buildup effect on spread rate
            'be_max': self.be_max,  # Maximum allowable BE value

            # Surface fuel variables
            'ffc': self.ffc,  # Estimated forest floor consumption
            'wfc': self.wfc,  # Estimated woody fuel consumption
            'sfc': self.sfc,  # Estimated total surface fuel consumption

            # Foliar moisture content variables
            'latn': self.latn,  # Normalized latitude
            'dj': self.dj,  # Julian date of day being modeled
            'd0': self.d0,  # Julian date of minimum foliar moisture content
            'nd': self.nd,  # Number of days between modeled fire date and d0
            'fmc': self.fmc,  # Foliar moisture content
            'fme': self.fme,  # Foliar moisture effect

            # Critical crown fire threshold variables
            'csfi': self.csfi,  # Critical intensity (kW/m)
            'rso': self.rso,  # Critical rate of spread (m/min)

            # Back fire spread variables
            'bfw': self.bfW,  # The back fire wind function
            'bisi': self.bisi,  # The ISI associated with the back fire rate of spread
            'bros': self.bros,  # Backing rate of spread (m/min)

            # Crown fuel parameters
            'cbh': self.cbh,  # Height to live crown base (m)
            'cfb': self.cfb,  # Crown fraction burned (proportion, value ranging from 0-1)
            'cfl': self.cfl,  # Crown fuel load (kg/m^2)
            'cfc': self.cfc,  # Crown fuel consumed

            # Acceleration parameter
            'accel': self.accel_param
        }

        # Retrieve requested parameters
        if self.return_array:
            if self.return_array_as == 'cupy':
                return [
                    cp.array(fbp_params.get(var)) if fbp_params.get(var) is not None
                    else 'Invalid output variable'
                    for var in out_request
                ]
            elif self.return_array_as == 'numpy':
                return [
                    np.array(cp.asnumpy(fbp_params.get(var))) if fbp_params.get(var) is not None
                    else 'Invalid output variable'
                    for var in out_request
                ]

        else:
            return [
                cp.asnumpy(fbp_params.get(var)).item() if fbp_params.get(var).ndim == 0
                else cp.asnumpy(fbp_params.get(var))[0].item() if fbp_params.get(var) is not None
                else 'Invalid output variable'
                for var in out_request
            ]

    def runFBP(self, block: Optional[cp.ndarray] = None) -> list[any]:
        """
        Function to automatically run CFFBPS modeling using CuPy.

        :param block: The array of partial data (block) to run FBP with.
        :returns:
            List of values requested through the `out_request` parameter. Default values are `fire_type`, `hfros`, and `hfi`.
        """
        if not self.initialized:
            raise ValueError('FBP class must be initialized before running calculations. Call "initialize" first.')

        if block is not None:
            self.block = cp.asarray(block)

        # ### Model fire behavior with CFFBPS
        # Invert wind direction and aspect
        self.invertWindAspect()
        # Calculate slope factor
        self.calcSF()
        # Calculate zero slope & zero wind ISI
        self.calcISZ()
        # Calculate foliar moisture content
        self.calcFMC()

        # Unique fuel types present in the data
        unique_fuel_types = cp.unique(self.fuel_type).tolist()

        for ftype in [ftype for ftype in unique_fuel_types
                      if ftype in self.rosParams.keys()]:
            # Calculate ISI, RSI, and BE
            self.calcISI_RSI_BE(ftype)
            # Calculate ROS
            self.calcROS(ftype)
            # Calculate surface fuel consumption
            self.calcSFC(ftype)
            # Calculate canopy base height and canopy fuel load
            self.getCBH_CFL(ftype)
        # Calculate critical surface fire intensity
        self.calcCSFI()
        # Calculate critical surface fire rate of spread
        self.calcRSO()
        # Calculate crown fraction burned
        self.calcCFB()
        # Calculate acceleration parameter
        self.calcAccelParam()
        # Calculate fire type
        self.calcFireType()
        # Calculate crown fuel consumed
        self.calcCFC()
        # Calculate C6 head fire rate of spread
        self.calcC6hfros()
        # Calculate total fuel consumption
        self.calcTFC()
        # Calculate head fire intensity
        self.calcHFI()

        # Return requested values
        if self.out_request is None:
            self.out_request = ['hfros', 'hfi', 'fire_type']
        return self.getParams(self.out_request)


def _get_cupy_array(raster_path):
    """
    Load a raster as a CuPy array using the specified backend (Rasterio or GDAL).

    :param raster_path: Path to the raster file.
    :return: CuPy array of the raster data.
    """
    with rio.open(raster_path) as src:
        return cp.asarray(src.read(1))


def _testFBP(test_functions: list,
             wx_date: int,
             lat: Union[float, int, cp.ndarray],
             long: Union[float, int, cp.ndarray],
             elevation: Union[float, int, cp.ndarray],
             slope: Union[float, int, cp.ndarray],
             aspect: Union[float, int, cp.ndarray],
             ws: Union[float, int, cp.ndarray],
             wd: Union[float, int, cp.ndarray],
             ffmc: Union[float, int, cp.ndarray],
             bui: Union[float, int, cp.ndarray],
             pc: Optional[Union[float, int, cp.ndarray]] = 50,
             pdf: Optional[Union[float, int, cp.ndarray]] = 35,
             gfl: Optional[Union[float, int, cp.ndarray]] = 0.35,
             gcf: Optional[Union[float, int, cp.ndarray]] = 80,
             out_request: Optional[list[str]] = None,
             out_folder: Optional[str] = None) -> None:
    """
    Function to test the cffbps module with various input types
    :param test_functions: List of functions to test
        (options: ['numeric', 'array', 'raster', 'raster_multiprocessing'])
    :param wx_date: Date of weather observation (used for fmc calculation) (YYYYMMDD)
    :param lat: Latitude of area being modelled (Decimal Degrees, floating point)
    :param long: Longitude of area being modelled (Decimal Degrees, floating point)
    :param elevation: Elevation of area being modelled (m)
    :param slope: Ground slope angle/tilt of area being modelled (%)
    :param aspect: Ground slope aspect/azimuth of area (degrees)
    :param ws: Wind speed (km/h @ 10m height)
    :param wd: Wind direction (degrees, direction wind is coming from)
    :param ffmc: CFFWIS Fine Fuel Moisture Code
    :param bui: CFFWIS Buildup Index
    :param pc: Percent conifer (%, value from 0-100)
    :param pdf: Percent dead fir (%, value from 0-100)
    :param gfl: Grass fuel load (kg/m^2)
    :param gcf: Grass curing factor (%, value from 0-100)
    :param out_request: Tuple or list of CFFBPS output variables
        # Default output variables
        fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
        hfros = Head fire rate of spread (m/min)
        hfi = head fire intensity (kW/m)

        # Weather variables
        ws = Observed wind speed (km/h)
        wd = Wind azimuth/direction (degrees)
        m = Moisture content equivalent of the FFMC (%, value from 0-100+)
        fF = Fine fuel moisture function in the ISI equation
        fW = Wind function in the ISI equation
        isi = Final ISI, accounting for wind and slope

        # Slope + wind effect variables
        a = Rate of spread equation coefficient
        b = Rate of spread equation coefficient
        c = Rate of spread equation coefficient
        RSZ = Surface spread rate with zero wind on level terrain
        SF = Slope factor
        RSF = spread rate with zero wind, upslope
        ISF = ISI, with zero wind upslope
        RSI = Initial spread rate without BUI effect
        WSE1 = Original slope equivalent wind speed value
        WSE2 = New slope equivalent sind speed value for cases where WSE1 > 40 (capped at max of 112.45)
        WSE = Slope equivalent wind speed
        WSX = Net vectorized wind speed in the x-direction
        WSY = Net vectorized wind speed in the y-direction
        WSV = (aka: slope-adjusted wind speed) Net vectorized wind speed (km/h)
        RAZ = (aka: slope-adjusted wind direction) Net vectorized wind direction (degrees)

        # BUI effect variables
        q = Proportion of maximum rate of spread at BUI equal to 50
        bui0 = Average BUI for each fuel type
        BE = Buildup effect on spread rate
        be_max = Maximum allowable BE value

        # Surface fuel variables
        ffc = Estimated forest floor consumption
        wfc = Estimated woody fuel consumption
        sfc = Estimated total surface fuel consumption

        # Foliar moisture content variables
        latn = Normalized latitude
        d0 = Julian date of minimum foliar moisture content
        nd = number of days between modelled fire date and d0
        fmc = foliar moisture content
        fme = foliar moisture effect

        # Critical crown fire threshold variables
        csfi = critical intensity (kW/m)
        rso = critical rate of spread (m/min)

        # Crown fuel parameters
        cbh = Height to live crown base (m)
        cfb = Crown fraction burned (proportion, value ranging from 0-1)
        cfl = Crown fuel load (kg/m^2)
        cfc = Crown fuel consumed
    :param out_folder: Location to save test rasters (Default: <location of script>/Test_Data/Outputs)
    :return: None
    """
    import ProcessRasters as pr
    import generate_test_fbp_rasters as genras

    fbp = FBP()

    # Create fuel type list
    fuel_type_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4',
                      'O1a', 'O1b', 'S1', 'S2', 'S3', 'NF', 'WA']

    # Put inputs into list
    input_data = [wx_date, lat, long,
                  elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, out_request]

    # ### Test non-raster modelling
    if any(var in test_functions for var in ['numeric', 'all']):
        print('Testing non-raster modelling')
        for ft in fuel_type_list:
            fbp.initialize(*([fbpFTCode_AlphaToNum_LUT.get(ft)] + input_data))
            print('\t' + ft, fbp.runFBP())

    # ### Test array modelling
    if any(var in test_functions for var in ['array', 'all']):
        print('Testing array modelling')
        fbp.initialize(*([cp.array([fbpFTCode_AlphaToNum_LUT.get(ft) for ft in fuel_type_list])] + input_data))
        print('\t', fbp.runFBP())

    # Get test folders
    input_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Inputs')
    multiprocess_folder = os.path.join(input_folder, 'Multiprocessing')
    if out_folder is None:
        output_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Outputs')
    else:
        output_folder = out_folder

    # ### Test simple raster GPU processing
    if any(var in test_functions for var in ['raster', 'all']):
        print('Testing simple raster GPU processing')
        # Generate test raster datasets using user-provided input values
        genras.gen_test_data(*input_data[:-2], dtype=cp.float32)

        # Get input dataset paths
        raster_paths = {
            'fuel_type': os.path.join(input_folder, 'FuelType.tif'),
            'lat': os.path.join(input_folder, 'LAT.tif'),
            'long': os.path.join(input_folder, 'LONG.tif'),
            'elevation': os.path.join(input_folder, 'ELV.tif'),
            'slope': os.path.join(input_folder, 'GS.tif'),
            'aspect': os.path.join(input_folder, 'Aspect.tif'),
            'ws': os.path.join(input_folder, 'WS.tif'),
            'wd': os.path.join(input_folder, 'WD.tif'),
            'ffmc': os.path.join(input_folder, 'FFMC.tif'),
            'bui': os.path.join(input_folder, 'BUI.tif'),
            'pc': os.path.join(input_folder, 'PC.tif'),
            'pdf': os.path.join(input_folder, 'PDF.tif'),
            'gfl': os.path.join(input_folder, 'GFL.tif'),
            'gcf': os.path.join(input_folder, 'cc.tif'),
        }

        # Create a reference raster profile for final raster outputs
        with rio.open(raster_paths['gfl']) as src:
            ref_ras_profile = src.profile

        # Read raster data into CuPy arrays
        raster_data = {key: cp.asarray(rio.open(path).read()) for key, path in raster_paths.items()}

        # Run the FBP modeling
        fbp.initialize(
            fuel_type=raster_data['fuel_type'], wx_date=wx_date,
            lat=raster_data['lat'], long=raster_data['long'], elevation=raster_data['elevation'],
            slope=raster_data['slope'], aspect=raster_data['aspect'],
            ws=raster_data['ws'], wd=raster_data['wd'], ffmc=raster_data['ffmc'],
            bui=raster_data['bui'], pc=raster_data['pc'], pdf=raster_data['pdf'],
            gfl=raster_data['gfl'], gcf=raster_data['gcf'],
            out_request=['wsv', 'raz', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc'],
            convert_fuel_type_codes=False
        )
        fbp_result = fbp.runFBP()

        # Get output dataset paths
        output_rasters = [
            os.path.join(output_folder, name + '.tif')
            for name in ['wsv', 'raz', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc']
        ]

        for dset, path in zip(fbp_result, output_rasters):
            # Save output datasets
            pr.arrayToRaster(array=cp.asnumpy(dset),  # Convert to NumPy for rasterio compatibility
                             out_file=path,
                             ras_profile=ref_ras_profile,
                             dtype=np.float32)

    # ### Test larger raster GPU processing
    if any(var in test_functions for var in ['raster_multiprocessing', 'all']):
        print('Testing larger raster GPU processing')
        if not os.path.exists(os.path.join(output_folder, 'GPU_LargeRaster')):
            os.mkdir(os.path.join(output_folder, 'GPU_LargeRaster'))

        # Get input dataset paths
        raster_paths = {
            'fuel_type': os.path.join(multiprocess_folder, 'FuelType.tif'),
            'lat': os.path.join(multiprocess_folder, 'LAT.tif'),
            'long': os.path.join(multiprocess_folder, 'LONG.tif'),
            'elevation': os.path.join(multiprocess_folder, 'ELV.tif'),
            'slope': os.path.join(multiprocess_folder, 'GS.tif'),
            'aspect': os.path.join(multiprocess_folder, 'Aspect.tif'),
            'ws': os.path.join(multiprocess_folder, 'WS.tif'),
            # 'wd': os.path.join(multiprocess_folder, 'WD.tif'),
            'ffmc': os.path.join(multiprocess_folder, 'FFMC.tif'),
            'bui': os.path.join(multiprocess_folder, 'BUI.tif'),
            'pc': os.path.join(multiprocess_folder, 'PC.tif'),
            'pdf': os.path.join(multiprocess_folder, 'PDF.tif'),
            'gfl': os.path.join(multiprocess_folder, 'GFL.tif'),
            # 'gcf': os.path.join(multiprocess_folder, 'cc.tif'),
        }

        # Create a reference raster profile for final raster outputs
        with rio.open(raster_paths['gfl']) as src:
            ref_ras_profile = src.profile

        # Read raster data into CuPy arrays
        # Check dtype and conditionally convert datasets > float32 to float32
        # to maximize GTX1650 performance
        raster_data = {}
        for key, path in raster_paths.items():
            with rio.open(path) as src:
                # Get the dtype of the raster
                raster_dtype = src.dtypes[0]  # Assuming single-band rasters; use indexing for multi-band rasters
                # Read raster data
                raster_array = src.read()
                # Conditionally convert to float32
                if np.dtype(raster_dtype).itemsize > np.dtype(np.float32).itemsize:
                    raster_data[key] = cp.asarray(raster_array.astype(np.float32))
                else:
                    raster_data[key] = cp.asarray(raster_array)

        # Run the FBP modeling
        fbp.initialize(
            fuel_type=raster_data['fuel_type'], wx_date=wx_date,
            lat=raster_data['lat'], long=raster_data['long'], elevation=raster_data['elevation'],
            slope=raster_data['slope'], aspect=raster_data['aspect'],
            ws=raster_data['ws'], wd=wd, ffmc=raster_data['ffmc'],
            bui=raster_data['bui'], pc=raster_data['pc'], pdf=raster_data['pdf'],
            gfl=raster_data['gfl'], gcf=getSeasonGrassCuring(season='summer', province='BC'),
            out_request=['wsv', 'raz', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc'],
            convert_fuel_type_codes=True
        )
        fbp_result = fbp.runFBP()

        # Get output dataset paths
        output_rasters = [
            os.path.join(output_folder, 'GPU_LargeRaster', name + '.tif')
            for name in ['wsv', 'raz', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc']
        ]

        for dset, path in zip(fbp_result, output_rasters):
            dset_profile = ref_ras_profile
            if dset.dtype.itemsize > cp.float32().itemsize:
                # Set raster profile
                dset_profile['nodata'] = np.finfo(np.float32).max
                dtype = np.float32
            else:
                dtype = dset.dtype
            # Save output datasets
            pr.arrayToRaster(array=dset,  # Convert to NumPy for rasterio compatibility
                             out_file=path,
                             ras_profile=dset_profile,
                             dtype=dtype)


if __name__ == '__main__':
    # _test_functions options: ['all', 'numeric', 'array', 'raster', 'raster_multiprocessing']
    _test_functions = ['all']
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
    _pc = 50
    _pdf = 50
    _gfl = 0.35
    _gcf = 80
    _out_request = ['latn', 'd0', 'dj', 'nd', 'fmc', 'fme', 'csfi', 'rso', 'hfros', 'hfi']
    _out_folder = None

    # Test the FBP functions
    _testFBP(test_functions=_test_functions,
             wx_date=_wx_date, lat=_lat, long=_long,
             elevation=_elevation, slope=_slope, aspect=_aspect,
             ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
             pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf,
             out_request=_out_request,
             out_folder=_out_folder)
