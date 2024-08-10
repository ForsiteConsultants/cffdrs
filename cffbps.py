# -*- coding: utf-8 -*-
"""
Created on Mon July 22 10:00:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import os
from typing import Union, Optional
import numpy as np
from numpy import ma as mask
from datetime import datetime as dt

# CFFBPS Fuel Type Numeric-Alphanumeric Code Lookup Table
fbpFTCode_NumToAlpha_LUT = {
    1: 'C1',  # C-1
    2: 'C2',  # C-2
    3: 'C3',  # C-3
    4: 'C4',  # C-4
    5: 'C5',  # C-5
    6: 'C6',  # C-6
    7: 'C7',  # C-7
    8: 'D1',  # D-1
    9: 'D2',  # D-2
    10: 'M1',  # M-1
    11: 'M2',  # M-2
    12: 'M3',  # M-3
    13: 'M4',  # M-4
    14: 'O1a',  # O-1a
    15: 'O1b',  # O-1b
    16: 'S1',  # S-1
    17: 'S2',  # S-2
    18: 'S3',  # S-3
    19: 'NF',  # NF (non-fuel)
    20: 'WA'  # WA (water)
}

# CFFBPS Fuel Type Alphanumeric-Numeric Code Lookup Table
fbpFTCode_AlphaToNum_LUT = {
    'C1': 1,  # C-1
    'C2': 2,  # C-2
    'C3': 3,  # C-3
    'C4': 4,  # C-4
    'C5': 5,  # C-5
    'C6': 6,  # C-6
    'C7': 7,  # C-7
    'D1': 8,  # D-1
    'D2': 9,  # D-2
    'M1': 10,  # M-1
    'M2': 11,  # M-2
    'M3': 12,  # M-3
    'M4': 13,  # M-4
    'O1a': 14,  # O-1a
    'O1b': 15,  # O-1b
    'S1': 16,  # S-1
    'S2': 17,  # S-2
    'S3': 18,  # S-3
    'NF': 19,  # NF (non-fuel)
    'WA': 20,  # WA (water)
}


def convert_grid_codes(fuel_type_array: np.ndarray) -> np.ndarray:
    """
    Function to convert array grid code values from the CFS cffdrs R version to the version used in this module
    :param fuel_type_array: CFFBPS fuel type array, containing the CFS cffdrs R version of grid codes
    :return: modified fuel_type_array
    """
    fuel_type_array = mask.where(fuel_type_array == 19,
                                 20,
                                 mask.where(fuel_type_array == 13,
                                            19,
                                            mask.where(fuel_type_array == 12,
                                                       13,
                                                       mask.where(fuel_type_array == 11,
                                                                  12,
                                                                  mask.where(fuel_type_array == 10,
                                                                             11,
                                                                             mask.where(fuel_type_array == 9,
                                                                                        10,
                                                                                        fuel_type_array))))))
    return fuel_type_array


##################################################################################################
# #### CLASS FOR CANADIAN FOREST FIRE BEHAVIOR PREDICTION SYSTEM (CFFBPS) MODELLING ####
##################################################################################################
class FBP:
    """
    Class to model fire type, head fire rate of spread, and head fire intensity with CFFBPS
    :param fuel_type: CFFBPS fuel type (numeric code: 1-18)
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
    : param convert_fuel_type_codes: Convert from CFS cffdrs R fuel type grid codes
        to the grid codes used in this module
    :returns:
        Tuple of values requested through out_request parameter. Default values are fire_type, hfros, and hfi.
    """

    def __init__(self,
                 fuel_type: Union[int, str, np.ndarray],
                 wx_date: int,
                 lat: Union[float, int, np.ndarray],
                 long: Union[float, int, np.ndarray],
                 elevation: Union[float, int, np.ndarray],
                 slope: Union[float, int, np.ndarray],
                 aspect: Union[float, int, np.ndarray],
                 ws: Union[float, int, np.ndarray],
                 wd: Union[float, int, np.ndarray],
                 ffmc: Union[float, int, np.ndarray],
                 bui: Union[float, int, np.ndarray],
                 pc: Optional[Union[float, int, np.ndarray]] = 50,
                 pdf: Optional[Union[float, int, np.ndarray]] = 35,
                 gfl: Optional[Union[float, int, np.ndarray]] = 0.35,
                 gcf: Optional[Union[float, int, np.ndarray]] = 80,
                 out_request: Optional[list[str]] = None,
                 convert_fuel_type_codes: Optional[bool] = False):

        # Initialize CFFBPS input parameters
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

        # Array verification parameter
        self.return_array = None
        self.ref_array = None

        # Verify input parameters
        self._checkArray()
        self._verifyInputs()

        # Initialize weather parameters
        self.isi = self.ref_array
        self.m = self.ref_array
        self.fF = self.ref_array
        self.fW = self.ref_array

        # Initialize slope effect parameters
        self.a = self.ref_array
        self.b = self.ref_array
        self.c = self.ref_array
        self.rsz = self.ref_array
        self.isz = self.ref_array
        self.sf = self.ref_array
        self.rsf = self.ref_array
        self.isf = self.ref_array
        self.rsi = self.ref_array
        self.wse1 = self.ref_array
        self.wse2 = self.ref_array
        self.wse = self.ref_array
        self.wsx = self.ref_array
        self.wsy = self.ref_array
        self.wsv = self.ref_array
        self.raz = self.ref_array

        # Initialize BUI effect parameters
        self.q = self.ref_array
        self.bui0 = self.ref_array
        self.be = self.ref_array
        self.be_max = self.ref_array

        # Initialize surface parameters
        self.cf = self.ref_array
        self.ffc = self.ref_array
        self.wfc = self.ref_array
        self.sfc = self.ref_array
        self.rss = self.ref_array

        # Initialize foliar moisture content parameters
        self.latn = self.ref_array
        self.dj = self.ref_array
        self.d0 = self.ref_array
        self.nd = self.ref_array
        self.fmc = self.ref_array
        self.fme = self.ref_array

        # Initialize crown and total fuel consumed parameters
        self.cbh = self.ref_array
        self.csfi = self.ref_array
        self.rso = self.ref_array
        self.rsc = self.ref_array
        self.cfb = self.ref_array
        self.cfl = self.ref_array
        self.cfc = self.ref_array
        self.tfc = self.ref_array

        # Initialize default CFFBPS output parameters
        self.fire_type = self.ref_array
        self.hfros = self.ref_array
        self.hfi = self.ref_array

        # Initialize other parameters
        self.ftype = self.ref_array
        self.sfros = self.ref_array
        self.cfros = self.ref_array

        # ### Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = ['cbh', 'fmc']
        self.rsoVarList = ['csfi', 'sfc']
        self.cfbVarList = ['cfros', 'rso']
        self.cfcVarList = ['cfb', 'cfl']
        self.cfiVarList = ['cfros', 'cfc']

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
            1: (90, 0.0649, 4.5, 0.9, 72, 1.076),  # C-1
            2: (110, 0.0282, 1.5, 0.7, 64, 1.321),  # C-2
            3: (110, 0.0444, 3, 0.75, 62, 1.261),  # C-3
            4: (110, 0.0293, 1.5, 0.8, 66, 1.184),  # C-4
            5: (30, 0.0697, 4, 0.8, 56, 1.220),  # C-5
            6: (30, 0.08, 3, 0.8, 62, 1.197),  # C-6
            7: (45, 0.0305, 2, 0.85, 106, 1.134),  # C-7
            8: (30, 0.0232, 1.6, 0.9, 32, 1.179),  # D-1
            9: (30, 0.0232, 1.6, 0.9, 32, 1.179),  # D-2
            10: (0.8, 50, 1.250),  # M-1
            11: (0.8, 50, 1.250),  # M-2
            12: (120, 0.0572, 1.4, 0.8, 50, 1.250),  # M-3
            13: (100, 0.0404, 1.48, 0.8, 50, 1.250),  # M-4
            14: (190, 0.0310, 1.4, 1, None, 1),  # O-1a
            15: (250, 0.0350, 1.7, 1, None, 1),  # O-1b
            16: (75, 0.0297, 1.3, 0.75, 38, 1.460),  # S-1
            17: (40, 0.0438, 1.7, 0.75, 63, 1.256),  # S-2
            18: (55, 0.0829, 3.2, 0.75, 31, 1.590)  # S-3
        }

    def _checkArray(self) -> None:
        """
        Function to check if any of the input parameters are numpy arrays.
        :return: None
        """
        # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
        input_list = [
            self.fuel_type, self.lat, self.long,
            self.elevation, self.slope, self.aspect,
            self.ws, self.wd, self.ffmc, self.bui,
            self.pc, self.pdf,
            self.gfl, self.gcf
        ]
        if any(isinstance(data, np.ndarray) for data in input_list):
            self.return_array = True

            # Get indices of input parameters that are arrays
            array_indices = [i for i in range(len(input_list)) if isinstance(input_list[i], np.ndarray)]
            # Get first input parameter array as a masked array
            first_array = input_list[array_indices[0]]
            if (array_indices[0] == 0) and ('<U' in str(first_array.dtype)):
                for ftype in np.unique(self.fuel_type):
                    self.fuel_type = np.where(self.fuel_type == ftype,
                                              fbpFTCode_AlphaToNum_LUT.get(ftype),
                                              self.fuel_type)
                self.fuel_type = self.fuel_type.astype(np.float32)
                first_array = self.fuel_type
            self.ref_array = mask.array(first_array,
                                        mask=np.isnan(first_array)).astype(np.float32) * 0
        else:
            self.return_array = False
            # Get first input parameter array as a masked array
            self.ref_array = mask.array([self.fuel_type],
                                        mask=np.isnan([self.fuel_type])).astype(np.float32) * 0

    def _verifyInputs(self) -> None:
        """
        Function to check if any of the input parameters are numpy arrays.
        :return: None
        """
        # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
        # Verify fuel_type
        if not isinstance(self.fuel_type, (int, str, np.ndarray)):
            raise TypeError('fuel_type must be either int, string, or numpy ndarray data types')
        elif isinstance(self.fuel_type, np.ndarray):
            self.fuel_type = mask.array(self.fuel_type, mask=np.isnan(self.fuel_type))
        elif isinstance(self.fuel_type, str):
            self.fuel_type = mask.array([fbpFTCode_AlphaToNum_LUT.get(self.fuel_type)],
                                        mask=np.isnan([fbpFTCode_AlphaToNum_LUT.get(self.fuel_type)]))
        else:
            self.fuel_type = mask.array([self.fuel_type], mask=np.isnan([self.fuel_type]))
        # Convert from cffdrs R fuel type grid codes to the grid codes used in this module
        if self.convert_fuel_type_codes:
            self.fuel_type = convert_grid_codes(self.fuel_type)

        # Verify wx_date
        if not isinstance(self.wx_date, int):
            raise TypeError('wx_date must be either int or numpy ndarray data types')
        try:
            date_string = str(self.wx_date)
            dt.fromisoformat(f'{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}')
        except ValueError:
            raise ValueError('wx_date must be formatted as: YYYYMMDD')

        # Verify lat
        if not isinstance(self.lat, (int, float, np.ndarray)):
            raise TypeError('lat must be either int, float, or numpy ndarray data types')
        elif isinstance(self.lat, np.ndarray):
            self.lat = mask.array(self.lat, mask=np.isnan(self.lat))
        else:
            self.lat = mask.array([self.lat], mask=np.isnan([self.lat]))

        # Verify long
        if not isinstance(self.long, (int, float, np.ndarray)):
            raise TypeError('long must be either int, float, or numpy ndarray data types')
        elif isinstance(self.long, np.ndarray):
            self.long = mask.array(self.long, mask=np.isnan(self.long))
        else:
            self.long = mask.array([self.long], mask=np.isnan([self.long]))
        # Get absolute longitude values
        self.long = np.absolute(self.long)

        # Verify elevation
        if not isinstance(self.elevation, (int, float, np.ndarray)):
            raise TypeError('elevation must be either int, float, or numpy ndarray data types')
        elif isinstance(self.elevation, np.ndarray):
            self.elevation = mask.array(self.elevation, mask=np.isnan(self.elevation))
        else:
            self.elevation = mask.array([self.elevation], mask=np.isnan([self.elevation]))

        # Verify slope
        if not isinstance(self.slope, (int, float, np.ndarray)):
            raise TypeError('slope must be either int, float, or numpy ndarray data types')
        elif isinstance(self.slope, np.ndarray):
            self.slope = mask.array(self.slope, mask=np.isnan(self.slope))
        else:
            self.slope = mask.array([self.slope], mask=np.isnan([self.slope]))

        # Verify aspect
        if not isinstance(self.aspect, (int, float, np.ndarray)):
            raise TypeError('aspect must be either int, float, or numpy ndarray data types')
        elif isinstance(self.aspect, np.ndarray):
            self.aspect = mask.array(self.aspect, mask=np.isnan(self.aspect))
        else:
            self.aspect = mask.array([self.aspect], mask=np.isnan([self.aspect]))

        # Verify ws
        if not isinstance(self.ws, (int, float, np.ndarray)):
            raise TypeError('ws must be either int, float, or numpy ndarray data types')
        elif isinstance(self.ws, np.ndarray):
            self.ws = mask.array(self.ws, mask=np.isnan(self.ws))
        else:
            self.ws = mask.array([self.ws], mask=np.isnan([self.ws]))

        # Verify wd
        if not isinstance(self.wd, (int, float, np.ndarray)):
            raise TypeError('wd must be either int, float, or numpy ndarray data types')
        elif isinstance(self.wd, np.ndarray):
            self.wd = mask.array(self.wd, mask=np.isnan(self.wd))
        else:
            self.wd = mask.array([self.wd], mask=np.isnan([self.wd]))

        # Verify ffmc
        if not isinstance(self.ffmc, (int, float, np.ndarray)):
            raise TypeError('ffmc must be either int, float, or numpy ndarray data types')
        elif isinstance(self.ffmc, np.ndarray):
            self.ffmc = mask.array(self.ffmc, mask=np.isnan(self.ffmc))
        else:
            self.ffmc = mask.array([self.ffmc], mask=np.isnan([self.ffmc]))

        # Verify bui
        if not isinstance(self.bui, (int, float, np.ndarray)):
            raise TypeError('bui must be either int, float, or numpy ndarray data types')
        elif isinstance(self.bui, np.ndarray):
            self.bui = mask.array(self.bui, mask=np.isnan(self.bui))
        else:
            self.bui = mask.array([self.bui], mask=np.isnan([self.bui]))

        # Verify pc
        if not isinstance(self.pc, (int, float, np.ndarray)):
            raise TypeError('pc must be either int, float, or numpy ndarray data types')
        elif isinstance(self.pc, np.ndarray):
            self.pc = mask.array(self.pc, mask=np.isnan(self.pc))
        else:
            self.pc = mask.array([self.pc], mask=np.isnan([self.pc]))

        # Verify pdf
        if not isinstance(self.pdf, (int, float, np.ndarray)):
            raise TypeError('pdf must be either int, float, or numpy ndarray data types')
        elif isinstance(self.pdf, np.ndarray):
            self.pdf = mask.array(self.pdf, mask=np.isnan(self.pdf))
        else:
            self.pdf = mask.array([self.pdf], mask=np.isnan([self.pdf]))

        # Verify gfl
        if not isinstance(self.gfl, (int, float, np.ndarray)):
            raise TypeError('gfl must be either int, float, or numpy ndarray data types')
        elif isinstance(self.gfl, np.ndarray):
            self.gfl = mask.array(self.gfl, mask=np.isnan(self.gfl))
        else:
            self.gfl = mask.array([self.gfl], mask=np.isnan([self.gfl]))

        # Verify gcf
        if not isinstance(self.gcf, (int, float, np.ndarray)):
            raise TypeError('gcf must be either int, float, or numpy ndarray data types')
        elif isinstance(self.gcf, np.ndarray):
            self.gcf = mask.array(self.gcf, mask=np.isnan(self.gcf))
        else:
            self.gcf = mask.array([self.gcf], mask=np.isnan([self.gcf]))

        # Verify out_request
        if not isinstance(self.out_request, (list, tuple)):
            raise TypeError('out_request must be a list or tuple')

    def invertWindAspect(self):
        """
        Function to invert/flip wind direction and aspect by 180 degrees
        :return: None
        """
        # Invert wind direction by 180 degrees
        self.wd = mask.where(self.wd > 180,
                             self.wd - 180,
                             self.wd + 180)

        # Invert aspect by 180 degrees
        self.aspect = mask.where(self.aspect > 180,
                                 self.aspect - 180,
                                 self.aspect + 180)

        return

    def calcSF(self) -> None:
        """
        Function to calculate the slope factor
        :return: None
        """
        self.sf = mask.where(self.slope < 70,
                             np.exp(3.533 * np.power((self.slope / 100), 1.2)),
                             10)

        return

    def calcISZ(self) -> None:
        """
        Function to calculate the initial spread index with no wind/no slope effects
        :return: None
        """
        # Calculate fine fuel moisture content in percent (default CFFBPS equation)
        self.m = (250 * (59.5 / 101) * (101 - self.ffmc)) / (59.5 + self.ffmc)

        # Calculate the FFMC function from ISI equation (fF)
        self.fF = (91.9 * np.exp(-0.1386 * self.m)) * (1 + (np.power(self.m, 5.31) / (4.93 * np.power(10, 7))))

        # Calculate no slope/no wind Initial Spread Index
        self.isz = 0.208 * self.fF

        return

    def calcFMC(self) -> None:
        """
        Function to calculate foliar moisture content (FMC) and foliar moisture effect (FME).
        :return: None
        """
        # Calculate normalized latitude
        self.latn = mask.where((self.elevation is not None) & (self.elevation > 0),
                               43 + (33.7 * np.exp(-0.0351 * (150 - np.abs(self.long)))),
                               46 + (23.4 * (np.exp(-0.036 * (150 - np.abs(self.long))))))
        # Calculate date of minimum foliar moisture content (D0)
        # This value is rounded to mimic the approach used in the cffdrs R package.
        self.d0 = mask.MaskedArray.round(mask.where((self.elevation is not None) & (self.elevation > 0),
                                                    142.1 * (self.lat / self.latn) + (0.0172 * self.elevation),
                                                    151 * (self.lat / self.latn)),
                                         0)

        # Calculate Julian date (Dj)
        self.dj = mask.where(np.isfinite(self.latn),
                             dt.strptime(str(self.wx_date), '%Y%m%d').timetuple().tm_yday,
                             0)

        # Number of days between Dj and D0 (ND)
        self.nd = np.absolute(self.dj - self.d0)

        # Calculate foliar moisture content (FMC)
        self.fmc = mask.where(self.nd < 30,
                              85 + (0.0189 * (self.nd ** 2)),
                              mask.where(self.nd < 50,
                                         32.9 + (3.17 * self.nd) - (0.0288 * (self.nd ** 2)),
                                         120))

        # Calculate foliar moisture effect (FME)
        self.fme = 1000 * np.power(1.5 - (0.00275 * self.fmc), 4) / (460 + (25.9 * self.fmc))

        return

    def calcROS(self) -> None:
        """
        Function to model the fire rate of spread (m/min).
        :return: None
        """

        def _calcISI_slopeWind() -> None:
            # Calculate the slope equivalent wind speed (for lower wind speeds)
            np.seterr(divide='ignore')
            self.wse1 = mask.where(self.fuel_type == self.ftype,
                                   (1 / 0.05039) * np.log(self.isf / (0.208 * self.fF)),
                                   self.wse1)
            np.seterr(divide='warn')

            # Calculate the slope equivalent wind speed (for higher wind speeds)
            self.wse2 = mask.where(self.fuel_type == self.ftype,
                                   mask.where(self.isf < (0.999 * 2.496 * self.fF),
                                              28 - (1 / 0.0818) * np.log(1 - (self.isf / (2.496 * self.fF))),
                                              112.45),
                                   self.wse2)

            # Assign slope equivalent wind speed
            self.wse = mask.where(self.fuel_type == self.ftype,
                                  mask.where(self.wse1 <= 40,
                                             self.wse1,
                                             self.wse2),
                                  self.wse)

            # Calculate vector magnitude in x-direction
            self.wsx = mask.where(self.fuel_type == self.ftype,
                                  (self.ws * np.sin(np.radians(self.wd))) +
                                  (self.wse * np.sin(np.radians(self.aspect))),
                                  self.wsx)

            # Calculate vector magnitude in y-direction
            self.wsy = mask.where(self.fuel_type == self.ftype,
                                  (self.ws * np.cos(np.radians(self.wd))) +
                                  (self.wse * np.cos(np.radians(self.aspect))),
                                  self.wsy)

            # Calculate the net effective wind speed
            self.wsv = mask.where(self.fuel_type == self.ftype,
                                  np.sqrt(np.power(self.wsx, 2) + np.power(self.wsy, 2)),
                                  self.wsv)

            # Calculate the net effective wind direction (RAZ)
            self.raz = mask.where(self.fuel_type == self.ftype,
                                  mask.where(self.wsx < 0,
                                             360 - np.degrees(np.arccos(self.wsy / self.wsv)),
                                             np.degrees(np.arccos(self.wsy / self.wsv))),
                                  self.raz)

            # Calculate the wind function of the ISI equation
            self.fW = mask.where(self.fuel_type == self.ftype,
                                 mask.where(self.wsv > 40,
                                            12 * (1 - np.exp(-0.0818 * (self.wsv - 28))),
                                            np.exp(0.05039 * self.wsv)),
                                 self.fW)

            # Calculate the new ISI with slope and wind effects
            self.isi = mask.where(self.fuel_type == self.ftype,
                                  0.208 * self.fW * self.fF,
                                  self.isi)

        # ### CFFBPS ROS models
        if self.ftype not in [10, 11]:
            # Get fuel type specific fixed rate of spread parameters
            self.a, self.b, self.c, self.q, self.bui0, self.be_max = self.rosParams[self.ftype]
            if self.ftype in [14, 15]:
                # ## Process O-1a/b fuel types...
                self.cf = mask.where(self.fuel_type == self.ftype,
                                     mask.where(self.gcf < 58.8,
                                                0.005 * (np.exp(0.061 * self.gcf) - 1),
                                                0.176 + (0.02 * (self.gcf - 58.8))),
                                     self.cf)

                # Calculate no slope/no wind rate of spread
                self.rsz = mask.where(self.fuel_type == self.ftype,
                                      self.a * np.power(1 - np.exp(-self.b * self.isz), self.c) * self.cf,
                                      self.rsz)

                # Calculate rate of spread with slope effect
                self.rsf = mask.where(self.fuel_type == self.ftype,
                                      self.rsz * self.sf,
                                      self.rsf)

                # Calculate initial spread index with slope effect
                self.isf = mask.where(self.fuel_type == self.ftype,
                                      mask.where(1 - np.power(self.rsf / (self.cf * self.a), 1 / self.c) >= 0.01,
                                                 np.log(
                                                     1 - np.power(self.rsf / (self.cf * self.a), 1 / self.c)) / -self.b,
                                                 np.log(0.01) / -self.b),
                                      self.isf)

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                self.rsi = mask.where(self.fuel_type == self.ftype,
                                      self.a * np.power(1 - np.exp(-self.b * self.isi), self.c) * self.cf,
                                      self.rsi)

                # Calculate Buildup Effect (BE)
                self.be = mask.where(self.fuel_type == self.ftype,
                                     1,
                                     self.be)

            elif self.ftype in [12, 13]:
                # ## Process M-3/4 fuel types...
                # Get D1 RSZ and ISF
                a_d1, b_d1, c_d1, q_d1, bui0_d1, be_max_d1 = self.rosParams[8]
                rsz_d1 = mask.where(self.fuel_type == self.ftype,
                                    a_d1 * np.power(1 - np.exp(-b_d1 * self.isz), c_d1),
                                    0)
                rsf_d1 = mask.where(self.fuel_type == self.ftype,
                                    rsz_d1 * self.sf,
                                    0)
                isf_d1 = mask.where(self.fuel_type == self.ftype,
                                    mask.where((1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) >= 0.01,
                                               np.log(1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) / -b_d1,
                                               np.log(0.01) / -b_d1),
                                    0)

                # Calculate no slope/no wind rate of spread
                self.rsz = mask.where(self.fuel_type == self.ftype,
                                      self.a * np.power(1 - np.exp(-self.b * self.isz), self.c),
                                      self.rsz)

                # Calculate rate of spread with slope effect
                self.rsf = mask.where(self.fuel_type == self.ftype,
                                      self.rsz * self.sf,
                                      self.rsf)

                # Calculate initial spread index with slope effect
                self.isf = mask.where(self.fuel_type == self.ftype,
                                      mask.where((1 - np.power(self.rsf / self.a, 1 / self.c)) >= 0.01,
                                                 (self.pdf / 100) *
                                                 (np.log(1 - np.power(self.rsf / self.a, 1 / self.c)) / -self.b) +
                                                 (1 - self.pdf / 100) *
                                                 isf_d1,
                                                 np.log(0.01) / -self.b),
                                      self.isf)

                # Calculate ISI and for D1
                # Calculate the slope equivalent wind speed (for lower wind speeds)
                np.seterr(divide='ignore')
                wse1_d1 = mask.where(self.fuel_type == self.ftype,
                                     (1 / 0.05039) * np.log(isf_d1 / (0.208 * self.fF)),
                                     0)
                np.seterr(divide='warn')

                # Calculate the slope equivalent wind speed (for higher wind speeds)
                wse2_d1 = mask.where(self.fuel_type == self.ftype,
                                     mask.where(isf_d1 < (0.999 * 2.496 * self.fF),
                                                28 - (1 / 0.0818) * np.log(1 - (isf_d1 / (2.496 * self.fF))),
                                                112.45),
                                     0)

                # Assign slope equivalent wind speed
                wse_d1 = mask.where(self.fuel_type == self.ftype,
                                    mask.where(wse1_d1 <= 40,
                                               wse1_d1,
                                               wse2_d1),
                                    0)

                # Calculate vector magnitude in x-direction
                wsx_d1 = mask.where(self.fuel_type == self.ftype,
                                    ((self.ws * np.sin(np.radians(self.wd))) +
                                     (wse_d1 * np.sin(np.radians(self.aspect)))),
                                    0)

                # Calculate vector magnitude in y-direction
                wsy_d1 = mask.where(self.fuel_type == self.ftype,
                                    ((self.ws * np.cos(np.radians(self.wd))) +
                                     (wse_d1 * np.cos(np.radians(self.aspect)))),
                                    0)

                # Calculate the net effective wind speed
                wsv_d1 = mask.where(self.fuel_type == self.ftype,
                                    np.sqrt(np.power(wsx_d1, 2) + np.power(wsy_d1, 2)),
                                    0)

                # Calculate the net effective wind direction (RAZ)
                # raz_d1 = mask.where(self.fuel_type == self.ftype,
                #                     mask.where(wsx_d1 < 0,
                #                                360 - np.degrees(np.arccos(wsy_d1 / wsv_d1)),
                #                                np.degrees(np.arccos(wsy_d1 / wsv_d1))),
                #                     0)

                # Calculate the wind function of the ISI equation
                fw_d1 = mask.where(self.fuel_type == self.ftype,
                                   mask.where(wsv_d1 > 40,
                                              12 * (1 - np.exp(-0.0818 * (wsv_d1 - 28))),
                                              np.exp(0.05039 * wsv_d1)),
                                   0)

                # Calculate the new ISI with slope and wind effects
                # isi_d1 = mask.where(self.fuel_type == self.ftype,
                #                     0.208 * fw_d1 * self.fF,
                #                     0)

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects for D1
                # Get D1 RSZ and ISF
                # TODO Verify that ISI here should be the M3/4 ISI, rather than the D1 ISI
                #  The CFS cffdrs R code uses the M3/4 (100% dead fir) ISI to calculate RSI for D1, which seems odd
                rsi_d1 = mask.where(self.fuel_type == self.ftype,
                                    a_d1 * np.power(1 - np.exp(-b_d1 * self.isi), c_d1),
                                    0)

                # Calculate rate of spread with slope and wind effects
                if self.ftype == 13:
                    self.rsi = mask.where(self.fuel_type == self.ftype,
                                          ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isi),
                                                                                self.c) +
                                           0.2 * (1 - self.pdf / 100) * rsi_d1),
                                          self.rsi)
                else:
                    self.rsi = mask.where(self.fuel_type == self.ftype,
                                          ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isi),
                                                                                self.c) +
                                           (1 - self.pdf / 100) * rsi_d1),
                                          self.rsi)

                # Calculate Buildup Effect (BE)
                self.be = mask.where(self.fuel_type == self.ftype,
                                     mask.where(self.bui == 0,
                                                0,
                                                np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))),
                                     self.be)

            else:
                # ## Process all other fuel types...
                # Calculate no slope/no wind rate of spread
                self.rsz = mask.where(self.fuel_type == self.ftype,
                                      self.a * np.power(1 - np.exp(-self.b * self.isz), self.c),
                                      self.rsz)

                # Calculate rate of spread with slope effect
                self.rsf = mask.where(self.fuel_type == self.ftype,
                                      self.rsz * self.sf,
                                      self.rsf)

                # Calculate initial spread index with slope effect
                isf_numer = mask.where(self.fuel_type == self.ftype,
                                       (1 - np.power(self.rsf / self.a, 1 / self.c)),
                                       0)
                np.seterr(divide='ignore')
                self.isf = mask.where(self.fuel_type == self.ftype,
                                      mask.where(isf_numer >= 0.01,
                                                 np.log(isf_numer) / -self.b,
                                                 np.log(0.01) / -self.b),
                                      self.isf)
                np.seterr(divide='warn')

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                self.rsi = mask.where(self.fuel_type == self.ftype,
                                      self.a * np.power(1 - np.exp(-self.b * self.isi), self.c),
                                      self.rsi)

                # Calculate Buildup Effect (BE)
                self.be = mask.where(self.fuel_type == self.ftype,
                                     mask.where(self.bui == 0,
                                                0,
                                                np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))),
                                     self.be)

        else:
            # ## Process M-1/2 fuel types...
            self.q, self.bui0, self.be_max = self.rosParams[self.ftype]

            # Calculate no slope/no wind rate of spread
            # Get C2 RSZ and ISF
            a_c2, b_c2, c_c2, q_c2, bui0_c2, be_max_c2 = self.rosParams[2]
            rsz_c2 = mask.where(self.fuel_type == self.ftype,
                                a_c2 * np.power(1 - np.exp(-b_c2 * self.isz), c_c2),
                                0)
            rsf_c2 = mask.where(self.fuel_type == self.ftype,
                                rsz_c2 * self.sf,
                                0)
            isf_c2 = mask.where(self.fuel_type == self.ftype,
                                mask.where((1 - np.power(rsf_c2 / a_c2, 1 / c_c2)) >= 0.01,
                                           np.log(1 - np.power(rsf_c2 / a_c2, 1 / c_c2)) / -b_c2,
                                           np.log(0.01) / -b_c2),
                                0)
            # Get D1 RSZ and ISF
            a_d1, b_d1, c_d1, q_d1, bui0_d1, be_max_d1 = self.rosParams[8]
            rsz_d1 = mask.where(self.fuel_type == self.ftype,
                                a_d1 * np.power(1 - np.exp(-b_d1 * self.isz), c_d1),
                                0)
            rsf_d1 = mask.where(self.fuel_type == self.ftype,
                                rsz_d1 * self.sf,
                                0)
            isf_d1 = mask.where(self.fuel_type == self.ftype,
                                mask.where((1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) >= 0.01,
                                           np.log(1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) / -b_d1,
                                           np.log(0.01) / -b_d1),
                                0)

            # Calculate initial spread index with slope effects
            self.isf = mask.where(self.fuel_type == self.ftype,
                                  (self.pc / 100) * isf_c2 + (1 - self.pc / 100) * isf_d1,
                                  self.isf)

            # Calculate slope effects on wind and ISI
            _calcISI_slopeWind()

            # Calculate rate of spread with slope and wind effects for C2 and D1
            # Get C2 RSI and ISI
            rsi_c2 = mask.where(self.fuel_type == self.ftype,
                                a_c2 * np.power(1 - np.exp(-b_c2 * self.isi), c_c2),
                                0)
            # Get D1 RSZ and ISF
            rsi_d1 = mask.where(self.fuel_type == self.ftype,
                                a_d1 * np.power(1 - np.exp(-b_d1 * self.isi), c_d1),
                                0)

            # Calculate rate of spread with slope and wind effects (RSI)
            if self.ftype == 11:
                self.rsi = mask.where(self.fuel_type == self.ftype,
                                      (self.pc / 100) * rsi_c2 + 0.2 * (1 - self.pc / 100) * rsi_d1,
                                      self.rsi)
            else:
                self.rsi = mask.where(self.fuel_type == self.ftype,
                                      (self.pc / 100) * rsi_c2 + (1 - self.pc / 100) * rsi_d1,
                                      self.rsi)

            # Calculate Buildup Effect (BE)
            self.be = mask.where(self.fuel_type == self.ftype,
                                 mask.where(self.bui == 0,
                                            0,
                                            np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))),
                                 self.be)

        # Ensure BE does not exceed be_max
        self.be = mask.where(self.fuel_type == self.ftype,
                             mask.where(self.be > self.be_max,
                                        self.be_max,
                                        self.be),
                             self.be)

        # Calculate Final ROS
        if self.ftype == 6:
            # C6 fuel type
            self.sfros = mask.where(self.fuel_type == self.ftype,
                                    self.rsi * self.be,
                                    self.sfros)
        else:
            # All other fuel types
            self.hfros = mask.where(self.fuel_type == self.ftype,
                                    self.rsi * self.be,
                                    self.hfros)
            if self.ftype == 9:
                # D2 fuel type
                self.hfros = mask.where(self.fuel_type == self.ftype,
                                        mask.where(self.bui < 70,
                                                   0,
                                                   self.hfros * 0.2),
                                        self.hfros)

        return

    def calcSFC(self) -> None:
        """
        Function to calculate forest floor consumption (FFC), woody fuel consumption (WFC),
        and total surface fuel consumption (SFC).
        :return: None
        """
        if self.ftype == 1:
            ffc = np.nan
            wfc = np.nan
            np.seterr(invalid='ignore', over='ignore')
            sfc = mask.where(self.ffmc > 84,
                             0.75 + 0.75 * np.sqrt(1 - np.exp(-0.23 * (self.ffmc - 84))),
                             0.75 - 0.75 * np.sqrt(1 - np.exp(0.23 * (self.ffmc - 84))))
            np.seterr(invalid='warn', over='warn')
        elif self.ftype == 2:
            ffc = np.nan
            wfc = np.nan
            sfc = 5 * (1 - np.exp(-0.0115 * self.bui))
        elif self.ftype in [3, 4]:
            ffc = np.nan
            wfc = np.nan
            sfc = 5 * np.power(1 - np.exp(-0.0164 * self.bui), 2.24)
        elif self.ftype in [5, 6]:
            ffc = np.nan
            wfc = np.nan
            sfc = 5 * np.power(1 - np.exp(-0.0149 * self.bui), 2.48)
        elif self.ftype == 7:
            ffc = 2 * (1 - np.exp(-0.104 * (self.ffmc - 70)))
            ffc = mask.where(ffc < 0,
                             0,
                             ffc)
            wfc = 1.5 * (1 - np.exp(-0.0201 * self.bui))
            sfc = ffc + wfc
        elif self.ftype in [8, 9]:
            ffc = np.nan
            wfc = np.nan
            sfc = 1.5 * (1 - np.exp(-0.0183 * self.bui))
        elif self.ftype in [10, 11]:
            c2_sfc = 5 * (1 - np.exp(-0.0115 * self.bui))
            d1_sfc = 1.5 * (1 - np.exp(-0.0183 * self.bui))
            ffc = np.nan
            wfc = np.nan
            sfc = ((self.pc / 100) * c2_sfc) + (((100 - self.pc) / 100) * d1_sfc)
        elif self.ftype in [12, 13]:
            ffc = np.nan
            wfc = np.nan
            sfc = 5 * (1 - np.exp(-0.0115 * self.bui))
        elif self.ftype in [14, 15]:
            ffc = np.nan
            wfc = np.nan
            sfc = self.gfl
        elif self.ftype == 16:
            ffc = 4 * (1 - np.exp(-0.025 * self.bui))
            wfc = 4 * (1 - np.exp(-0.034 * self.bui))
            sfc = ffc + wfc
        elif self.ftype == 17:
            ffc = 10 * (1 - np.exp(-0.013 * self.bui))
            wfc = 6 * (1 - np.exp(-0.06 * self.bui))
            sfc = ffc + wfc
        elif self.ftype == 18:
            ffc = 12 * (1 - np.exp(-0.0166 * self.bui))
            wfc = 20 * (1 - np.exp(-0.021 * self.bui))
            sfc = ffc + wfc
        elif self.ftype == 19:
            ffc = 0
            wfc = 0
            sfc = 0
        elif self.ftype == 20:
            ffc = 0
            wfc = 0
            sfc = 0
        else:
            ffc = np.nan
            wfc = np.nan
            sfc = np.nan

        # Assign FFC
        self.ffc = mask.where(self.fuel_type == self.ftype,
                              ffc,
                              self.ffc)
        # Assign WFC
        self.wfc = mask.where(self.fuel_type == self.ftype,
                              wfc,
                              self.wfc)
        # Assign SFC
        self.sfc = mask.where(self.fuel_type == self.ftype,
                              sfc,
                              self.sfc)

        return

    def getCBH_CFL(self) -> None:
        """
        Function to get the default CFFBPS canopy base height (CBH) and
        canopy fuel load (CFL) values for a specified fuel type.
        :return: None
        """
        # Get canopy base height (CBH) for fuel type
        self.cbh = mask.where(self.fuel_type == self.ftype,
                              self.fbpCBH_CFL_HT_LUT.get(self.ftype)[0],
                              self.cbh)
        # Get canopy fuel load (CFL) for fuel type
        self.cfl = mask.where(self.fuel_type == self.ftype,
                              self.fbpCBH_CFL_HT_LUT.get(self.ftype)[1],
                              self.cbh)

        return

    def calcCSFI(self) -> None:
        """
        Function to calculate the critical surface fire intensity (CSFI).
        :return: None
        """
        # Calculate critical surface fire intensity (CSFI)
        self.csfi = mask.where(self.fuel_type < 19,
                               np.power(0.01 * self.cbh * (460 + (25.9 * self.fmc)), 1.5),
                               0)

        return

    def calcRSO(self) -> None:
        """
        Function to calculate the critical surface fire rate of spread (RSO).
        :return: None
        """
        # Calculate critical surface fire rate of spread (RSO)
        self.rso = mask.where(self.sfc > 0,
                              self.csfi / (300.0 * self.sfc),
                              0)

        return

    def calcCFB(self) -> None:
        """
        Function calculates crown fraction burned using equation in Forestry Canada Fire Danger Group (1992)
        :return: None
        """
        if self.ftype == 6:
            # For C-6 fuel type
            self.cfb = mask.where(self.fuel_type == self.ftype,
                                  mask.where((self.sfros - self.rso) < -3086,
                                             0,
                                             1 - np.exp(-0.23 * (self.sfros - self.rso))),
                                  self.cfb)
        else:
            # For all other fuel types
            self.cfb = mask.where(self.fuel_type == self.ftype,
                                  mask.where((self.hfros - self.rso) < -3086,
                                             0,
                                             1 - np.exp((-0.23 * (self.hfros - self.rso)).astype(np.float32))),
                                  self.cfb)

        # Replace negative values with 0
        self.cfb = mask.where(self.cfb < 0,
                              0,
                              self.cfb)

        return

    def calcFireType(self) -> None:
        """
        Function to calculate fire type (1: surface, 2: intermittent crown, 3: active crown)
        :return: None
        """
        self.fire_type = mask.where((self.fuel_type < 19),
                                    mask.where(self.cfb <= 0.1,
                                               # Surface fire
                                               1,
                                               mask.where((self.cfb > 0.1) & (self.cfb < 0.9),
                                                          # Intermittent crown fire
                                                          2,
                                                          mask.where(self.cfb >= 0.9,
                                                                     # Active crown fire
                                                                     3,
                                                                     # No fire type
                                                                     0))),
                                    0)

        return

    def calcCFC(self) -> None:
        """
        Function calculates crown fuel consumed (kg/m^2).
        :return: None
        """
        self.cfc = mask.where((self.fuel_type == 10) | (self.fuel_type == 11),
                              self.cfb * self.cfl * self.pc / 100,
                              mask.where((self.fuel_type == 12) | (self.fuel_type == 13),
                                         self.cfb * self.cfl * self.pdf / 100,
                                         self.cfb * self.cfl))

        return

    def calcC6hfros(self) -> None:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type
        :returns: None
        """
        self.cfros = mask.where(self.fuel_type == 6,
                                mask.where(self.cfc == 0,
                                           0,
                                           60 * np.power(1 - np.exp(-0.0497 * self.isi), 1) * (self.fme / 0.778237)),
                                self.cfros)

        self.hfros = mask.where(self.fuel_type == 6,
                                self.sfros + (self.cfb * (self.cfros - self.sfros)),
                                self.hfros)

        return

    def calcTFC(self) -> None:
        """
        Function to calculate total fuel consumed (kg/m^2)
        :return: None
        """
        self.tfc = self.sfc + self.cfc

        return

    def calcHFI(self) -> None:
        """
        Function to calculate fire type, total fuel consumption, and head fire intensity
        :returns: None
        """
        self.hfi = 300 * self.hfros * self.tfc

        return

    def getOutputs(self, out_request: list[str]) -> list[any]:
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
            'RSZ': self.rsz,  # Surface spread rate with zero wind on level terrain
            'SF': self.sf,  # Slope factor
            'RSF': self.rsf,  # Spread rate with zero wind, upslope
            'ISF': self.isf,  # ISI, with zero wind upslope
            'RSI': self.rsi,  # Initial spread rate without BUI effect
            'WSE1': self.wse1,  # Original slope equivalent wind speed value for cases where WSE1 <= 40
            'WSE2': self.wse2,  # New slope equivalent sind speed value for cases where WSE1 > 40
            'WSE': self.wse,  # Slope equivalent wind speed
            'WSX': self.wsx,  # Net vectorized wind speed in the x-direction
            'WSY': self.wsy,  # Net vectorized wind speed in the y-direction
            'WSV': self.wsv,  # Net vectorized wind speed
            'RAZ': self.raz,  # Net vectorized wind direction

            # BUI effect variables
            'q': self.q,  # Proportion of maximum rate of spread at BUI equal to 50
            'bui0': self.bui0,  # Average BUI for each fuel type
            'BE': self.be,  # Buildup effect on spread rate
            'be_max': self.be_max,  # Maximum allowable BE value

            # Surface fuel variables
            'ffc': self.ffc,  # Estimated forest floor consumption
            'wfc': self.wfc,  # Estimated woody fuel consumption
            'sfc': self.sfc,  # Estimated total surface fuel consumption

            # Foliar moisture content variables
            'latn': self.latn,  # Normalized latitude
            'dj': self.dj,  # Julian date of day being modelled
            'd0': self.d0,  # Julian date of minimum foliar moisture content
            'nd': self.nd,  # number of days between modelled fire date and d0
            'fmc': self.fmc,  # foliar moisture content
            'fme': self.fme,  # foliar moisture effect

            # Critical crown fire threshold variables
            'csfi': self.csfi,  # critical intensity (kW/m)
            'rso': self.rso,  # critical rate of spread (m/min)

            # Crown fuel parameters
            'cbh': self.cbh,  # Height to live crown base (m)
            'cfb': self.cfb,  # Crown fraction burned (proportion, value ranging from 0-1)
            'cfl': self.cfl,  # Crown fuel load (kg/m^2)
            'cfc': self.cfc  # Crown fuel consumed
        }

        if self.return_array:
            return [fbp_params.get(var).data if fbp_params.get(var, None) is not None
                    else 'Invalid output variable' for var in out_request]
        else:
            return [fbp_params.get(var).data[0] if fbp_params.get(var, None) is not None
                    else 'Invalid output variable' for var in out_request]

    def runFBP(self) -> list[any]:
        """
        Function to automatically run CFFBPS modelling
        """
        # Model fire behavior with CFFBPS
        # print('Inverting wind direction and aspect')
        self.invertWindAspect()
        # print('Calculating slope factor')
        self.calcSF()
        # print('Calculating zero slope & zero wind ISI')
        self.calcISZ()
        # print('Calculating foliar moisture content')
        self.calcFMC()
        for self.ftype in [ftype for ftype in np.unique(self.fuel_type)
                           if ftype in list(self.rosParams.keys())]:
            # print(f'Processing {fbpFTCode_NumToAlpha_LUT.get(self.ftype)} fuel type')
            # print('\tCalculating ROS')
            self.calcROS()
            # print('\tCalculating surface fuel consumption')
            self.calcSFC()
            # print('\tCalculating canopy base height and canopy fuel load')
            self.getCBH_CFL()
        # print('Calculating critical surface fire intensity')
        self.calcCSFI()
        # print('Calculating critical surface fire rate of spread')
        self.calcRSO()
        # print(f'Calculating crown fraction burned')
        for self.ftype in [ftype for ftype in np.unique(self.fuel_type)
                           if ftype in list(self.rosParams.keys())]:
            # print(f'\tProcessing {fbpFTCode_NumToAlpha_LUT.get(self.ftype)} fuel type')
            self.calcCFB()
        # print('Calculating fire type')
        self.calcFireType()
        # print('Calculating crown fuel consumed')
        self.calcCFC()
        # print('Calculating C6 head fire rate of spread')
        self.calcC6hfros()
        # print('Calculating total fuel consumption')
        self.calcTFC()
        # print('Calculating head fire intensity')
        self.calcHFI()

        # print('<< Returning requested values >>\n')
        return self.getOutputs(self.out_request)


def _testFBP(wx_date: int,
             lat: Union[float, int, np.ndarray],
             long: Union[float, int, np.ndarray],
             elevation: Union[float, int, np.ndarray],
             slope: Union[float, int, np.ndarray],
             aspect: Union[float, int, np.ndarray],
             ws: Union[float, int, np.ndarray],
             wd: Union[float, int, np.ndarray],
             ffmc: Union[float, int, np.ndarray],
             bui: Union[float, int, np.ndarray],
             pc: Optional[Union[float, int, np.ndarray]] = 50,
             pdf: Optional[Union[float, int, np.ndarray]] = 35,
             gfl: Optional[Union[float, int, np.ndarray]] = 0.35,
             gcf: Optional[Union[float, int, np.ndarray]] = 80,
             out_request: Optional[list[str]] = None,
             out_folder: Optional[str] = None,
             convertGridCodes: bool = False) -> None:
    """
    Function to test the cffbps module with various input types
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
    :param out_folder: Location to save test rasters (Default: <location of script>\Test_Data\Outputs)
    :param convertGridCodes: Convert from cffdrs R fuel type grid codes to codes used in this module (default: False)
    :return: None
    """
    import ProcessRasters as pr
    import generate_test_fbp_rasters as genras

    # Create fuel type list
    fuel_type_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4',
                      'O1a', 'O1b', 'S1', 'S2', 'S3']

    # Put inputs into list
    input_data = [wx_date, lat, long,
                  elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, out_request]

    # ### Test non-raster modelling
    for ft in fuel_type_list:
        print(ft, FBP(*([fbpFTCode_AlphaToNum_LUT.get(ft)] + input_data)).runFBP())

    # ### Test array modelling
    print(FBP(*([np.array(fuel_type_list)] + input_data)).runFBP())

    # ### Test raster modelling
    # Get test folders
    input_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Inputs')
    if out_folder is None:
        output_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Outputs')
    else:
        output_folder = out_folder

    # Generate test raster datasets using user-provided input values
    genras.gen_test_data(*input_data[:-2])

    # Get input dataset paths
    fuel_type_path = os.path.join(input_folder, 'FuelType.tif')
    lat_path = os.path.join(input_folder, 'LAT.tif')
    long_path = os.path.join(input_folder, 'LONG.tif')
    elev_path = os.path.join(input_folder, 'ELV.tif')
    slope_path = os.path.join(input_folder, 'GS.tif')
    aspect_path = os.path.join(input_folder, 'Aspect.tif')
    ws_path = os.path.join(input_folder, 'WS.tif')
    wd_path = os.path.join(input_folder, 'WD.tif')
    ffmc_path = os.path.join(input_folder, 'FFMC.tif')
    bui_path = os.path.join(input_folder, 'BUI.tif')
    pc_path = os.path.join(input_folder, 'PC.tif')
    pdf_path = os.path.join(input_folder, 'PDF.tif')
    gfl_path = os.path.join(input_folder, 'GFL.tif')
    gcf_path = os.path.join(input_folder, 'cc.tif')

    # Create a reference raster profile for final raster outputs
    ref_ras_profile = pr.getRaster(gfl_path).profile

    # Get input dataset arrays
    fuel_type_array = pr.getRaster(fuel_type_path).read()
    lat_array = pr.getRaster(lat_path).read()
    long_array = pr.getRaster(long_path).read()
    elev_array = pr.getRaster(elev_path).read()
    slope_array = pr.getRaster(slope_path).read()
    aspect_array = pr.getRaster(aspect_path).read()
    ws_array = pr.getRaster(ws_path).read()
    wd_array = pr.getRaster(wd_path).read()
    ffmc_array = pr.getRaster(ffmc_path).read()
    bui_array = pr.getRaster(bui_path).read()
    pc_array = pr.getRaster(pc_path).read()
    pdf_array = pr.getRaster(pdf_path).read()
    gfl_array = pr.getRaster(gfl_path).read()
    gcf_array = pr.getRaster(gcf_path).read()

    # Convert from cffdrs R fuel type grid codes to the grid codes used in this module
    if convertGridCodes:
        fuel_type_array = convert_grid_codes(fuel_type_array)

    wsv, raz, fire_type, hfros, hfi, ffc, wfc, sfc = FBP(
        fuel_type=fuel_type_array, wx_date=wx_date, lat=lat_array, long=long_array,
        elevation=elev_array, slope=slope_array, aspect=aspect_array,
        ws=ws_array, wd=wd_array, ffmc=ffmc_array, bui=bui_array,
        pc=pc_array, pdf=pdf_array, gfl=gfl_array, gcf=gcf_array,
        out_request=['WSV', 'RAZ', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc']).runFBP()

    # Get output dataset paths
    wsv_out = os.path.join(output_folder, 'wsv.tif')
    raz_out = os.path.join(output_folder, 'raz.tif')
    fire_type_out = os.path.join(output_folder, 'fire_type.tif')
    hfros_out = os.path.join(output_folder, 'hfros.tif')
    hfi_out = os.path.join(output_folder, 'hfi.tif')
    ffc_out = os.path.join(output_folder, 'ffc.tif')
    wfc_out = os.path.join(output_folder, 'wfc.tif')
    sfc_out = os.path.join(output_folder, 'sfc.tif')

    output_list = [
        (wsv, wsv_out),
        (raz, raz_out),
        (fire_type, fire_type_out),
        (hfros, hfros_out),
        (hfi, hfi_out),
        (ffc, ffc_out),
        (wfc, wfc_out),
        (sfc, sfc_out),
    ]

    for dset, path in output_list:
        # Save output datasets
        pr.arrayToRaster(array=dset,
                         out_file=path,
                         ras_profile=ref_ras_profile,
                         data_type=np.float64)


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
    _pc = 50
    _pdf = 50
    _gfl = 0.35
    _gcf = 80
    _out_request = ['latn', 'd0', 'dj', 'nd', 'fmc', 'fme', 'csfi', 'rso', 'hfros', 'hfi']
    _out_folder = None
    _convertGridCodes = False

    # Test the FBP functions
    _testFBP(wx_date=_wx_date, lat=_lat, long=_long,
             elevation=_elevation, slope=_slope, aspect=_aspect,
             ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
             pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf,
             out_request=_out_request,
             out_folder=_out_folder,
             convertGridCodes=_convertGridCodes)
