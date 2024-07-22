# -*- coding: utf-8 -*-
"""
Created on Mon July 22 10:00:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

from typing import Union, Optional
import numpy as np
from datetime import datetime as dt

# TODO - Revise FBP Class to accept raster and non-raster data
#  The code is currently specific to rasters (but untested)


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
    :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
        inFT = fuel type to be changed
        subFT = fuel type replacing inFT
    :returns:
        Tuple of values requested through out_request parameter. Default values are fire_type, hfros, and hfi.
    """

    def __init__(self,
                 fuel_type: np.ndarray,
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
                 pc: Optional[float, int, np.ndarray] = 50,
                 pdf: Optional[float, int, np.ndarray] = 35,
                 gfl: Optional[float, int, np.ndarray] = 0.35,
                 gcf: Optional[float, int, np.ndarray] = 80,
                 out_request: Optional[list[str]] = None):
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

        # Initialize weather parameters
        self.isi = None
        self.m = None
        self.fF = None
        self.fW = None

        # Initialize slope effect parameters
        self.a = None
        self.b = None
        self.c = None
        self.rsz = None
        self.isz = None
        self.sf = None
        self.rsf = None
        self.isf = None
        self.rsi = None
        self.wse1 = None
        self.wse2 = None
        self.wse = None
        self.wsx = None
        self.wsy = None
        self.wsv = None
        self.raz = None

        # Initialize BUI effect parameters
        self.q = None
        self.bui0 = None
        self.be = None
        self.be_max = None

        # Initialize surface parameters
        self.cf = 0
        self.ffc = 0
        self.wfc = 0
        self.sfc = 0
        self.rss = None

        # Initialize foliar moisture content parameters
        self.latn = None
        self.dj = None
        self.d0 = None
        self.nd = None
        self.fmc = None
        self.fme = None

        # Initialize crown and total fuel consumed parameters
        self.cbh = None
        self.csfi = None
        self.rso = None
        self.rsc = None
        self.cfb = None
        self.cfl = None
        self.cfc = 0
        self.tfc = 0

        # Initialize default CFFBPS output parameters
        self.fire_type = None
        self.hfros = None
        self.hfi = None

        # Initialize other parameters
        self.ftype = None
        self.sfros = None
        self.cfros = None

        # ### Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = ['cbh', 'fmc']
        self.rsoVarList = ['csfi', 'sfc']
        self.cfbVarList = ['cfros', 'rso']
        self.cfcVarList = ['cfb', 'cfl']
        self.cfiVarList = ['cfros', 'cfc']

        # CFFBPS Fuel Type Code Lookup Table
        self.fbpFTCode_LUT = {
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
            19: 'N',  # N (non fuel)
        }
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

    def invertWindAspect(self):
        """
        Function to invert/flip wind direction and aspect by 180 degrees
        :return: None
        """
        # Invert wind direction by 180 degrees
        self.wd = np.where(self.wd > 180,
                           self.wd - 180,
                           self.wd + 180)

        # Invert aspect by 180 degrees
        self.aspect = np.where(self.aspect > 180,
                               self.aspect - 180,
                               self.aspect + 180)

        return

    def calcSF(self) -> None:
        """
        Function to calculate the slope factor
        :return: None
        """
        self.sf = np.where(self.slope < 70,
                           np.exp(3.533 * np.power((self.slope / 100), 1.2)),
                           10)

        return

    def calcISZ(self) -> None:
        """
        Function to calculate the initial spread index with no wind/no slope effects
        :return: None
        """
        # Calculate fine fuel moisture content in percent (default CFFBPS equation)
        self.m = (147.2 * (101 - self.ffmc)) / (59.5 + self.ffmc)

        # Calculate the FFMC function from ISI equation
        self.fF = (91.9 * np.exp(-0.1386 * self.m)) * (1 + (np.power(self.m, 5.31) / (4.93 * np.power(10, 7))))

        # Calculate no slope/no wind Initial Spread Index
        self.isz = 0.208 * self.fF

        return

    def calcFMC(self) -> None:
        """
        Function returns a tuple of foliar moisture content (FMC) and foliar moisture effect (FME).
        :return: None
        """
        self.latn = np.where(self.elevation > 0,
                             43 + (33.7 * np.exp(-0.0351 * (150 - np.abs(self.long)))),
                             46 + (23.4 * (np.exp(-0.036 * (150 - np.abs(self.long))))))
        self.d0 = np.where(self.elevation > 0,
                           142.1 * (self.lat / self.latn) + (0.0172 * self.elevation),
                           151 * (self.lat / self.latn))

        try:
            self.dj = dt.strptime(str(self.wx_date), '%Y%m%d%H').timetuple().tm_yday
        except:
            try:
                self.dj = dt.strptime(str(self.wx_date), '%Y%m%d').timetuple().tm_yday
            except:
                raise ValueError('Invalid Weather Date format')

        # Number of days between current date and day 0
        self.nd = abs(self.dj - self.d0)

        if self.nd < 30:
            self.fmc = 85 + (0.0189 * self.nd ** 2)
        elif self.nd < 50:
            self.fmc = 32.9 + (3.17 * self.nd) - (0.0288 * self.nd ** 2)
        else:
            self.fmc = 120

        self.fme = 1000 * np.power(1.5 - (0.00275 * self.fmc), 4) / (460 + (25.9 * self.fmc))

        return

    def calcROS(self) -> None:
        """
        Function to model the fire rate of spread (m/min).
        :return: None
        """

        def _calcISI_slopeWind() -> None:
            # Calculate the slope equivalent wind speed (for lower wind speeds)
            self.wse1 = (1 / 0.05039) * np.log(self.isf / (0.208 * self.fF))

            # Calculate the slope equivalent wind speed (for higher wind speeds)
            self.wse2 = np.where(self.isf < (0.999 * 2.496 * self.fF),
                                 28 - (1 / 0.0818) * np.log(1 - (self.isf / (2.496 * self.fF))),
                                 112.45)

            # Assign slope equivalent wind speed
            self.wse = np.where(self.wse1 <= 40,
                                self.wse1,
                                self.wse2)

            # Calculate vector magnitude in x-direction
            self.wsx = (self.ws * np.sin(np.radians(self.wd))) + (self.wse * np.sin(np.radians(self.aspect)))

            # Calculate vector magnitude in y-direction
            self.wsy = (self.ws * np.cos(np.radians(self.wd))) + (self.wse * np.cos(np.radians(self.aspect)))

            # Calculate the net effective wind speed
            self.wsv = np.sqrt(np.power(self.wsx, 2) + np.power(self.wsy, 2))

            # Calculate the net effective wind direction (RAZ)
            self.raz = np.where(self.wsx < 0,
                                360 - np.degrees(np.arccos(self.wsy / self.wsv)),
                                np.degrees(np.arccos(self.wsy / self.wsv)))

            # Calculate the wind function of the ISI equation
            self.fW = np.where(self.wsv > 40,
                               12 * (1 - np.exp(-0.0818 * (self.wsv - 28))),
                               np.exp(0.05039 * self.wsv))

            # Calculate the new ISI with slope and wind effects
            self.isi = 0.208 * self.fW * self.fF

        # ### CFFBPS ROS models
        if self.ftype not in [10, 11]:
            # Get fuel type specific fixed rate of spread parameters
            self.a, self.b, self.c, self.q, self.bui0, self.be_max = self.rosParams[self.ftype]
            if self.ftype in [14, 15]:
                # ## Process O-1a/b fuel types...
                if self.gcf < 58.8:
                    self.cf = 0.005 * (np.exp(0.061 * self.gcf) - 1)
                else:
                    self.cf = 0.176 + (0.02 * (self.gcf - 58.8))

                # Calculate no slope/no wind rate of spread
                self.rsz = self.a * np.power(1 - np.exp(-self.b * self.isz), self.c) * self.cf

                # Calculate rate of spread with slope effect
                self.rsf = self.rsz * self.sf

                # Calculate initial spread index with slope effect
                self.isf = np.where(1 - np.power(self.rsf / (self.cf * self.a), 1 / self.c) >= 0.01,
                                    np.log(1 - np.power(self.rsf / (self.cf * self.a), 1 / self.c)) / -self.b,
                                    np.log(0.01) / -self.b)

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                self.rsi = self.a * np.power(1 - np.exp(-self.b * self.isi), self.c) * self.cf

                # Calculate Buildup Effect (BE)
                self.be = 1

            elif self.ftype in [12, 13]:
                # ## Process M-3/4 fuel types...
                # Get D1 RSZ and ISF
                a_d1, b_d1, c_d1, q_d1, bui0_d1, be_max_d1 = self.rosParams[8]
                rsz_d1 = a_d1 * np.power(1 - np.exp(-b_d1 * self.isz), c_d1)
                rsf_d1 = rsz_d1 * self.sf
                if (1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) >= 0.01:
                    isf_d1 = np.log(1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) / -b_d1
                else:
                    isf_d1 = np.log(0.01) / -b_d1

                # Calculate no slope/no wind rate of spread
                if self.ftype == 13:
                    self.rsz = ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isz), self.c) +
                                0.2 * (1 - self.pdf / 100) * rsz_d1)
                else:
                    self.rsz = ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isz), self.c) +
                                (1 - self.pdf / 100) * rsz_d1)

                # Calculate rate of spread with slope effect
                self.rsf = self.rsz * self.sf

                # Calculate initial spread index with slope effect
                if (1 - np.power(self.rsf / self.a, 1 / self.c)) >= 0.01:
                    self.isf = ((self.pdf / 100) *
                                (np.log(1 - np.power(self.rsf / self.a, 1 / self.c)) / -self.b) +
                                (1 - self.pdf / 100) *
                                isf_d1)
                else:
                    self.isf = np.log(0.01) / -self.b

                # Calculate ISI and for D1
                # Calculate the slope equivalent wind speed (for lower wind speeds)
                wse1_d1 = (1 / 0.05039) * np.log(isf_d1 / (0.208 * self.fF))

                # Calculate the slope equivalent wind speed (for higher wind speeds)
                if isf_d1 < (0.999 * 2.496 * self.fF):
                    wse2_d1 = 28 - (1 / 0.0818) * np.log(1 - (isf_d1 / (2.496 * self.fF)))
                else:
                    wse2_d1 = 112.45

                # Assign slope equivalent wind speed
                if wse1_d1 <= 40:
                    wse_d1 = wse1_d1
                else:
                    wse_d1 = wse2_d1

                # Calculate vector magnitude in x-direction
                wsx_d1 = ((self.ws * np.sin(np.radians(self.wd))) +
                          (wse_d1 * np.sin(np.radians(self.aspect))))

                # Calculate vector magnitude in y-direction
                wsy_d1 = ((self.ws * np.cos(np.radians(self.wd))) +
                          (wse_d1 * np.cos(np.radians(self.aspect))))

                # Calculate the net effective wind speed
                wsv_d1 = np.sqrt(np.power(wsx_d1, 2) + np.power(wsy_d1, 2))

                # Calculate the net effective wind direction (RAZ)
                raz_d1 = np.where(wsx_d1 < 0,
                                  360 - np.degrees(np.arccos(wsy_d1 / wsv_d1)),
                                  np.degrees(np.arccos(wsy_d1 / wsv_d1)))

                # Calculate the wind function of the ISI equation
                fw_d1 = np.where(wsv_d1 > 40,
                                 12 * (1 - np.exp(-0.0818 * (wsv_d1 - 28))),
                                 np.exp(0.05039 * wsv_d1))

                # Calculate the new ISI with slope and wind effects
                isi_d1 = 0.208 * fw_d1 * self.fF

                # Calculate rate of spread with slope and wind effects for D1
                # Get D1 RSZ and ISF
                rsi_d1 = a_d1 * np.power(1 - np.exp(-b_d1 * isi_d1), c_d1)

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                if self.ftype == 13:
                    self.rsi = ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isi), self.c) +
                                0.2 * (1 - self.pdf / 100) * rsi_d1)
                else:
                    self.rsi = ((self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.isi), self.c) +
                                (1 - self.pdf / 100) * rsi_d1)

                # Calculate Buildup Effect (BE)
                if self.bui == 0:
                    self.be = 0
                else:
                    self.be = np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))

            else:
                # ## Process all other fuel types...
                # Calculate no slope/no wind rate of spread
                self.rsz = self.a * np.power(1 - np.exp(-self.b * self.isz), self.c)

                # Calculate rate of spread with slope effect
                self.rsf = self.rsz * self.sf

                # Calculate initial spread index with slope effect
                if (1 - np.power(self.rsf / self.a, 1 / self.c)) >= 0.01:
                    self.isf = np.log(1 - np.power(self.rsf / self.a, 1 / self.c)) / -self.b
                else:
                    self.isf = np.log(0.01) / -self.b

                # Calculate slope effects on wind and ISI
                _calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                self.rsi = self.a * np.power(1 - np.exp(-self.b * self.isi), self.c)

                # Calculate Buildup Effect (BE)
                if self.bui == 0:
                    self.be = 0
                else:
                    self.be = np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))

        else:
            # ## Process M-1/2 fuel types...
            self.q, self.bui0, self.be_max = self.rosParams[self.ftype]

            # Calculate no slope/no wind rate of spread
            # Get C2 RSZ and ISF
            a_c2, b_c2, c_c2, q_c2, bui0_c2, be_max_c2 = self.rosParams[2]
            rsz_c2 = a_c2 * np.power(1 - np.exp(-b_c2 * self.isz), c_c2)
            rsf_c2 = rsz_c2 * self.sf
            if (1 - np.power(rsf_c2 / a_c2, 1 / c_c2)) >= 0.01:
                isf_c2 = np.log(1 - np.power(rsf_c2 / a_c2, 1 / c_c2)) / -b_c2
            else:
                isf_c2 = np.log(0.01) / -b_c2
            # Get D1 RSZ and ISF
            a_d1, b_d1, c_d1, q_d1, bui0_d1, be_max_d1 = self.rosParams[8]
            rsz_d1 = a_d1 * np.power(1 - np.exp(-b_d1 * self.isz), c_d1)
            rsf_d1 = rsz_d1 * self.sf
            if (1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) >= 0.01:
                isf_d1 = np.log(1 - np.power(rsf_d1 / a_d1, 1 / c_d1)) / -b_d1
            else:
                isf_d1 = np.log(0.01) / -b_d1

            # Calculate initial spread index with slope effects
            self.isf = (self.pc / 100) * isf_c2 + (1 - self.pc / 100) * isf_d1

            # Calculate slope effects on wind and ISI
            _calcISI_slopeWind()

            # Calculate rate of spread with slope and wind effects for C2 and D1
            # Get C2 RSI and ISI
            rsi_c2 = a_c2 * np.power(1 - np.exp(-b_c2 * self.isi), c_c2)
            # Get D1 RSZ and ISF
            rsi_d1 = a_d1 * np.power(1 - np.exp(-b_d1 * self.isi), c_d1)

            # Calculate rate of spread with slope and wind effects (RSI)
            if self.ftype == 11:
                self.rsi = (self.pc / 100) * rsi_c2 + 0.2 * (1 - self.pc / 100) * rsi_d1
            else:
                self.rsi = (self.pc / 100) * rsi_c2 + (1 - self.pc / 100) * rsi_d1

            # Calculate Buildup Effect (BE)
            if self.bui == 0:
                self.be = 0
            else:
                self.be = np.exp(50 * np.log(self.q) * ((1 / self.bui) - (1 / self.bui0)))

        # Ensure BE does not exceed be_max
        if self.be > self.be_max:
            self.be = self.be_max

        # Calculate Final ROS
        if self.ftype == 6:
            # C6 fuel type
            self.sfros = self.rsi * self.be
        else:
            # All other fuel types
            self.hfros = self.rsi * self.be
            if self.ftype == 9:
                # D2 fuel type
                self.hfros *= 0.2

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
            sfc = np.where(self.ffmc > 84,
                           0.75 + 0.75 * np.sqrt(1 - np.exp(-0.23 * (self.ffmc - 84))),
                           0.75 - 0.75 * np.sqrt(1 - np.exp(0.23 * (self.ffmc - 84))))
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
            ffc = np.where(ffc < 0,
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
        else:
            ffc = np.nan
            wfc = np.nan
            sfc = np.nan

        # Assign FFC
        self.ffc = np.where(self.fuel_type == self.ftype,
                            ffc,
                            self.ffc)

        # Assign WFC
        self.wfc = np.where(self.fuel_type == self.ftype,
                            wfc,
                            self.wfc)

        # Assign SFC
        self.sfc = np.where(self.fuel_type == self.ftype,
                            sfc,
                            self.sfc)

        return

    def getCBH_CFL(self) -> None:
        """
        Function to get the default CFFBPS canopy base height (CBH) and
        canopy fuel load (CFL) values for a specified fuel type.
        :return: None
        """
        # Get CBH for fuel type
        self.cbh = np.where(self.fuel_type == self.ftype,
                            self.fbpCBH_CFL_HT_LUT.get(self.ftype)[0],
                            self.cbh)
        # Get CFL for fuel type
        self.cfl = np.where(self.fuel_type == self.ftype,
                            self.fbpCBH_CFL_HT_LUT.get(self.ftype)[1],
                            self.cbh)

        return

    def calcCSFI(self) -> None:
        """
        Function to calculate the critical surface fire intensity (CSFI).
        :return: None
        """
        # Calculate critical surface fire intensity
        self.csfi = np.power(0.01 * self.cbh * (460 + (25.9 * self.fmc)), 1.5)

        return

    def calcRSO(self) -> None:
        """
        Function to calculate the critical surface fire rate of spread (RSO).
        :return: None
        """
        # Calculate critical surface fire rate of spread
        self.rso = np.where(self.sfc > 0,
                            self.csfi / (300 * self.sfc),
                            0)

        return

    def calcCFB(self) -> None:
        """
        Function calculates crown fraction burned using equation in Forestry Canada Fire Danger Group (1992)
        :return: None
        """
        self.cfb = np.where(self.fuel_type == 6,
                            # For C-6 fuel type
                            np.where((self.sfros - self.rso) < -3086,
                                     0,
                                     1 - np.exp(-0.23 * (self.sfros - self.rso))),
                            # For all other fuel types
                            np.where((self.hfros - self.rso) < -3086,
                                     0,
                                     1 - np.exp(-0.23 * (self.sfros - self.rso))))

        # Replace negative values with 0
        self.cfb = np.where(self.cfb < 0,
                            0,
                            self.cfb)

        return

    def calcFireType(self) -> None:
        """
        Function to calculate fire type (1: surface, 2: intermittent crown, 3: active crown)
        :return: None
        """
        self.fire_type = np.where(self.cfb <= 0.1,
                                  # Surface fire
                                  1,
                                  np.where((self.cfb > 0.1) & (self.cfb < 0.9),
                                           # Intermittent crown fire
                                           2,
                                           np.where(self.cfb >= 0.9,
                                                    # Active crown fire
                                                    3,
                                                    # No fire type
                                                    0)))

        return

    def calcCFC(self) -> None:
        """
        Function calculates crown fuel consumed (kg/m^2).
        :return: None
        """
        self.cfc = np.where((self.fuel_type == 10) | (self.fuel_type == 11),
                            self.cfb * self.cfl * self.pc / 100,
                            np.where((self.fuel_type == 12) | (self.fuel_type == 13),
                                     self.cfb * self.cfl * self.pdf / 100,
                                     self.cfb * self.cfl))

        return

    def calcC6hfros(self) -> None:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type
        :returns: None
        """
        self.cfros = np.where(self.fuel_type == 6,
                              np.where(self.cfc == 0,
                                       0,
                                       60 * np.power(1 - np.exp(-0.0497 * self.isi), 1) * (self.fme / 0.778237)),
                              self.cfros)

        self.hfros = np.where(self.fuel_type == 6,
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

        return [fbp_params.get(var, 'Invalid output variable') for var in out_request]

    def runFBP(self) -> list[any]:
        """
        Function to automatically run CFFBPS modelling
        """
        # Model fire behavior with CFFBPS
        self.invertWindAspect()
        self.calcSF()
        self.calcISZ()
        self.calcFMC()
        for self.ftype in [ftype for ftype in np.unique(self.fuel_type)
                           if ftype in list(self.rosParams.keys())]:
            self.calcROS()
            self.calcSFC()
            self.getCBH_CFL()
        self.calcCSFI()
        self.calcRSO()
        self.calcCFB()
        self.calcFireType()
        self.calcCFC()
        self.calcC6hfros()
        self.calcTFC()
        self.calcHFI()

        return self.getOutputs(self.out_request)


def testFBP():
    for ft in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4', 'O1a', 'O1b', 'S1', 'S2',
               'S3']:
        print(ft, FBP(fuel_type=ft, wx_date=20170616, lat=52.1152209277778, long=121.911361891667,
                      elevation=779.613, slope=0, aspect=156, ws=18, wd=189.7, ffmc=93.5, bui=70.00987167,
                      out_request=['WSV', 'RAZ', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc']).runFBP())


if __name__ == '__main__':
    testFBP()
