# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:00:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from datetime import datetime as dt


##################################################################################################
# #### CLASS FOR CANADIAN FOREST FIRE BEHAVIOR PREDICTION SYSTEM (CFFBPS) ######
##################################################################################################
class CFFBPS:
    def __init__(self):
        # Initialize CFFBPS input parameters
        self.fuel_type = None
        self.pc = None
        self.ph = None
        self.pdf = None
        self.gfl = 0.35
        self.gcf = None

        # Initialize default CFFBPS output parameters
        self.fire_type = None
        self.hfros = None
        self.hfi = None

        # Initialize weather parameters
        self.ffmc = None
        self.isi = None
        self.bui = None
        self.m = None
        self.fF = None
        self.fW = None
        self.ws = None
        self.wd = None

        # Initialize slope effect parameters
        self.slope = None
        self.aspect = None
        self.a = None
        self.b = None
        self.c = None
        self.rsz = None
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
        self.beMax = None

        # Initialize surface parameters
        self.cf = 0
        self.ffc = 0
        self.wfc = 0
        self.sfc = 0
        self.rss = None

        # Initialize foliar moisture content parameters
        self.lat = None
        self.long = None
        self.elevation = None
        self.wxDate = None
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

        ### Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = ['cbh', 'fmc']
        self.rsoVarList = ['csfi', 'sfc']
        self.cfbVarList = ['cfros', 'rso']
        self.cfcVarList = ['cfb', 'cfl']
        self.cfiVarList = ['cfros', 'cfc']

        # CFFBPS Fuel Type Code Lookup Table
        self.fbpFTCode_LUT = {
            1: 'C1',    # C-1
            2: 'C2',    # C-2
            3: 'C3',    # C-3
            4: 'C4',    # C-4
            5: 'C5',    # C-5
            6: 'C6',    # C-6
            7: 'C7',    # C-7
            8: 'D1',    # D-1
            9: 'D2',    # D-2
            10: 'M1',   # M-1
            11: 'M2',   # M-2
            12: 'M3',   # M-3
            13: 'M4',   # M-4
            14: 'O1a',  # O-1a
            15: 'O1b',  # O-1b
            16: 'S1',   # S-1
            17: 'S2',   # S-2
            18: 'S3',   # S-3
            19: 'N',     # N (non fuel)
            'C1': 1,    # C-1
            'C2': 2,    # C-2
            'C3': 3,    # C-3
            'C4': 4,    # C-4
            'C5': 5,    # C-5
            'C6': 6,    # C-6
            'C7': 7,    # C-7
            'D1': 8,    # D-1
            'D2': 9,    # D-2
            'M1': 10,   # M-1
            'M2': 11,   # M-2
            'M3': 12,   # M-3
            'M4': 13,   # M-4
            'O1a': 14,  # O-1a
            'O1b': 15,  # O-1b
            'S1': 16,   # S-1
            'S2': 17,   # S-2
            'S3': 18,   # S-3
            'N': 19		# N
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
            18: (0, 0, 0),
            'C1': (2, 0.75, 10),
            'C2': (3, 0.8, 7),
            'C3': (8, 1.15, 18),
            'C4': (4, 1.2, 10),
            'C5': (18, 1.2, 25),
            'C6': (7, 1.8, 14),
            'C7': (10, 0.5, 20),
            'D1': (0, 0, 0),
            'D2': (0, 0, 0),
            'M1': (6, 0.8, 13),
            'M2': (6, 0.8, 13),
            'M3': (6, 0.8, 8),
            'M4': (6, 0.8, 8),
            'O1a': (0, 0, 0),
            'O1b': (0, 0, 0),
            'S1': (0, 0, 0),
            'S2': (0, 0, 0),
            'S3': (0, 0, 0)
        }
        # CFFBPS Surface Fire Rate of Spread Parameters (a, b, c, q, BUI0, BEmax)
        self.rosParams = {
            1: (90, 0.0649, 4.5, 0.9, 72, 1.076),       # C-1
            2: (110, 0.0282, 1.5, 0.7, 64, 1.321),      # C-2
            3: (110, 0.0444, 3, 0.75, 62, 1.261),       # C-3
            4: (110, 0.0293, 1.5, 0.8, 66, 1.184),      # C-4
            5: (30, 0.0697, 4, 0.8, 56, 1.220),         # C-5
            6: (30, 0.08, 3, 0.8, 62, 1.197),           # C-6
            7: (45, 0.0305, 2, 0.85, 106, 1.134),       # C-7
            8: (30, 0.0232, 1.6, 0.9, 32, 1.179),       # D-1
            9: (30, 0.0232, 1.6, 0.9, 32, 1.179),       # D-2
            10: (0.8, 50, 1.250),                       # M-1
            11: (0.8, 50, 1.250),                       # M-2
            12: (120, 0.0572, 1.4, 0.8, 50, 1.250),     # M-3
            13: (100, 0.0404, 1.48, 0.8, 50, 1.250),     # M-4
            14: (190, 0.0310, 1.4, 1, None, 1),         # O-1a
            15: (250, 0.0350, 1.7, 1, None, 1),         # O-1b
            16: (75, 0.0297, 1.3, 0.75, 38, 1.460),     # S-1
            17: (40, 0.0438, 1.7, 0.75, 63, 1.256),     # S-2
            18: (55, 0.0829, 3.2, 0.75, 31, 1.590),     # S-3
            'C1': (90, 0.0649, 4.5, 0.9, 72, 1.076),    # C-1
            'C2': (110, 0.0282, 1.5, 0.7, 64, 1.321),   # C-2
            'C3': (110, 0.0444, 3, 0.75, 62, 1.261),    # C-3
            'C4': (110, 0.0293, 1.5, 0.8, 66, 1.184),   # C-4
            'C5': (30, 0.0697, 4, 0.8, 56, 1.220),      # C-5
            'C6': (30, 0.08, 3, 0.8, 62, 1.197),        # C-6
            'C7': (45, 0.0305, 2, 0.85, 106, 1.134),    # C-7
            'D1': (30, 0.0232, 1.6, 0.9, 32, 1.179),    # D-1
            'D2': (30, 0.0232, 1.6, 0.9, 32, 1.179),    # D-2
            'M1': (0.8, 50, 1.250),                     # M-1
            'M2': (0.8, 50, 1.250),                     # M-2
            'M3': (120, 0.0572, 1.4, 0.8, 50, 1.250),   # M-3
            'M4': (100, 0.0404, 1.48, 0.8, 50, 1.250),   # M-4
            'O1a': (190, 0.0310, 1.4, 1, None, 1),      # O-1a
            'O1b': (250, 0.0350, 1.7, 1, None, 1),      # O-1b
            'S1': (75, 0.0297, 1.3, 0.75, 38, 1.460),   # S-1
            'S2': (40, 0.0438, 1.7, 0.75, 63, 1.256),   # S-2
            'S3': (55, 0.0829, 3.2, 0.75, 31, 1.590),   # S-3
        }

    def calcROS(self,
                fuel_type: str | int = None,
                slope: float = None,
                aspect: float = None,
                ws: float = None,
                wd: float = None,
                ffmc: float = None,
                bui: float = None,
                pc: float = None,
                pdf: float = None,
                gcf: float = None,
                subFT_with: list[str | int] | tuple[str | int] = None) -> float:
        """
        Function to model the fire rate of spread (m/min).\n
        :param fuel_type: CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param slope: Ground slope angle/tilt of area being modelled (%)
        :param aspect: Ground slope aspect/azimuth of area being modelled (degrees)
        :param ws: Wind speed (km/h @ 10m height)
        :param wd: Wind direction (degrees, direction wind is coming from)
        :param ffmc: Canadian Forest Fire Weather Index System (CFFWIS) Fine Fuel Moisture Code
        :param bui: Canadian Forest Fire Weather Index System (CFFWIS) Buildup Index
        :param pc: percent conifer in stand (%)
        :param pdf: percent dead fir in stand (%)
        :param gcf: percent of grass cured (%)
        :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
            inFT = fuel type to be changed
            subFT = fuel type replacing inFT
        :returns:
            Modelled fire rate of spread (m/min)
        """

        if fuel_type:
            if subFT_with:
                if fuel_type == subFT_with[0]:
                    fuel_type = subFT_with[1]
            self.fuel_type = fuel_type.replace('-', '')
        if slope:
            self.slope = slope
        if aspect:
            self.aspect = aspect
        if ws:
            self.ws = ws
        if wd:
            self.wd = wd
        if ffmc:
            self.ffmc = ffmc
        if bui:
            self.bui = bui
        if pc:
            self.pc = pc
        if pdf:
            self.pdf = pdf
        if gcf:
            self.gcf = gcf

        def calcISI_slopeWind() -> tuple[float, float]:
            # Calculate the slope equivalent wind speed (for lower wind speeds)
            self.wse1 = (1 / 0.05039) * np.log(self.isf / (0.208 * self.fF))

            # Calculate the slope equivalent wind speed (for higher wind speeds)
            if self.isf < (0.999 * 2.496 * self.fF):
                self.wse2 = 28 - (1 / 0.0818) * np.log(1 - (self.isf / (2.496 * self.fF)))
            else:
                self.wse2 = 112.45

            # Assign slope equivalent wind speed
            if self.wse1 <= 40:
                self.wse = self.wse1
            else:
                self.wse = self.wse2

            # Calculate vector magnitude in x-direction
            self.wsx = (self.ws * np.sin(np.radians(self.wd))) + (self.wse * np.sin(np.radians(self.aspect)))

            # Calculate vector magnitude in y-direction
            self.wsy = (self.ws * np.cos(np.radians(self.wd))) + (self.wse * np.cos(np.radians(self.aspect)))

            # Calculate the net effective wind speed
            self.wsv = np.sqrt(pow(self.wsx, 2) + pow(self.wsy, 2))

            # Calculate the net effective wind direction (RAZ)
            if self.wsx < 0:
                self.raz = 360 - np.degrees(np.arccos(self.wsy / self.wsv))
            else:
                self.raz = np.degrees(np.arccos(self.wsy / self.wsv))

            # Calculate the wind function of the ISI equation
            if self.wsv > 40:
                self.fW = 12 * (1 - np.exp(-0.0818 * (self.wsv - 28)))
            else:
                self.fW = np.exp(0.05039 * self.wsv)

            # Calculate the new ISI with slope and wind effects
            self.isi = 0.208 * self.fW * self.fF

            return

        # Invert wind direction by 180 degrees
        if self.wd > 180:
            self.wd -= 180
        else:
            self.wd += 180

        # Invert aspect by 180 degrees
        if self.aspect > 180:
            self.aspect -= 180
        else:
            self.aspect += 180

        # Calculate fine fuel moisture content in percent (default CFFBPS equation)
        self.m = (147.2 * (101 - self.ffmc)) / (59.5 + self.ffmc)

        # Calculate the FFMC function from ISI equation
        self.fF = (91.9 * np.exp(-0.1386 * self.m)) * (1 + (pow(self.m, 5.31) / (4.93 * pow(10, 7))))

        # Calculate slope factor
        if self.slope < 70:
            self.sf = np.exp(3.533 * pow((self.slope / 100), 1.2))
        else:
            self.sf = 10

        # Calculate no slope/no wind Initial Spread Index
        self.ISZ = 0.208 * self.fF

        if self.fuel_type in list(self.rosParams.keys()):
            ### CFFBPS ROS models
            if self.fuel_type not in [10, 11, 'M1', 'M2']:
                # Get fuel type specific fixed rate of spread parameters
                self.a, self.b, self.c, self.q, self.bui0, self.beMax = self.rosParams[self.fuel_type]
                if self.fuel_type in [14, 15, 'O1a', 'O1b']:
                    ## Process O-1a/b fuel types...
                    if self.gcf < 58.8:
                        self.cf = 0.005 * (np.exp(0.061 * self.gcf) - 1)
                    else:
                        self.cf = 0.176 + (0.02 * (self.gcf - 58.8))

                    # Calculate no slope/no wind rate of spread
                    self.rsz = self.a * pow(1 - np.exp(-self.b * self.ISZ), self.c) * self.cf

                    # Calculate rate of spread with slope effect
                    self.rsf = self.rsz * self.sf

                    # Calculate initial spread index with slope effect
                    if (1 - pow(self.rsf/(self.cf * self.a), 1/self.c)) >= 0.01:
                        self.isf = np.log(1 - pow(self.rsf/(self.cf * self.a), 1/self.c)) / -self.b
                    else:
                        self.isf = np.log(0.01) / -self.b

                    # Calculate slope effects on wind and ISI
                    calcISI_slopeWind()

                    # Calculate rate of spread with slope and wind effects
                    self.rsi = self.a * pow(1 - np.exp(-self.b * self.isi), self.c) * self.cf

                    # Calculate Buildup Effect (BE)
                    self.be = 1

                elif self.fuel_type in [12, 13, 'M3', 'M4']:
                    ## Process M-3/4 fuel types...
                    # Get D1 RSZ and ISF
                    a_d1, b_d1, c_d1, q_d1, bui0_d1, beMax_d1 = self.rosParams[8]
                    rsz_d1 = a_d1 * pow(1 - np.exp(-b_d1 * self.ISZ), c_d1)
                    rsf_d1 = rsz_d1 * self.sf
                    if (1 - pow(rsf_d1/a_d1, 1/c_d1)) >= 0.01:
                        isf_d1 = np.log(1 - pow(rsf_d1/a_d1, 1/c_d1)) / -b_d1
                    else:
                        isf_d1 = np.log(0.01) / -b_d1

                    # Calculate no slope/no wind rate of spread
                    if self.fuel_type in [13, 'M4']:
                        self.rsz = ((self.pdf/100) * self.a * pow(1 - np.exp(-self.b * self.ISZ), self.c) +
                                    0.2 * (1 - self.pdf/100) * rsz_d1)
                    else:
                        self.rsz = ((self.pdf/100) * self.a * pow(1 - np.exp(-self.b * self.ISZ), self.c) +
                                    (1 - self.pdf/100) * rsz_d1)

                    # Calculate rate of spread with slope effect
                    self.rsf = self.rsz * self.sf

                    # Calculate initial spread index with slope effect
                    if (1 - pow(self.rsf/self.a, 1/self.c)) >= 0.01:
                        self.isf = ((self.pdf/100) *
                                    (np.log(1 - pow(self.rsf/self.a, 1/self.c)) / -self.b) +
                                    (1 - self.pdf/100) *
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
                    wsx_d1 = (self.ws * np.sin(np.radians(self.wd))) + (
                                wse_d1 * np.sin(np.radians(self.aspect)))

                    # Calculate vector magnitude in y-direction
                    wsy_d1 = (self.ws * np.cos(np.radians(self.wd))) + (
                                wse_d1 * np.cos(np.radians(self.aspect)))

                    # Calculate the net effective wind speed
                    wsv_d1 = np.sqrt(pow(wsx_d1, 2) + pow(wsy_d1, 2))

                    # Calculate the net effective wind direction (RAZ)
                    if wsx_d1 < 0:
                        raz_d1 = 360 - np.degrees(np.arccos(wsy_d1 / wsv_d1))
                    else:
                        raz_d1 = np.degrees(np.arccos(wsy_d1 / wsv_d1))

                    # Calculate the wind function of the ISI equation
                    if wsv_d1 > 40:
                        fW_d1 = 12 * (1 - np.exp(-0.0818 * (wsv_d1 - 28)))
                    else:
                        fW_d1 = np.exp(0.05039 * wsv_d1)

                    # Calculate the new ISI with slope and wind effects
                    isi_d1 = 0.208 * fW_d1 * self.fF


                    # Calculate rate of spread with slope and wind effects for D1
                    # Get D1 RSZ and ISF
                    rsi_d1 = a_d1 * pow(1 - np.exp(-b_d1 * isi_d1), c_d1)

                    # Calculate slope effects on wind and ISI
                    calcISI_slopeWind()

                    # Calculate rate of spread with slope and wind effects
                    if self.fuel_type in [13, 'M4']:
                        self.rsi = ((self.pdf / 100) * self.a * pow(1 - np.exp(-self.b * self.isi), self.c) +
                                    0.2 * (1 - self.pdf / 100) * rsi_d1)
                    else:
                        self.rsi = ((self.pdf/100) * self.a * pow(1 - np.exp(-self.b * self.isi), self.c) +
                                    (1 - self.pdf/100) * rsi_d1)

                    # Calculate Buildup Effect (BE)
                    if self.bui == 0:
                        self.be = 0
                    else:
                        self.be = np.exp(50 * np.log(self.q) * ((1/self.bui) - (1/self.bui0)))

                else:
                    ## Process all other fuel types...
                    # Calculate no slope/no wind rate of spread
                    self.rsz = self.a * pow(1 - np.exp(-self.b * self.ISZ), self.c)

                    # Calculate rate of spread with slope effect
                    self.rsf = self.rsz * self.sf

                    # Calculate initial spread index with slope effect
                    if (1 - pow(self.rsf/self.a, 1/self.c)) >= 0.01:
                        self.isf = np.log(1 - pow(self.rsf/self.a, 1/self.c)) / -self.b
                    else:
                        self.isf = np.log(0.01) / -self.b

                    # Calculate slope effects on wind and ISI
                    calcISI_slopeWind()

                    # Calculate rate of spread with slope and wind effects
                    self.rsi = self.a * pow(1 - np.exp(-self.b * self.isi), self.c)

                    # Calculate Buildup Effect (BE)
                    if self.bui == 0:
                        self.be = 0
                    else:
                        self.be = np.exp(50 * np.log(self.q) * ((1/self.bui) - (1/self.bui0)))

            else:
                ## Process M-1/2 fuel types...
                self.q, self.bui0, self.beMax = self.rosParams[self.fuel_type]

                # Calculate no slope/no wind rate of spread
                # Get C2 RSZ and ISF
                a_c2, b_c2, c_c2, q_c2, bui0_c2, beMax_c2 = self.rosParams[2]
                rsz_c2 = a_c2 * pow(1 - np.exp(-b_c2 * self.ISZ), c_c2)
                rsf_c2 = rsz_c2 * self.sf
                if (1 - pow(rsf_c2/a_c2, 1/c_c2)) >= 0.01:
                    isf_c2 = np.log(1 - pow(rsf_c2/a_c2, 1/c_c2)) / -b_c2
                else:
                    isf_c2 = np.log(0.01) / -b_c2
                # Get D1 RSZ and ISF
                a_d1, b_d1, c_d1, q_d1, bui0_d1, beMax_d1 = self.rosParams[8]
                rsz_d1 = a_d1 * pow(1 - np.exp(-b_d1 * self.ISZ), c_d1)
                rsf_d1 = rsz_d1 * self.sf
                if (1 - pow(rsf_d1/a_d1, 1/c_d1)) >= 0.01:
                    isf_d1 = np.log(1 - pow(rsf_d1/a_d1, 1/c_d1)) / -b_d1
                else:
                    isf_d1 = np.log(0.01) / -b_d1

                # Calculate initial spread index with slope effects
                self.isf = (self.pc/100) * isf_c2 + (1 - self.pc/100) * isf_d1

                # Calculate slope effects on wind and ISI
                calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects for C2 and D1
                # Get C2 RSI and ISI
                rsi_c2 = a_c2 * pow(1 - np.exp(-b_c2 * self.isi), c_c2)
                # Get D1 RSZ and ISF
                rsi_d1 = a_d1 * pow(1 - np.exp(-b_d1 * self.isi), c_d1)

                # Calculate rate of spread with slope and wind effects (RSI)
                if self.fuel_type in [11, 'M2']:
                    self.rsi = (self.pc / 100) * rsi_c2 + 0.2 * (1 - self.pc / 100) * rsi_d1
                else:
                    self.rsi = (self.pc / 100) * rsi_c2 + (1 - self.pc / 100) * rsi_d1

                # Calculate Buildup Effect (BE)
                if self.bui == 0:
                    self.be = 0
                else:
                    self.be = np.exp(50 * np.log(self.q) * ((1/self.bui) - (1/self.bui0)))

            # Ensure BE does not exceed beMax
            if self.be > self.beMax:
                self.be = self.beMax

            # Calculate Final ROS
            if self.fuel_type in [6, 'C6']:
                self.sfros = self.rsi * self.be
            else:
                self.hfros = self.rsi * self.be
                if self.fuel_type in [9, 'D2']:
                    self.hfros *= 0.2
        else:
            # Incorrect FBP model selected
            self.isi = np.nan
            self.hfros = np.nan

        return self.hfros

    def calcSFC(self,
                fuel_type: str | int = None,
                ft_modifier: float = None,
                ffmc: float = None,
                bui: float = None,
                gfl: float = None,
                pc: float = None,
                subFT_with: list[str | int] | tuple[str | int] = None) -> tuple[float, float, float]:
        """
        Function returns three variables, in the following order: FFC, WFC, SFC.\n
        :param fuel_type: CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param ft_modifier: CFFBPS fuel type modifier
        :param ffmc: CFFWIS Fine Fuel Moisture Code
        :param bui: CFFWIS Buildup Index
        :param gfl: Grass Fuel Load (kg/m2)
        :param pc: Percent Conifer (%, value from 0-100)
        :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
            inFT = fuel type to be changed
            subFT = fuel type replacing inFT
        :returns:
            Forest floor consumption (kg/m^2), Woody fuel consumption (kg/m^2), Surface fuel consumption (kg/m^2)
        """

        if fuel_type:
            if subFT_with:
                if fuel_type == subFT_with[0]:
                    fuel_type = subFT_with[1]
            self.fuel_type = fuel_type.replace('-', '')
        if ft_modifier:
            self.ft_modifier = ft_modifier
        if ffmc:
            self.ffmc = ffmc
        if bui:
            self.bui = bui
        if gfl:
            self.gfl = gfl
        if pc:
            self.pc = pc

        if self.fuel_type in [1, 'C1']:
            if self.ffmc > 84:
                self.ffc = np.nan
                self.wfc = np.nan
                self.sfc = 0.75 + 0.75 * np.sqrt(1 - np.exp(-0.23 * (self.ffmc - 84)))
            else:
                self.ffc = np.nan
                self.wfc = np.nan
                self.sfc = 0.75 - 0.75 * np.sqrt(1 - np.exp(0.23 * (self.ffmc - 84)))
        elif self.fuel_type in [2, 'C2']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = 5 * (1 - np.exp(-0.0115 * self.bui))
        elif self.fuel_type in [3, 4, 'C3', 'C4']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = 5 * pow(1 - np.exp(-0.0164 * self.bui), 2.24)
        elif self.fuel_type in [5, 6, 'C5', 'C6']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = 5 * pow(1 - np.exp(-0.0149 * self.bui), 2.48)
        elif self.fuel_type in [7, 'C7']:
            self.ffc = 2 * (1 - np.exp(-0.104 * (self.ffmc - 70)))
            if self.ffc < 0:
                self.ffc = 0
            self.wfc = 1.5 * (1 - np.exp(-0.0201 * self.bui))
            self.sfc = self.ffc + self.wfc
        elif self.fuel_type in [8, 9, 'D1', 'D2']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = 1.5 * (1 - np.exp(-0.0183 * self.bui))
        elif self.fuel_type in [10, 11, 'M1', 'M2']:
            c2_SFC = 5 * (1 - np.exp(-0.0115 * self.bui))
            d1_SFC = 1.5 * (1 - np.exp(-0.0183 * self.bui))
            if ft_modifier:
                self.pc = ft_modifier
            self.ph = 100 - self.pc
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = ((self.pc / 100) * c2_SFC) + ((self.ph / 100) * d1_SFC)
        elif self.fuel_type in [12, 13, 'M3', 'M4']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = 5 * (1 - np.exp(-0.0115 * self.bui))
        elif self.fuel_type in [14, 15, 'O1a', 'O1b']:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = self.gfl
        elif self.fuel_type in [16, 'S1']:
            self.ffc = 4 * (1 - np.exp(-0.025 * self.bui))
            self.wfc = 4 * (1 - np.exp(-0.034 * self.bui))
            self.sfc = self.ffc + self.wfc
        elif self.fuel_type in [17, 'S2']:
            self.ffc = 10 * (1 - np.exp(-0.013 * self.bui))
            self.wfc = 6 * (1 - np.exp(-0.06 * self.bui))
            self.sfc = self.ffc + self.wfc
        elif self.fuel_type in [18, 'S3']:
            self.ffc = 12 * (1 - np.exp(-0.0166 * self.bui))
            self.wfc = 20 * (1 - np.exp(-0.021 * self.bui))
            self.sfc = self.ffc + self.wfc
        elif self.fuel_type == [19, 'N']:
            self.ffc = 0
            self.wfc = 0
            self.sfc = 0
        else:
            self.ffc = np.nan
            self.wfc = np.nan
            self.sfc = np.nan

        return self.ffc, self.wfc, self.sfc

    def calcFMC(self,
                wxDate: int = None,
                lat: float = None,
                long: float = None,
                elevation: float = None) -> tuple[float, float]:
        """
        Function returns a tuple of foliar moisture content (FMC) and foliar moisture effect (FME).\n
        # Not recommended for conifers other than boreal pine or spruce.
        :param wxDate: date of weather observation (YYYYMMDD)
        :param lat: site latitude (decimal degrees)
        :param long: site longitude (decimal degrees)
        :param elevation: site elevation (meters)
        :returns:
            Foliar moisture content (%, value >= 0), foliar moisture effect (floating point number)
        """
        if wxDate:
            self.wxDate = wxDate
        if lat:
            self.lat = lat
        if long:
            self.long = long
        if elevation:
            self.elevation = elevation

        if self.elevation > 0:
            self.latn = 43 + (33.7 * np.exp(-0.0351 * (150 - abs(self.long))))
            self.d0 = 142.1 * (self.lat / self.latn) + (0.0172 * self.elevation)
        else:
            self.latn = 46 + (23.4 * (np.exp(-0.036 * (150 - abs(self.long)))))
            self.d0 = 151 * (self.lat / self.latn)

        try:
            self.dj = dt.strptime(str(self.wxDate), '%Y%m%d%H').timetuple().tm_yday
        except:
            try:
                self.dj = dt.strptime(str(self.wxDate), '%Y%m%d').timetuple().tm_yday
            except:
                raise Exception('Invalid Weather Date format')

        # Number of days between current date and day 0
        self.nd = abs(self.dj - self.d0)

        if self.nd < 30:
            self.fmc = 85 + (0.0189 * self.nd**2)
        elif self.nd < 50:
            self.fmc = 32.9 + (3.17 * self.nd) - (0.0288 * self.nd**2)
        else:
            self.fmc = 120

        self.fme = 1000 * pow(1.5 - (0.00275 * self.fmc), 4)/(460 + (25.9 * self.fmc))

        return self.fmc, self.fme

    def calcCSFI_RSO(self,
                     fuel_type: str | int = None,
                     cbh: float = None,
                     fmc: float = None,
                     sfc: float = None,
                     subFT_with: list[str | int] | tuple[str | int] = None) -> tuple[float, float]:
        """
        Function calculates the critical surface fire intensity (CSFI) and critical surface fire
        rate of spread (RSO).\n
        :param fuel_type: CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param cbh: Foliar moisture content (%, value from 0-100+)
        :param fmc: Foliar moisture content (%, value from 0-100+)
        :param sfc: Surface fuel consumption estimate (kg/m^2)
        :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
            inFT = fuel type to be changed
            subFT = fuel type replacing inFT
        :returns:
            Tuple of critical surface fire intensity (kW/m), and critical surface fire rate of spread (m/min)
        """

        if fuel_type:
            if subFT_with:
                if fuel_type == subFT_with[0]:
                    fuel_type = subFT_with[1]
            self.fuel_type = fuel_type.replace('-', '')
        if fmc:
            self.fmc = fmc
        if sfc:
            self.sfc = sfc
        if cbh:
            # Use input CBH
            self.cbh = cbh
        else:
            # Get CBH for fuel type
            self.cbh = self.fbpCBH_CFL_HT_LUT.get(self.fuel_type)[0]

        # Calculate critical surface fire intensity
        self.csfi = pow(0.01 * self.cbh * (460 + (25.9 * self.fmc)), 1.5)

        # Calculate critical surface fire rate of spread
        if (self.sfc is not None) and (self.sfc > 0):
            self.rso = self.csfi / (300 * self.sfc)
        else:
            self.rso = 0

        return self.csfi, self.rso

    def calcCFB_FireType(self,
                         fuel_type: str | int = None,
                         ros: float = None,
                         rso: float = None,
                         subFT_with: list[str | int] | tuple[str | int] = None) -> tuple[float, str]:
        """
        Function calculates crown fraction burned using equation in Forestry Canada Fire Danger Group (1992)
        :param fuel_type:  CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param ros: fire rate of spread (m/min); must be the surface fire rate of spread for C6 fuel type
        :param rso: Critical rate of spread (m/min)
        :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
            inFT = fuel type to be changed
            subFT = fuel type replacing inFT
        :returns:
            Crown fraction burned (proportion; value from 0-1)
        """

        if fuel_type:
            if subFT_with:
                if fuel_type == subFT_with[0]:
                    fuel_type = subFT_with[1]
            self.fuel_type = fuel_type.replace('-', '')

        if ros:
            if self.fuel_type in [6, 'C6']:
                self.sfros = ros
            else:
                self.hfros = ros
        if rso:
            self.rso = rso

        if self.fuel_type in [6, 'C6']:
            # For C-6 fuel type
            if (self.sfros - self.rso) < -3086:
                self.cfb = 0
            else:
                self.cfb = 1 - np.exp(-0.23 * (self.sfros - self.rso))
        else:
            # For all other fuel types
            if (self.hfros - self.rso) < -3086:
                self.cfb = 0
            else:
                self.cfb = 1 - np.exp(-0.23 * (self.hfros - self.rso))
        if self.cfb < 0:
            self.cfb = 0

        if self.cfb < 0:
            self.cfb = 0

        if self.cfb <= 0.1:
            self.fire_type = 'surface'
        elif (self.cfb > 0.1) & (self.cfb < 0.9):
            self.fire_type = 'intermittent crown'
        else:
            self.fire_type = 'active crown'

        return self.cfb, self.fire_type

    def calcCFC(self,
                fuel_type: str | int = None,
                cfb: float = None,
                cfl: float = None,
                pc: float = None,
                subFT_with: list[str | int] | tuple[str | int] = None) -> float:
        """
        Function calculates crown fuel consumed (kg/m^2)
        :param fuel_type: CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param cfb: Crown fraction burned (proportion; value from 0-1)
        :param cfl: Canopy fuel load (kg/m^2)
        :param pc: Percent conifer (%; value from 0-100)
        :param subFT_with: Tuple or list of fuel type substitutions (inFT, subFT)
            inFT = fuel type to be changed
            subFT = fuel type replacing inFT
        :returns:
            Crown fuel consumed (kg/m^2)
        """

        if fuel_type:
            if subFT_with:
                if fuel_type == subFT_with[0]:
                    fuel_type = subFT_with[1]
            self.fuel_type = fuel_type.replace('-', '')
        if cfb:
            self.cfb = cfb
        if pc:
            self.pc = pc
        if cfl:
            # Use input CFL value
            self.cfl = cfl
        else:
            # Get CFL for fuel type
            self.cfl = self.fbpCBH_CFL_HT_LUT.get(self.fuel_type)[1]

        if self.fuel_type in [10, 11, 'M1', 'M2']:
            self.cfc = self.cfb * self.cfl * self.pc / 100
        elif self.fuel_type in [12, 13, 'M3', 'M4']:
            self.cfc = self.cfb * self.cfl * self.pdf / 100
        else:
            self.cfc = self.cfb * self.cfl

        return self.cfc

    def calcC6hfros(self,
                    sfros: float = None,
                    cfc: float = None,
                    isi: float = None,
                    fme: float = None) -> tuple[float, float]:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type
        :param sfros: C6 surface fire rate of spread
        :param cfc: Crown fuel consumed
        :param isi: Final ISI, accounting for wind and slope
        :param fme: Foliar moisture effect
        :returns:
            C6 crown and head fire rate of spread (m/min)
        """
        if sfros:
            self.sfros = sfros
        if cfc:
            self.cfc = cfc
        if isi:
            self.isi = isi
        if fme:
            self.fme = fme

        if self.cfc == 0:
            self.cfros = 0
        else:
            self.cfros = 60 * pow(1 - np.exp(-0.0497 * self.isi), 1) * (self.fme / 0.778237)

        self.hfros = self.sfros + (self.cfb * (self.cfros - self.sfros))

        return self.cfros, self.hfros

    def calcTFC(self,
                sfc: float = None,
                cfc: float = None) -> float:
        """
        Function to calculate total fuel consumed (kg/m^2)
        :param sfc: Estimated surface fuel consumed (kg/m^2)
        :param cfc: Estimated crown fuel consumed (kg/m^2)
        :returns:
            Total fuel consumed (kg/m^2)
        """
        if sfc:
            self.sfc = sfc
        if cfc:
            self.cfc = cfc

        self.tfc = self.sfc + self.cfc

        return self.tfc

    def calcHFI(self,
                ros: float = None,
                tfc: float = None) -> float:
        """
        Function to calculate fire type, total fuel consumption, and head fire intensity
        :param ros: Fire rate of spread
        :param tfc: Estimated total fuel consumed (kg/m^2)
        :returns:
            Tuple of fire type and head fire intensity
        """
        if ros:
            self.hfros = ros
        if tfc:
            self.tfc = tfc

        self.hfi = 300 * self.hfros * self.tfc

        return self.hfi

    def getOutputs(self, out_request: list[str]) -> list[any]:
        # Dictionary of CFFBPS parameters
        fbp_params = {
            # Default output variables
            'fire_type': self.fire_type,    # Type of fire (surface, intermittent crown, active crown)
            'hfros': self.hfros,            # Head fire rate of spread (m/min)
            'hfi': self.hfi,                # head fire intensity (kW/m)

            # Weather variables
            'ws': self.ws,      # Observed wind speed (km/h)
            'wd': self.wd,      # Wind azimuth/direction (degrees)
            'm': self.m,        # Moisture content equivalent of the FFMC (%, value from 0-100+)
            'fF': self.fF,      # Fine fuel moisture function in the ISI
            'fW': self.fW,      # Wind function in the ISI
            'isi': self.isi,    # Final ISI, accounting for wind and slope

            # Slope + wind effect variables
            'a': self.a,        # Rate of spread equation coefficient
            'b': self.b,        # Rate of spread equation coefficient
            'c': self.c,        # Rate of spread equation coefficient
            'RSZ': self.rsz,    # Surface spread rate with zero wind on level terrain
            'SF': self.sf,      # Slope factor
            'RSF': self.rsf,    # Spread rate with zero wind, upslope
            'ISF': self.isf,    # ISI, with zero wind upslope
            'RSI': self.rsi,    # Initial spread rate without BUI effect
            'WSE1': self.wse1,  # Original slope equivalent wind speed value for cases where WSE1 <= 40
            'WSE2': self.wse2,  # New slope equivalent sind speed value for cases where WSE1 > 40
            'WSE': self.wse,    # Slope equivalent wind speed
            'WSX': self.wsx,    # Net vectorized wind speed in the x-direction
            'WSY': self.wsy,    # Net vectorized wind speed in the y-direction
            'WSV': self.wsv,    # Net vectorized wind speed
            'RAZ': self.raz,    # Net vectorized wind direction

            # BUI effect variables
            'q': self.q,            # Proportion of maximum rate of spread at BUI equal to 50
            'bui0': self.bui0,      # Average BUI for each fuel type
            'BE': self.be,          # Buildup effect on spread rate
            'beMax': self.beMax,    # Maximum allowable BE value

            # Surface fuel variables
            'ffc': self.ffc,    # Estimated forest floor consumption
            'wfc': self.wfc,    # Estimated woody fuel consumption
            'sfc': self.sfc,    # Estimated total surface fuel consumption

            # Foliar moisture content variables
            'latn': self.latn,  # Normalized latitude
            'dj': self.dj,      # Julian date of day being modelled
            'd0': self.d0,      # Julian date of minimum foliar moisture content
            'nd': self.nd,      # number of days between modelled fire date and d0
            'fmc': self.fmc,    # foliar moisture content
            'fme': self.fme,    # foliar moisture effect

            # Critical crown fire threshold variables
            'csfi': self.csfi,  # critical intensity (kW/m)
            'rso': self.rso,    # critical rate of spread (m/min)

            # Crown fuel parameters
            'cbh': self.cbh,    # Height to live crown base (m)
            'cfb': self.cfb,    # Crown fraction burned (proportion, value ranging from 0-1)
            'cfl': self.cfl,    # Crown fuel load (kg/m^2)
            'cfc': self.cfc     # Crown fuel consumed
        }

        return [fbp_params.get(var, 'Invalid output variable') for var in out_request]

    def getFBP(self,
               fuel_type: str | int,
               wxDate: int,
               lat: float,
               long: float,
               elevation: float,
               slope: float,
               aspect: float,
               ws: float,
               wd: float,
               ffmc: float,
               bui: float,
               pc: float = 50,
               pdf: float = 35,
               gfl: float = 0.3,
               gcf: float = 80,
               out_request: list[str] = None,
               subFT_with: list[str | int] | tuple[str | int] = None) -> list[any]:
        """
        Function calculates fire type, head fire rate of spread, head fire intensity using CFFBPS
        :param fuel_type: CFFBPS fuel type (either code: 1-18, or alpha: C1-S3) used to select ROS model
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
        :param wxDate: Date of weather observation (to model fire for) (YYYYMMDD)
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
            fire_type = TYpe of fire predicted to occur (surface, intermittent crown, active crown)
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
            beMax = Maximum allowable BE value

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

        if subFT_with:
            if fuel_type == subFT_with[0]:
                fuel_type = subFT_with[1]
        self.fuel_type = fuel_type.replace('-', '')
        self.wxDate = wxDate
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

        if out_request is None:
            out_request = ['fire_type', 'hfros', 'hfi']

        # Model fire behavior with CFFBPS
        self.calcROS()
        self.calcSFC()
        self.calcFMC()
        self.calcCSFI_RSO()
        self.calcCFB_FireType()
        self.calcCFC()
        if self.fuel_type in [6, 'C6']:
            self.calcC6hfros()
        self.calcTFC()
        self.calcHFI()

        return self.getOutputs(out_request)


def testCFFBPS():
    fbp = CFFBPS()
    for ft in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4', 'O1a', 'O1b', 'S1', 'S2',
               'S3']:
        print(ft, fbp.getFBP(fuel_type=ft, wxDate=20170616, lat=52.1152209277778, long=121.911361891667,
                             elevation=779.613, slope=0, aspect=156, ws=18, wd=189.7, ffmc=93.5, bui=70.00987167,
                             out_request=['WSV', 'RAZ', 'fire_type', 'hfros', 'hfi', 'ffc', 'wfc', 'sfc']))


if __name__ == "__main__":
    testCFFBPS()
