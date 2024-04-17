# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:45:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from datetime import datetime as dt


# ### Lists for CFFBPS Crown Fire Metric variables
csfiVarList = ['cbh', 'fmc']
rsoVarList = ['csfi', 'sfc']
cfbVarList = ['cfros', 'rso']
cfcVarList = ['cfb', 'cfl']
cfiVarList = ['cfros', 'cfc']

# CFFBPS Fuel Type Code Lookup Table
fbpFTCode_LUT = {
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
fbpCBH_CFL_HT_LUT = {
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
rosParams = {
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


def calcROS(fuel_type=None, slope=None, aspect=None, ws=None, wd=None,
            ffmc=None, bui=None, pc=None, pdf=None, gcf=None, subFT_with=None):
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
        fuel_type = fuel_type.replace('-', '')

    def calcISI_slopeWind():
        # Calculate the slope equivalent wind speed (for lower wind speeds)
        wse1 = (1 / 0.05039) * np.log(isf / (0.208 * fF))

        # Calculate the slope equivalent wind speed (for higher wind speeds)
        if isf < (0.999 * 2.496 * fF):
            wse2 = 28 - (1 / 0.0818) * np.log(1 - (isf / (2.496 * fF)))
        else:
            wse2 = 112.45

        # Assign slope equivalent wind speed
        if wse1 <= 40:
            wse = wse1
        else:
            wse = wse2

        # Calculate vector magnitude in x-direction
        wsx = (ws * np.sin(np.radians(wd))) + (wse * np.sin(np.radians(aspect)))

        # Calculate vector magnitude in y-direction
        wsy = (ws * np.cos(np.radians(wd))) + (wse * np.cos(np.radians(aspect)))

        # Calculate the net effective wind speed
        wsv = np.sqrt(pow(wsx, 2) + pow(wsy, 2))

        # Calculate the net effective wind direction (RAZ)
        if wsx < 0:
            raz = 360 - np.degrees(np.arccos(wsy / wsv))
        else:
            raz = np.degrees(np.arccos(wsy / wsv))

        # Calculate the wind function of the ISI equation
        if wsv > 40:
            fW = 12 * (1 - np.exp(-0.0818 * (wsv - 28)))
        else:
            fW = np.exp(0.05039 * wsv)

        # Calculate the new ISI with slope and wind effects
        isi = 0.208 * fW * fF

        # Return the net effective wind speed and ISI with slope and wind effects
        return raz, isi

    # Invert wind direction by 180 degrees
    if wd > 180:
        wd -= 180
    else:
        wd += 180

    # Invert aspect by 180 degrees
    if aspect > 180:
        aspect -= 180
    else:
        aspect += 180

    # Calculate fine fuel moisture content in percent (default CFFBPS equation)
    m = (147.2 * (101 - ffmc)) / (59.5 + ffmc)

    # Calculate the FFMC function from ISI equation
    fF = (91.9 * np.exp(-0.1386 * m)) * (1 + (pow(m, 5.31) / (4.93 * pow(10, 7))))

    # Calculate slope factor
    if slope < 70:
        sf = np.exp(3.533 * pow((slope / 100), 1.2))
    else:
        sf = 10

    # Calculate no slope/no wind Initial Spread Index
    ISZ = 0.208 * fF

    if fuel_type in list(rosParams.keys()):
        ### CFFBPS ROS models
        if fuel_type not in [10, 11, 'M1', 'M2']:
            # Get fuel type specific fixed rate of spread parameters
            a, b, c, q, bui0, beMax = rosParams[fuel_type]
            if fuel_type in [14, 15, 'O1a', 'O1b']:
                ## Process O-1a/b fuel types...
                if gcf < 58.8:
                    cf = 0.005 * (np.exp(0.061 * gcf) - 1)
                else:
                    cf = 0.176 + (0.02 * (gcf - 58.8))

                # Calculate no slope/no wind rate of spread
                rsz = a * pow(1 - np.exp(-b * ISZ), c) * cf

                # Calculate rate of spread with slope effect
                rsf = rsz * sf

                # Calculate initial spread index with slope effect
                if (1 - pow(rsf/(cf * a), 1/c)) >= 0.01:
                    isf = np.log(1 - pow(rsf/(cf * a), 1/c)) / -b
                else:
                    isf = np.log(0.01) / -b

                # Calculate slope effects on wind and ISI
                raz, isi = calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                rsi = a * pow(1 - np.exp(-b * isi), c) * cf

                # Calculate Buildup Effect (BE)
                be = 1

            elif fuel_type in [12, 13, 'M3', 'M4']:
                ## Process M-3/4 fuel types...
                # Get D1 RSZ and ISF
                a_d1, b_d1, c_d1, q_d1, bui0_d1, beMax_d1 = rosParams[8]
                rsz_d1 = a_d1 * pow(1 - np.exp(-b_d1 * ISZ), c_d1)
                rsf_d1 = rsz_d1 * sf
                if (1 - pow(rsf_d1/a_d1, 1/c_d1)) >= 0.01:
                    isf_d1 = np.log(1 - pow(rsf_d1/a_d1, 1/c_d1)) / -b_d1
                else:
                    isf_d1 = np.log(0.01) / -b_d1

                # Calculate no slope/no wind rate of spread
                if fuel_type in [13, 'M4']:
                    rsz = ((pdf/100) * a * pow(1 - np.exp(-b * ISZ), c) +
                           0.2 * (1 - pdf/100) * rsz_d1)
                else:
                    rsz = ((pdf/100) * a * pow(1 - np.exp(-b * ISZ), c) +
                           (1 - pdf/100) * rsz_d1)

                # Calculate rate of spread with slope effect
                rsf = rsz * sf

                # Calculate initial spread index with slope effect
                if (1 - pow(rsf/a, 1/c)) >= 0.01:
                    isf = ((pdf/100) *
                           (np.log(1 - pow(rsf/a, 1/c)) / -b) +
                           (1 - pdf/100) *
                           isf_d1)
                else:
                    isf = np.log(0.01) / -b


                # Calculate ISI and for D1
                # Calculate the slope equivalent wind speed (for lower wind speeds)
                wse1_d1 = (1 / 0.05039) * np.log(isf_d1 / (0.208 * fF))

                # Calculate the slope equivalent wind speed (for higher wind speeds)
                if isf_d1 < (0.999 * 2.496 * fF):
                    wse2_d1 = 28 - (1 / 0.0818) * np.log(1 - (isf_d1 / (2.496 * fF)))
                else:
                    wse2_d1 = 112.45

                # Assign slope equivalent wind speed
                if wse1_d1 <= 40:
                    wse_d1 = wse1_d1
                else:
                    wse_d1 = wse2_d1

                # Calculate vector magnitude in x-direction
                wsx_d1 = (ws * np.sin(np.radians(wd))) + (
                            wse_d1 * np.sin(np.radians(aspect)))

                # Calculate vector magnitude in y-direction
                wsy_d1 = (ws * np.cos(np.radians(wd))) + (
                            wse_d1 * np.cos(np.radians(aspect)))

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
                isi_d1 = 0.208 * fW_d1 * fF


                # Calculate rate of spread with slope and wind effects for D1
                # Get D1 RSZ and ISF
                rsi_d1 = a_d1 * pow(1 - np.exp(-b_d1 * isi_d1), c_d1)

                # Calculate slope effects on wind and ISI
                raz, isi = calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                if fuel_type in [13, 'M4']:
                    rsi = ((pdf / 100) * a * pow(1 - np.exp(-b * isi), c) +
                           0.2 * (1 - pdf / 100) * rsi_d1)
                else:
                    rsi = ((pdf/100) * a * pow(1 - np.exp(-b * isi), c) +
                                (1 - pdf/100) * rsi_d1)

                # Calculate Buildup Effect (BE)
                if bui == 0:
                    be = 0
                else:
                    be = np.exp(50 * np.log(q) * ((1/bui) - (1/bui0)))

            else:
                ## Process all other fuel types...
                # Calculate no slope/no wind rate of spread
                rsz = a * pow(1 - np.exp(-b * ISZ), c)

                # Calculate rate of spread with slope effect
                rsf = rsz * sf

                # Calculate initial spread index with slope effect
                if (1 - pow(rsf/a, 1/c)) >= 0.01:
                    isf = np.log(1 - pow(rsf/a, 1/c)) / -b
                else:
                    isf = np.log(0.01) / -b

                # Calculate slope effects on wind and ISI
                raz, isi = calcISI_slopeWind()

                # Calculate rate of spread with slope and wind effects
                rsi = a * pow(1 - np.exp(-b * isi), c)

                # Calculate Buildup Effect (BE)
                if bui == 0:
                    be = 0
                else:
                    be = np.exp(50 * np.log(q) * ((1/bui) - (1/bui0)))

        else:
            ## Process M-1/2 fuel types...
            q, bui0, beMax = rosParams[fuel_type]

            # Calculate no slope/no wind rate of spread
            # Get C2 RSZ and ISF
            a_c2, b_c2, c_c2, q_c2, bui0_c2, beMax_c2 = rosParams[2]
            rsz_c2 = a_c2 * pow(1 - np.exp(-b_c2 * ISZ), c_c2)
            rsf_c2 = rsz_c2 * sf
            if (1 - pow(rsf_c2/a_c2, 1/c_c2)) >= 0.01:
                isf_c2 = np.log(1 - pow(rsf_c2/a_c2, 1/c_c2)) / -b_c2
            else:
                isf_c2 = np.log(0.01) / -b_c2
            # Get D1 RSZ and ISF
            a_d1, b_d1, c_d1, q_d1, bui0_d1, beMax_d1 = rosParams[8]
            rsz_d1 = a_d1 * pow(1 - np.exp(-b_d1 * ISZ), c_d1)
            rsf_d1 = rsz_d1 * sf
            if (1 - pow(rsf_d1/a_d1, 1/c_d1)) >= 0.01:
                isf_d1 = np.log(1 - pow(rsf_d1/a_d1, 1/c_d1)) / -b_d1
            else:
                isf_d1 = np.log(0.01) / -b_d1

            # Calculate initial spread index with slope effects
            isf = (pc/100) * isf_c2 + (1 - pc/100) * isf_d1

            # Calculate slope effects on wind and ISI
            raz, isi = calcISI_slopeWind()

            # Calculate rate of spread with slope and wind effects for C2 and D1
            # Get C2 RSI and ISI
            rsi_c2 = a_c2 * pow(1 - np.exp(-b_c2 * isi), c_c2)
            # Get D1 RSZ and ISF
            rsi_d1 = a_d1 * pow(1 - np.exp(-b_d1 * isi), c_d1)

            # Calculate rate of spread with slope and wind effects (RSI)
            if fuel_type in [11, 'M2']:
                rsi = (pc / 100) * rsi_c2 + 0.2 * (1 - pc / 100) * rsi_d1
            else:
                rsi = (pc / 100) * rsi_c2 + (1 - pc / 100) * rsi_d1

            # Calculate Buildup Effect (BE)
            if bui == 0:
                be = 0
            else:
                be = np.exp(50 * np.log(q) * ((1/bui) - (1/bui0)))

        # Ensure BE does not exceed beMax
        if be > beMax:
            be = beMax

        # Calculate Final ROS
        if fuel_type in [6, 'C6']:
            sfros = rsi * be
            hfros = None
        else:
            sfros = None
            hfros = rsi * be
            if fuel_type in [9, 'D2']:
                hfros *= 0.2
    else:
        # Incorrect FBP model selected
        raz = None
        isi = None
        sfros = None
        hfros = None

    return raz, isi, sfros, hfros


def calcSFC(fuel_type=None, ft_modifier=None, ffmc=None, bui=None, gfl=None, pc=None, subFT_with=None):
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
        fuel_type = fuel_type.replace('-', '')
    if ft_modifier:
        ft_modifier = ft_modifier
    if ffmc:
        ffmc = ffmc
    if bui:
        bui = bui
    if gfl:
        gfl = gfl
    if pc:
        pc = pc

    if fuel_type in [1, 'C1']:
        if ffmc > 84:
            ffc = np.nan
            wfc = np.nan
            sfc = 0.75 + 0.75 * np.sqrt(1 - np.exp(-0.23 * (ffmc - 84)))
        else:
            ffc = np.nan
            wfc = np.nan
            sfc = 0.75 - 0.75 * np.sqrt(1 - np.exp(0.23 * (ffmc - 84)))
    elif fuel_type in [2, 'C2']:
        ffc = np.nan
        wfc = np.nan
        sfc = 5 * (1 - np.exp(-0.0115 * bui))
    elif fuel_type in [3, 4, 'C3', 'C4']:
        ffc = np.nan
        wfc = np.nan
        sfc = 5 * pow(1 - np.exp(-0.0164 * bui), 2.24)
    elif fuel_type in [5, 6, 'C5', 'C6']:
        ffc = np.nan
        wfc = np.nan
        sfc = 5 * pow(1 - np.exp(-0.0149 * bui), 2.48)
    elif fuel_type in [7, 'C7']:
        ffc = 2 * (1 - np.exp(-0.104 * (ffmc - 70)))
        if ffc < 0:
            ffc = 0
        wfc = 1.5 * (1 - np.exp(-0.0201 * bui))
        sfc = ffc + wfc
    elif fuel_type in [8, 9, 'D1', 'D2']:
        ffc = np.nan
        wfc = np.nan
        sfc = 1.5 * (1 - np.exp(-0.0183 * bui))
    elif fuel_type in [10, 11, 'M1', 'M2']:
        c2_SFC = 5 * (1 - np.exp(-0.0115 * bui))
        d1_SFC = 1.5 * (1 - np.exp(-0.0183 * bui))
        if ft_modifier:
            pc = ft_modifier
        ph = 100 - pc
        ffc = np.nan
        wfc = np.nan
        sfc = ((pc / 100) * c2_SFC) + ((ph / 100) * d1_SFC)
    elif fuel_type in [12, 13, 'M3', 'M4']:
        ffc = np.nan
        wfc = np.nan
        sfc = 5 * (1 - np.exp(-0.0115 * bui))
    elif fuel_type in [14, 15, 'O1a', 'O1b']:
        ffc = np.nan
        wfc = np.nan
        sfc = gfl
    elif fuel_type in [16, 'S1']:
        ffc = 4 * (1 - np.exp(-0.025 * bui))
        wfc = 4 * (1 - np.exp(-0.034 * bui))
        sfc = ffc + wfc
    elif fuel_type in [17, 'S2']:
        ffc = 10 * (1 - np.exp(-0.013 * bui))
        wfc = 6 * (1 - np.exp(-0.06 * bui))
        sfc = ffc + wfc
    elif fuel_type in [18, 'S3']:
        ffc = 12 * (1 - np.exp(-0.0166 * bui))
        wfc = 20 * (1 - np.exp(-0.021 * bui))
        sfc = ffc + wfc
    elif fuel_type == [19, 'N']:
        ffc = 0
        wfc = 0
        sfc = 0
    else:
        ffc = np.nan
        wfc = np.nan
        sfc = np.nan

    return ffc, wfc, sfc


# Estimate CFFBPS foliar moisture content
def calcFMC(wxDate=None, lat=None, long=None, elevation=None):
    """
    Function returns a tuple of foliar moisture content (FMC) and foliar moisture effect (FME).\n
    # Not recommended for conifers other than boreal pine or spruce.
    :param lat: site latitude (decimal degrees)
    :param long: site longitude (decimal degrees)
    :param elevation: site elevation (meters)
    :param wxDate: string representing date of weather observation (YYYYMMDD)
    :returns:
        Foliar moisture content (%, value >= 0), foliar moisture effect (floating point number)
    """
    if wxDate:
        wxDate = wxDate
    if lat:
        lat = lat
    if long:
        long = long
    if elevation:
        elevation = elevation

    if elevation > 0:
        latn = 43 + (33.7 * np.exp(-0.0351 * (150 - abs(long))))
        d0 = 142.1 * (lat / latn) + (0.0172 * elevation)
    else:
        latn = 46 + (23.4 * (np.exp(-0.036 * (150 - abs(long)))))
        d0 = 151 * (lat / latn)

    try:
        dj = dt.strptime(str(wxDate), '%Y%m%d%H').timetuple().tm_yday
    except:
        try:
            dj = dt.strptime(str(wxDate), '%Y%m%d').timetuple().tm_yday
        except:
            raise Exception('Invalid Weather Date format')

    # Number of days between current date and day 0
    nd = abs(dj - d0)

    if nd < 30:
        fmc = 85 + (0.0189 * nd**2)
    elif nd < 50:
        fmc = 32.9 + (3.17 * nd) - (0.0288 * nd**2)
    else:
        fmc = 120

    fme = 1000 * pow(1.5 - (0.00275 * fmc), 4)/(460 + (25.9 * fmc))

    return fmc, fme

# Calculate CFFBPS Critical Intensity
def calcCSFI_RSO(fuel_type=None, cbh=None, fmc=None, sfc=None, subFT_with=None):
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
        fuel_type = fuel_type.replace('-', '')
    if fmc:
        fmc = fmc
    if sfc:
        sfc = sfc
    if cbh:
        # Use input CBH
        cbh = cbh
    else:
        # Get CBH for fuel type
        cbh = fbpCBH_CFL_HT_LUT.get(fuel_type)[0]

    # Calculate critical surface fire intensity
    csfi = pow(0.01 * cbh * (460 + (25.9 * fmc)), 1.5)

    # Calculate critical surface fire rate of spread
    if (sfc is not None) and (sfc > 0):
        rso = csfi / (300 * sfc)
    else:
        rso = 0

    return csfi, rso

# Calculate Crown Fraction Burned
def calcCFB_FireType(fuel_type=None, ros=None, rso=None, subFT_with=None):
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
        fuel_type = fuel_type.replace('-', '')

    if ros:
        if fuel_type in [6, 'C6']:
            sfros = ros
        else:
            hfros = ros
    if rso:
        rso = rso

    if fuel_type in [6, 'C6']:
        # For C-6 fuel type
        if (sfros - rso) < -3086:
            cfb = 0
        else:
            cfb = 1 - np.exp(-0.23 * (sfros - rso))
    else:
        # For all other fuel types
        if (hfros - rso) < -3086:
            cfb = 0
        else:
            cfb = 1 - np.exp(-0.23 * (hfros - rso))
    if cfb < 0:
        cfb = 0

    if cfb < 0:
        cfb = 0

    if cfb <= 0.1:
        fire_type = 'surface'
    elif (cfb > 0.1) & (cfb < 0.9):
        fire_type = 'intermittent crown'
    else:
        fire_type = 'active crown'

    return cfb, fire_type

# Calculate Crown Fuel Consumed
def calcCFC(fuel_type=None, cfb=None, cfl=None, pc=None, pdf=None, subFT_with=None):
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
    :param pdf: Percent dead fir (%; value from 0-100)
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
        fuel_type = fuel_type.replace('-', '')
    if cfl:
        # Use input CFL value
        cfl = cfl
    else:
        # Get CFL for fuel type
        cfl = fbpCBH_CFL_HT_LUT.get(fuel_type)[1]

    if fuel_type in [10, 11, 'M1', 'M2']:
        cfc = cfb * cfl * pc / 100
    elif fuel_type in [12, 13, 'M3', 'M4']:
        cfc = cfb * cfl * pdf / 100
    else:
        cfc = cfb * cfl

    return cfc


def calcC6hfros(sfros=None, cfb=None, isi=None, fme=None):
    """
    Function to calculate crown and total head fire rate of spread for the C6 fuel type
    :param sfros: C6 surface fire rate of spread
    :param isi: Final ISI, accounting for wind and slope
    :param fme: Foliar moisture effect
    :returns:
        C6 crown and head fire rate of spread (m/min)
    """
    if cfb == 0:
        cfros = 0
    else:
        cfros = 60 * pow(1 - np.exp(-0.0497 * isi), 1) * (fme / 0.778237)

    hfros = sfros + (cfb * (cfros - sfros))

    return cfros, hfros

def calcTFC(sfc=None, cfc=None):
    """
    Function to calculate total fuel consumed (kg/m^2)
    :param sfc: Estimated surface fuel consumed (kg/m^2)
    :param cfc: Estimated crown fuel consumed (kg/m^2)
    :returns:
        Total fuel consumed (kg/m^2)
    """
    if sfc:
        sfc = sfc
    if cfc:
        cfc = cfc

    tfc = sfc + cfc

    return tfc

def calcHFI(ros=None, tfc=None):
    """
    Function to calculate fire type, total fuel consumption, and head fire intensity
    :param ros: Fire rate of spread
    :param tfc: Estimated total fuel consumed (kg/m^2)
    :returns:
        Tuple of fire type and head fire intensity
    """
    if ros:
        hfros = ros
    if tfc:
        tfc = tfc

    hfi = 300 * hfros * tfc

    return hfi
