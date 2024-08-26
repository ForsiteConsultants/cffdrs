# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:00:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from typing import Union


# ### DRYING PHASE
dc_daylength_dict = {
    'January': -1.6,
    'February': -1.6,
    'March': -1.6,
    'April': 0.9,
    'May': 3.8,
    'June': 5.8,
    'July': 6.4,
    'August': 5,
    'September': 2.4,
    'October': 0.4,
    'November': -1.6,
    'December': -1.6,
    '01': -1.6,
    '02': -1.6,
    '03': -1.6,
    '04': 0.9,
    '05': 3.8,
    '06': 5.8,
    '07': 6.4,
    '08': 5,
    '09': 2.4,
    '10': 0.4,
    '11': -1.6,
    '12': -1.6
}


"""
The diurnalFFMC function is still under development...
"""
# def diurnalFFMC(time_lookup, rh, ffmc0_1600, ffmc_1600):
#     """
#     Function to calculate diurnal Hourly FFMC per Lawson et al. (1996)
#     :param time_lookup: string or list of strings; time of day to estimate diurnal FFMC
#     :param rh: int, float or numpy ndarray; relative humidity value (%)
#     :param ffmc0_1600: float or numpy ndarray; yesterday's FFMC value at 1600-hr (unitless code)
#     :param ffmc_1600: float or numpy ndarray; today's FFMC value at 1600-hr (unitless code)
#     :return: float or numpy ndarray; diurnal FFMC value (unitless code)
#     """
#     if not isinstance(time_lookup, (str, np.ndarray)):
#         raise TypeError('time_lookup must be either string or numpy ndarray data types')
#     if not isinstance(rh, (int, float, np.ndarray)):
#         raise TypeError('rh must be either int, float or numpy ndarray data types')
#     if not isinstance(ffmc0_1600, (int, float, np.ndarray)):
#         raise TypeError('ffmc0_1600 must be either int, float or numpy ndarray data types')
#     if not isinstance(ffmc_1600, (int, float, np.ndarray)):
#         raise TypeError('ffmc_1600 must be either int, float or numpy ndarray data types')
#
#     # ### YESTERDAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
#     m0_1600 = 147.2 * (101 - ffmc0_1600) / (59.5 + ffmc0_1600)
#
#     # ### TODAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
#     m_1600 = 147.2 * (101 - ffmc_1600) / (59.5 + ffmc_1600)
#     return


def hourlyFFMC(ffmc0: Union[int, float, np.ndarray],
               temp: Union[int, float, np.ndarray],
               rh: Union[int, float, np.ndarray],
               wind: Union[int, float, np.ndarray],
               precip: Union[int, float, np.ndarray],
               use_precise_values: bool = False) -> Union[float, np.ndarray]:
    """
    Function to calculate hourly FFMC values per Van Wagner (1977) and Alexander et al. (1984).
    :param ffmc0: previous hour's FFMC value (unitless code)
    :param temp: temperature value (C)
    :param rh: relative humidity value (%)
    :param wind: wind speed value (km/h)
    :param precip: precipitation value (mm)
    :param use_precise_values: use higher precision for m0 & Daily FFMC equations for drying/wetting moisture
    :return: the hourly FFMC value
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [ffmc0, temp, rh, wind, precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify ffmc0
    if not isinstance(ffmc0, (int, float, np.ndarray)):
        raise TypeError('ffmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(ffmc0, np.ndarray):
        ffmc0 = np.ma.array(ffmc0, mask=np.isnan(ffmc0))
    else:
        ffmc0 = np.ma.array([ffmc0], mask=np.isnan([ffmc0]))

    # Verify temp
    if not isinstance(temp, (int, float, np.ndarray)):
        raise TypeError('temp must be either int, float or numpy ndarray data types')
    elif isinstance(temp, np.ndarray):
        temp = np.ma.array(temp, mask=np.isnan(temp))
    else:
        temp = np.ma.array([temp], mask=np.isnan([temp]))

    # Verify rh
    if not isinstance(rh, (int, float, np.ndarray)):
        raise TypeError('rh must be either int, float or numpy ndarray data types')
    elif isinstance(rh, np.ndarray):
        rh = np.ma.array(rh, mask=np.isnan(rh))
    else:
        rh = np.ma.array([rh], mask=np.isnan([rh]))

    # Verify wind
    if not isinstance(wind, (int, float, np.ndarray)):
        raise TypeError('wind must be either int, float or numpy ndarray data types')
    elif isinstance(wind, np.ndarray):
        wind = np.ma.array(wind, mask=np.isnan(wind))
    else:
        wind = np.ma.array([wind], mask=np.isnan([wind]))

    # Verify precip
    if not isinstance(precip, (int, float, np.ndarray)):
        raise TypeError('precip must be either int, float or numpy ndarray data types')
    elif isinstance(precip, np.ndarray):
        precip = np.ma.array(precip, mask=np.isnan(precip))
    else:
        precip = np.ma.array([precip], mask=np.isnan([precip]))

    # ### PREVIOUS HOURS ESTIMATED FINE FUEL MOISTURE CONTENT
    # This equation has been revised from Van Wagner (1977) to match Van Wagner (1987)
    # Doing this uses the newer FF scale, over the old F scale (per Anderson 2009)
    if use_precise_values:
        # This equation uses a more precise multiplier (~147.27723 instead of 147.2) per Wang et al. (2017)
        ffmc_coeff = 250.0 * 59.5 / 101.0
        m0 = ffmc_coeff * (101 - ffmc0) / (59.5 + ffmc0)
    else:
        m0 = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)

    # ### DRYING PHASE
    # Equilibrium Moisture Content (E)
    # Drying from above
    ed = (0.942 * rh**0.679 + 11 * np.exp((rh - 100) / 10) +
          0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))
    # Wetting from below
    ew = (0.618 * rh**0.753 + 10 * np.exp((rh - 100) / 10) +
          0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))

    # LOG DRYING RATE (k)
    # Calculate wetting rate
    k0d = (0.424 * (1 - (rh / 100)**1.7) +
           0.0694 * (wind**0.5) * (1 - (rh / 100)**8))
    kd = k0d * 0.0579 * np.exp(0.0365 * temp)
    # Calculate drying rate
    k0w = (0.424 * (1 - ((100 - rh) / 100)**1.7) +
           0.0694 * (wind**0.5) * (1 - ((100 - rh) / 100)**8))
    kw = k0w * 0.0579 * np.exp(0.0365 * temp)

    # Calculate drying/wetting moisture content (mdw)
    if use_precise_values:
        # USES DAILY EQUATIONS FOR BETTER PRECISION
        mdw = np.ma.where(m0 > ed,
                          ed + (m0 - ed) * (10**-kd),
                          np.ma.where(m0 < ew,
                                      ew - (ew - m0) * (10**-kw),
                                      m0))
    else:
        # ORIGINAL HOURLY EQUATIONS
        mdw = np.ma.where(m0 > ed,
                          ed + (m0 - ed) * np.exp(-2.303 * kd),
                          np.ma.where(m0 < ew,
                                      ew - (ew - m0) * np.exp(-2.303 * kw),
                                      m0))

    # ### RAINFALL PHASE
    # Rainfall Effectiveness (delta_mrf)
    np.seterr(over='ignore')
    delta_mrf = np.ma.where(precip > 0,
                            m0 + 42.5 * precip * np.exp(-100 / (251 - m0)) * (1 - np.exp(-6.93 / precip)),
                            mdw)
    np.seterr(over='warn')

    # Rainfall Moisture
    m = np.ma.where(m0 > 150,
                    delta_mrf + 0.0015 * ((m0 - 150)**2) * (precip**0.5),
                    delta_mrf)

    # Cap m at 250 to reflect max moisture content of pine litter
    m = np.ma.where(m > 250,
                    250,
                    m)

    # Set moisture minimum to 0
    m = np.ma.where(m < 0,
                    0,
                    m)

    # ### RETURN FINAL FFMC VALUE
    # This equation has been revised from Van Wagner (1977) to match Van Wagner (1987)
    # Doing this uses the newer FF scale, over the old F scale (per Anderson 2009)
    if use_precise_values:
        # This equation uses a more precise multiplier (147.27723 instead of 147.2) per Wang et al. (2017)
        ffmc_coeff = 250.0 * 59.5 / 101.0
        ffmc = 59.5 * (250 - m) / (ffmc_coeff + m)
    else:
        ffmc = 59.5 * (250 - m) / (147.2 + m)

    # Restrict FFMC values to range between 0 and 101
    ffmc[ffmc > 101] = 101
    ffmc[ffmc < 0] = 0

    if return_array:
        return ffmc.data
    else:
        return ffmc.data[0]


def dailyFFMC(ffmc0: Union[int, float, np.ndarray],
              temp: Union[int, float, np.ndarray],
              rh: Union[int, float, np.ndarray],
              wind: Union[int, float, np.ndarray],
              precip: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate daily FFMC values per Van Wagner (1987).
    :param ffmc0: yesterday's FFMC value (unitless code)
    :param temp: temperature value (C)
    :param rh: relative humidity value (%)
    :param wind: wind speed value (km/h)
    :param precip: precipitation value (mm)
    :return: the daily FFMC value
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [ffmc0, temp, rh, wind, precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify ffmc0
    if not isinstance(ffmc0, (int, float, np.ndarray)):
        raise TypeError('ffmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(ffmc0, np.ndarray):
        ffmc0 = np.ma.array(ffmc0, mask=np.isnan(ffmc0))
    else:
        ffmc0 = np.ma.array([ffmc0], mask=np.isnan([ffmc0]))

    # Verify temp
    if not isinstance(temp, (int, float, np.ndarray)):
        raise TypeError('temp must be either int, float or numpy ndarray data types')
    elif isinstance(temp, np.ndarray):
        temp = np.ma.array(temp, mask=np.isnan(temp))
    else:
        temp = np.ma.array([temp], mask=np.isnan([temp]))

    # Verify rh
    if not isinstance(rh, (int, float, np.ndarray)):
        raise TypeError('rh must be either int, float or numpy ndarray data types')
    elif isinstance(rh, np.ndarray):
        rh = np.ma.array(rh, mask=np.isnan(rh))
    else:
        rh = np.ma.array([rh], mask=np.isnan([rh]))

    # Verify wind
    if not isinstance(wind, (int, float, np.ndarray)):
        raise TypeError('wind must be either int, float or numpy ndarray data types')
    elif isinstance(wind, np.ndarray):
        wind = np.ma.array(wind, mask=np.isnan(wind))
    else:
        wind = np.ma.array([wind], mask=np.isnan([wind]))

    # Verify precip
    if not isinstance(precip, (int, float, np.ndarray)):
        raise TypeError('precip must be either int, float or numpy ndarray data types')
    elif isinstance(precip, np.ndarray):
        precip = np.ma.array(precip, mask=np.isnan(precip))
    else:
        precip = np.ma.array([precip], mask=np.isnan([precip]))

    # ### YESTERDAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
    m0 = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)

    # ### DRYING PHASE
    # LOG DRYING RATE (k)
    # Calculate wetting rate
    k0d = (0.424 * (1 - (rh / 100)**1.7) +
           (0.0694 * (wind**0.5)) * (1 - (rh / 100)**8))
    kd = k0d * 0.581 * np.exp(0.0365 * temp)
    # Calculate drying rate
    k0w = (0.424 * (1 - ((100 - rh) / 100)**1.7) +
           (0.0694 * wind**0.5) * (1 - ((100 - rh) / 100)**8))
    kw = k0w * 0.581 * np.exp(0.0365 * temp)

    # Equilibrium Moisture Content (E)
    # Drying from above
    ed = (0.942 * rh**0.679 + 11 * np.exp((rh - 100) / 10) +
          0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))
    # Wetting from below
    ew = (0.618 * rh**0.753 + 10 * np.exp((rh - 100) / 10) +
          0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))

    # MOISTURE CONTENT (m)
    m = np.ma.where(m0 > ed,
                    ed + (m0 - ed) * 10**-kd,
                    np.ma.where(m0 < ew,
                                ew - (ew - m0) * 10**-kw,
                                m0))

    # ### RAINFALL PHASE
    # Rainfall Effectiveness (delta_mrf)
    rf = precip - 0.5
    np.seterr(over='ignore')
    delta_mrf = 42.5 * np.exp(-100 / (251 - m0)) * (1 - np.exp(-6.93 / rf))
    np.seterr(over='warn')

    # Rainfall Moisture
    m = np.ma.where(rf <= 0,
                    m,
                    np.ma.where(m0 > 150,
                                m + delta_mrf + 0.0015 * ((m0 - 150)**2) * (rf**0.5),
                                m + delta_mrf))

    # Cap m at 250 to reflect max moisture content of pine litter
    m = np.ma.where(m > 250,
                    250,
                    m)

    # Set moisture minimum to 0
    m = np.ma.where(m < 0,
                    0,
                    m)

    # ### RETURN FINAL FFMC VALUE
    ffmc = 59.5 * (250 - m) / (147.2 + m)

    # Restrict FFMC values to range between 0 and 101
    ffmc[ffmc > 101] = 101
    ffmc[ffmc < 0] = 0

    if return_array:
        return ffmc.data
    else:
        return ffmc.data[0]


def dailyDMC(dmc0: Union[int, float, np.ndarray],
             temp: Union[int, float, np.ndarray],
             rh: Union[int, float, np.ndarray],
             precip: Union[int, float, np.ndarray],
             month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate today's DMC per Van Wagner (1987).
    :param dmc0: yesterday's DMC value (unitless code)
    :param temp: today's temperature value (C)
    :param rh: today's relative humidity value (%)
    :param precip: today's precipitation value (mm)
    :param month: the current month (e.g., 9, '09', 'September')
    :return: the current DMC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [dmc0, temp, rh, precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify dmc0
    if not isinstance(dmc0, (int, float, np.ndarray)):
        raise TypeError('dmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(dmc0, np.ndarray):
        dmc0 = np.ma.array(dmc0, mask=np.isnan(dmc0))
    else:
        dmc0 = np.ma.array([dmc0], mask=np.isnan([dmc0]))

    # Verify temp
    if not isinstance(temp, (int, float, np.ndarray)):
        raise TypeError('temp must be either int, float or numpy ndarray data types')
    elif isinstance(temp, np.ndarray):
        temp = np.ma.array(temp, mask=np.isnan(temp))
    else:
        temp = np.ma.array([temp], mask=np.isnan([temp]))

    # Verify rh
    if not isinstance(rh, (int, float, np.ndarray)):
        raise TypeError('rh must be either int, float or numpy ndarray data types')
    elif isinstance(rh, np.ndarray):
        rh = np.ma.array(rh, mask=np.isnan(rh))
    else:
        rh = np.ma.array([rh], mask=np.isnan([rh]))

    # Verify precip
    if not isinstance(precip, (int, float, np.ndarray)):
        raise TypeError('precip must be either int, float or numpy ndarray data types')
    elif isinstance(precip, np.ndarray):
        precip = np.ma.array(precip, mask=np.isnan(precip))
    else:
        precip = np.ma.array([precip], mask=np.isnan([precip]))

    # Verify month
    if not isinstance(month, (int, str)):
        raise TypeError('month must be either int or string data types')
    elif isinstance(month, int):
        month = str(month).zfill(2)

    # ### YESTERDAYS MOISTURE CONTENT
    np.seterr(over='ignore')
    m0 = 20 + np.exp((244.72 - dmc0) / 43.43)
    np.seterr(over='warn')

    # ### DRYING PHASE
    dmc_daylength_dict = {
        'January': 6.5,
        'February': 7.5,
        'March': 9,
        'April': 12.8,
        'May': 13.9,
        'June': 13.9,
        'July': 12.4,
        'August': 10.9,
        'September': 9.4,
        'October': 8,
        'November': 7,
        'December': 6,
        '01': 6.5,
        '02': 7.5,
        '03': 9,
        '04': 12.8,
        '05': 13.9,
        '06': 13.9,
        '07': 12.4,
        '08': 10.9,
        '09': 9.4,
        '10': 8,
        '11': 7,
        '12': 6
    }
    # Log Drying Rate
    le = dmc_daylength_dict.get(month, None)
    if le is None:
        raise ValueError(f'Month value is invalid: {month}')
    k = 1.894 * (temp + 1.1) * (100 - rh) * le * 10**-6

    # ### RAINFALL PHASE
    np.seterr(divide='ignore')
    b = np.ma.where(dmc0 <= 33,
                    100 / (0.5 + 0.3 * dmc0),
                    np.ma.where(dmc0 <= 65,
                                14 - 1.3 * np.log(dmc0),
                                6.2 * np.log(dmc0) - 17.2))
    np.seterr(divide='warn')

    # Effective rain (re)
    re = np.ma.where(precip > 1.5,
                     (0.92 * precip) - 1.27,
                     0)

    # Moisture content after rain (mr)
    mr = m0 + 1000 * re / (48.77 + b * re)
    mr[mr < 0] = 0

    # ### RETURN FINAL DMC VALUES
    dmc = np.ma.where(precip > 1.5,
                      (244.72 - 43.43 * np.log(mr - 20)) + 100 * k,
                      dmc0 + 100 * k)

    # Ensure DMC >= 0
    dmc[dmc < 0] = 0

    if return_array:
        return dmc.data
    else:
        return dmc.data[0]


def dailyDC(dc0: Union[int, float, np.ndarray],
            temp: Union[int, float, np.ndarray],
            precip: Union[int, float, np.ndarray],
            month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate today's DMC per Van Wagner (1987).
    :param dc0: yesterday's DC value (unitless code)
    :param temp: today's temperature value (C)
    :param precip: today's precipitation value (mm)
    :param month: the current month (e.g., 9, '09', 'September')
    :return: the current DC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [dc0, temp, precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify dc0
    if not isinstance(dc0, (int, float, np.ndarray)):
        raise TypeError('dc0 must be either int, float or numpy ndarray data types')
    elif isinstance(dc0, np.ndarray):
        dc0 = np.ma.array(dc0, mask=np.isnan(dc0))
    else:
        dc0 = np.ma.array([dc0], mask=np.isnan([dc0]))

    # Verify temp
    if not isinstance(temp, (int, float, np.ndarray)):
        raise TypeError('temp must be either int, float or numpy ndarray data types')
    elif isinstance(temp, np.ndarray):
        temp = np.ma.array(temp, mask=np.isnan(temp))
    else:
        temp = np.ma.array([temp], mask=np.isnan([temp]))

    # Verify precip
    if not isinstance(precip, (int, float, np.ndarray)):
        raise TypeError('precip must be either int, float or numpy ndarray data types')
    elif isinstance(precip, np.ndarray):
        precip = np.ma.array(precip, mask=np.isnan(precip))
    else:
        precip = np.ma.array([precip], mask=np.isnan([precip]))

    # Verify month
    if not isinstance(month, (int, str)):
        raise TypeError('month must be either int or string data types')
    elif isinstance(month, int):
        month = str(month).zfill(2)

    # ### YESTERDAYS MOISTURE EQUIVALENT VALUE
    q0 = 800 / np.exp(dc0 / 400)

    # Potential Evapotranspiration (v)
    lf = dc_daylength_dict.get(month, None)
    if lf is None:
        raise ValueError(f'Month value is invalid: {month}')
    v = 0.36 * (temp + 2.8) + lf

    # ### RAINFALL PHASE
    # Effective rainfall (rd)
    rd = np.ma.where(precip > 2.8,
                     (0.83 * precip) - 1.27,
                     0)
    # Moisture content after rain (mr)
    qr = q0 + 3.937 * rd

    # ### RETURN FINAL DMC VALUES
    np.seterr(divide='ignore')
    dc = np.ma.where(precip > 2.8,
                     400 * np.log(800 / qr) + 0.5 * v,
                     dc0 + 0.5 * v)
    np.seterr(divide='warn')

    # Ensure DC >= 0
    dc[dc < 0] = 0

    if return_array:
        return dc.data
    else:
        return dc.data[0]


def dailyISI(wind: Union[int, float, np.ndarray],
             ffmc: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate ISI per Van Wagner (1987).
    The daily ISI equation is used for both hourly and daily ISI calculations.\n
    -- For hourly ISI, use the prior hour's wind and FFMC values.\n
    -- For daily ISI, use the current day's noon-time (1200) wind, and the prior day's noon-time (1200) FFMC value.\n
    :param wind: 10-m wind speed (km/h)
    :param ffmc: current FFMC value (unitless code)
    :return: the current ISI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [wind, ffmc]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify wind
    if not isinstance(wind, (int, float, np.ndarray)):
        raise TypeError('wind must be either int, float or numpy ndarray data types')
    elif isinstance(wind, np.ndarray):
        wind = np.ma.array(wind, mask=np.isnan(wind))
    else:
        wind = np.ma.array([wind], mask=np.isnan([wind]))

    # Verify ffmc
    if not isinstance(ffmc, (int, float, np.ndarray)):
        raise TypeError('ffmc must be either int, float or numpy ndarray data types')
    elif isinstance(ffmc, np.ndarray):
        ffmc = np.ma.array(ffmc, mask=np.isnan(ffmc))
    else:
        ffmc = np.ma.array([ffmc], mask=np.isnan([ffmc]))

    # ### CURRENT ESTIMATED FINE FUEL MOISTURE CONTENT
    m = 147.2 * (101 - ffmc) / (59.5 + ffmc)

    # ### WIND COMPONENT OF ISI
    fw = np.exp(0.05039 * wind)

    # ### FFMC COMPONENT OF ISI
    ff = 91.9 * np.exp(-0.1386 * m) * (1 + ((m**5.31) / 49300000))

    # ### RETURN FINAL ISI VALUE
    isi = 0.208 * fw * ff
    if return_array:
        return isi.data
    else:
        return isi.data[0]


def dailyBUI(dmc: Union[int, float, np.ndarray],
             dc: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate daily Build Up Index values per Van Wagner (1987).
    :param dmc: current DMC value (unitless code)
    :param dc: current DC value (unitless code)
    :return: current BUI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [dmc, dc]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify dmc
    if not isinstance(dmc, (int, float, np.ndarray)):
        raise TypeError('dmc must be either int, float or numpy ndarray data types')
    elif isinstance(dmc, np.ndarray):
        dmc = np.ma.array(dmc, mask=np.isnan(dmc))
    else:
        dmc = np.ma.array([dmc], mask=np.isnan([dmc]))

    # Verify dc
    if not isinstance(dc, (int, float, np.ndarray)):
        raise TypeError('dc must be either int, float or numpy ndarray data types')
    elif isinstance(dc, np.ndarray):
        dc = np.ma.array(dc, mask=np.isnan(dc))
    else:
        dc = np.ma.array([dc], mask=np.isnan([dc]))

    # ### RETURN FINAL BUI VALUE
    bui = np.ma.where(dmc == 0,
                      0,
                      np.ma.where(dmc <= 0.4 * dc,
                                  0.8 * dmc * dc / (dmc + 0.4 * dc),
                                  dmc - (1 - (0.8 * dc / (dmc + 0.4 * dc))) * (0.92 + (0.0114 * dmc)**1.7)))
    if return_array:
        return bui.data
    else:
        return bui.data[0]


def dailyFWI(isi: Union[int, float, np.ndarray],
             bui: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate FWI per Van Wagner (1987).
    The daily FWI equation is used for both hourly and daily FWI calculations.\n
    -- For hourly FWI, use the current hour's ISI and BUI values.\n
    -- For daily FWI, use the current day's noon-time (1200) ISI and BUI values.\n
    :param isi: the current ISI value (unitless code)
    :param bui: the current BUI value (unitless code)
    :return: the current FWI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [isi, bui]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify isi
    if not isinstance(isi, (int, float, np.ndarray)):
        raise TypeError('isi must be either int, float or numpy ndarray data types')
    elif isinstance(isi, np.ndarray):
        isi = np.ma.array(isi, mask=np.isnan(isi))
    else:
        isi = np.ma.array([isi], mask=np.isnan([isi]))

    # Verify bui
    if not isinstance(bui, (int, float, np.ndarray)):
        raise TypeError('bui must be either int, float or numpy ndarray data types')
    elif isinstance(bui, np.ndarray):
        bui = np.ma.array(bui, mask=np.isnan(bui))
    else:
        bui = np.ma.array([bui], mask=np.isnan([bui]))

    # ### DUFF MOISTURE FUNCTION (fD)
    np.seterr(over='ignore')
    fd = np.ma.where(bui <= 80,
                     0.626 * (bui**0.809) + 2,
                     1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    np.seterr(over='warn')

    # ### INTERMEDIATE FWI (B)
    b = 0.1 * isi * fd

    # ### RETURN FINAL FWI VALUE
    np.seterr(divide='ignore')
    fwi = np.ma.where(b > 1,
                      np.exp(2.72 * (0.434 * np.log(b))**0.647),
                      b)
    np.seterr(divide='warn')
    if return_array:
        return fwi.data
    else:
        return fwi.data[0]


def dailyDSR(fwi: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate the Daily Severity Rating (DSR) per Van Wagner (1987)
    :param fwi: current FWI value (unitless code)
    :return: current DSR value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if isinstance(fwi, np.ndarray):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify fwi
    if not isinstance(fwi, (int, float, np.ndarray)):
        raise TypeError('fwi must be either int, float or numpy ndarray data types')
    elif isinstance(fwi, np.ndarray):
        fwi = np.ma.array(fwi, mask=np.isnan(fwi))
    else:
        fwi = np.ma.array([fwi], mask=np.isnan([fwi]))

    # ### RETURN DSR VALUE
    dsr = 0.0272 * fwi**1.77
    if return_array:
        return dsr.data
    else:
        return dsr.data[0]


def startupDC(dc_stop: Union[int, float, np.ndarray],
              moist_stop: Union[int, float, np.ndarray],
              moist_start: Union[int, float, np.ndarray],
              precip_ow: Union[int, float, np.ndarray],
              temp: Union[int, float, np.ndarray],
              month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate the DC startup values after overwintering.\n
    This function implements new procedures outlined in Hanes and Wotton (2024).
    :param dc_stop: DC value of the last day of FWI System calculation prior to overwintering (unitless code)
    :param moist_stop: moisture value from the last day of FWI system calculations prior to overwintering (%)
    :param moist_start: moisture value for the first day of FWI System calculations since overwintering (%)
    :param precip_ow: total precipitation throughout the overwintering period (mm)
    :param temp: today's temperature value (C)
    :param month: the current month (e.g., 9, '09', 'September')
    :return: startup DC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [dc_stop, moist_stop, moist_start, precip_ow]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify dc_stop
    if not isinstance(dc_stop, (int, float, np.ndarray)):
        raise TypeError('dc_stop must be either int, float or numpy ndarray data types')
    elif isinstance(dc_stop, np.ndarray):
        dc_stop = np.ma.array(dc_stop, mask=np.isnan(dc_stop))
    else:
        dc_stop = np.ma.array([dc_stop], mask=np.isnan([dc_stop]))

    # Verify moist_stop
    if not isinstance(moist_stop, (int, float, np.ndarray)):
        raise TypeError('moist_stop must be either int, float or numpy ndarray data types')
    elif isinstance(moist_stop, np.ndarray):
        moist_stop = np.ma.array(moist_stop, mask=np.isnan(moist_stop))
    else:
        moist_stop = np.ma.array([moist_stop], mask=np.isnan([moist_stop]))

    # Verify moist_start
    if not isinstance(moist_start, (int, float, np.ndarray)):
        raise TypeError('moist_start must be either int, float or numpy ndarray data types')
    elif isinstance(moist_start, np.ndarray):
        moist_start = np.ma.array(moist_start, mask=np.isnan(moist_start))
    else:
        moist_start = np.ma.array([moist_start], mask=np.isnan([moist_start]))

    # Verify precip_ow
    if not isinstance(precip_ow, (int, float, np.ndarray)):
        raise TypeError('p_ow must be either int, float or numpy ndarray data types')
    elif isinstance(precip_ow, np.ndarray):
        precip_ow = np.ma.array(precip_ow, mask=np.isnan(precip_ow))
    else:
        precip_ow = np.ma.array([precip_ow], mask=np.isnan([precip_ow]))

    # Verify temp
    if not isinstance(temp, (int, float, np.ndarray)):
        raise TypeError('temp must be either int, float or numpy ndarray data types')
    elif isinstance(temp, np.ndarray):
        temp = np.ma.array(temp, mask=np.isnan(temp))
    else:
        temp = np.ma.array([temp], mask=np.isnan([temp]))

    # Verify month
    if not isinstance(month, (int, str)):
        raise TypeError('month must be either int or string data types')
    elif isinstance(month, int):
        month = str(month).zfill(2)

    # Potential Evapotranspiration (v)
    lf = dc_daylength_dict.get(month, None)
    if lf is None:
        raise ValueError(f'Month value is invalid: {month}')
    v = 0.36 * (temp + 2.8) + lf

    # Carryover fraction of the fall moisture deficit
    # New approach: a is always 1 to remove a potential source of error
    # on the front end of the overwinter calculation
    a = 1

    # Fraction of winter precipitation effective at recharging depleted moisture reserves in spring
    b = np.ma.where(moist_start < moist_stop,
                    0,
                    (moist_start - moist_stop) / moist_stop)

    # Final fall moisture equivalent
    q_f = 800 * np.exp(-dc_stop / 400)

    # Starting spring moisture equivalent
    q_s = a * q_f + b * (3.937 * precip_ow)

    # ### RETURN DC STARTUP VALUE
    np.seterr(divide='ignore')
    dc_start = 400 * np.log(800 / q_s) + 0.5 * v
    np.seterr(divide='warn')

    # Ensure DC >= 0
    dc_start[dc_start < 0] = 0

    if return_array:
        return dc_start.data
    else:
        return dc_start.data[0]
