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
    '12': -1.6,
}


"""
The diurnalFFMC function is still under development...
"""
# def diurnalFFMC(_time_lookup, _rh, _ffmc0_1600, _ffmc_1600):
#     """
#     Function to calculate diurnal Hourly FFMC per Lawson et al. (1996)
#     :param _time_lookup: string or list of strings; time of day to estimate diurnal FFMC
#     :param _rh: int, float or numpy ndarray; relative humidity value (%)
#     :param _ffmc0_1600: float or numpy ndarray; yesterday's FFMC value at 1600-hr (unitless code)
#     :param _ffmc_1600: float or numpy ndarray; today's FFMC value at 1600-hr (unitless code)
#     :return: float or numpy ndarray; diurnal FFMC value (unitless code)
#     """
#     if not isinstance(_time_lookup, (str, np.ndarray)):
#         raise TypeError('_time_lookup must be either string or numpy ndarray data types')
#     if not isinstance(_rh, (int, float, np.ndarray)):
#         raise TypeError('_rh must be either int, float or numpy ndarray data types')
#     if not isinstance(_ffmc0_1600, (int, float, np.ndarray)):
#         raise TypeError('_ffmc0_1600 must be either int, float or numpy ndarray data types')
#     if not isinstance(_ffmc_1600, (int, float, np.ndarray)):
#         raise TypeError('_ffmc_1600 must be either int, float or numpy ndarray data types')
#
#     # ### YESTERDAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
#     m0_1600 = 147.2 * (101 - _ffmc0_1600) / (59.5 + _ffmc0_1600)
#
#     # ### TODAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
#     m_1600 = 147.2 * (101 - _ffmc_1600) / (59.5 + _ffmc_1600)
#     return


def hourlyFFMC(_ffmc0: Union[int, float, np.ndarray],
               _temp: Union[int, float, np.ndarray],
               _rh: Union[int, float, np.ndarray],
               _wind: Union[int, float, np.ndarray],
               _precip: Union[int, float, np.ndarray],
               _use_precise_values: bool = False) -> Union[float, np.ndarray]:
    """
    Function to calculate hourly FFMC values per Van Wagner (1977) and Alexander et al. (1984).
    :param _ffmc0: previous hour's FFMC value (unitless code)
    :param _temp: temperature value (C)
    :param _rh: relative humidity value (%)
    :param _wind: wind speed value (km/h)
    :param _precip: precipitation value (mm)
    :param _use_precise_values: use higher precision for m0 & Daily FFMC equations for drying/wetting moisture
    :return: the hourly FFMC value
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_ffmc0, _temp, _rh, _wind, _precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _ffmc0
    if not isinstance(_ffmc0, (int, float, np.ndarray)):
        raise TypeError('_ffmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(_ffmc0, np.ndarray):
        _ffmc0 = np.ma.array(_ffmc0, mask=np.isnan(_ffmc0))
    else:
        _ffmc0 = np.ma.array([_ffmc0], mask=np.isnan([_ffmc0]))

    # Verify _temp
    if not isinstance(_temp, (int, float, np.ndarray)):
        raise TypeError('_temp must be either int, float or numpy ndarray data types')
    elif isinstance(_temp, np.ndarray):
        _temp = np.ma.array(_temp, mask=np.isnan(_temp))
    else:
        _temp = np.ma.array([_temp], mask=np.isnan([_temp]))

    # Verify _rh
    if not isinstance(_rh, (int, float, np.ndarray)):
        raise TypeError('_rh must be either int, float or numpy ndarray data types')
    elif isinstance(_rh, np.ndarray):
        _rh = np.ma.array(_rh, mask=np.isnan(_rh))
    else:
        _rh = np.ma.array([_rh], mask=np.isnan([_rh]))

    # Verify _wind
    if not isinstance(_wind, (int, float, np.ndarray)):
        raise TypeError('_wind must be either int, float or numpy ndarray data types')
    elif isinstance(_wind, np.ndarray):
        _wind = np.ma.array(_wind, mask=np.isnan(_wind))
    else:
        _wind = np.ma.array([_wind], mask=np.isnan([_wind]))

    # Verify _precip
    if not isinstance(_precip, (int, float, np.ndarray)):
        raise TypeError('_precip must be either int, float or numpy ndarray data types')
    elif isinstance(_precip, np.ndarray):
        _precip = np.ma.array(_precip, mask=np.isnan(_precip))
    else:
        _precip = np.ma.array([_precip], mask=np.isnan([_precip]))

    # ### PREVIOUS HOURS ESTIMATED FINE FUEL MOISTURE CONTENT
    # This equation has been revised from Van Wagner (1977) to match Van Wagner (1987)
    # Doing this uses the newer FF scale, over the old F scale (per Anderson 2009)
    if _use_precise_values:
        # This equation uses a more precise multiplier (147.27723 instead of 147.2) per Wang et al. (2017)
        m0 = 147.27723 * (101 - _ffmc0) / (59.5 + _ffmc0)
    else:
        m0 = 147.2 * (101 - _ffmc0) / (59.5 + _ffmc0)

    # ### DRYING PHASE
    # Equilibrium Moisture Content (E)
    # Drying from above
    ed = (0.942 * _rh**0.679 + 11 * np.exp((_rh - 100) / 10) +
          0.18 * (21.1 - _temp) * (1 - np.exp(-0.115 * _rh)))
    # Wetting from below
    ew = (0.618 * _rh**0.753 + 10 * np.exp((_rh - 100) / 10) +
          0.18 * (21.1 - _temp) * (1 - np.exp(-0.115 * _rh)))

    # LOG DRYING RATE (k)
    # Calculate wetting rate
    k0d = (0.424 * (1 - (_rh / 100)**1.7) +
           0.0694 * (_wind**0.5) * (1 - (_rh / 100)**8))
    kd = k0d * 0.0579 * np.exp(0.0365 * _temp)
    # Calculate drying rate
    k0w = (0.424 * (1 - ((100 - _rh) / 100)**1.7) +
           0.0694 * (_wind**0.5) * (1 - ((100 - _rh) / 100)**8))
    kw = k0w * 0.0579 * np.exp(0.0365 * _temp)

    # Calculate drying/wetting moisture content (mdw)
    if _use_precise_values:
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
    delta_mrf = np.ma.where(_precip > 0,
                            m0 + 42.5 * _precip * np.exp(-100 / (251 - m0)) * (1 - np.exp(-6.93 / _precip)),
                            mdw)
    np.seterr(over='warn')

    # Rainfall Moisture
    m = np.ma.where(m0 > 150,
                    delta_mrf + 0.0015 * ((m0 - 150)**2) * (_precip**0.5),
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
    if _use_precise_values:
        # This equation uses a more precise multiplier (147.27723 instead of 147.2) per Wang et al. (2017)
        ffmc = 59.5 * (250 - m) / (147.27723 + m)
    else:
        ffmc = 59.5 * (250 - m) / (147.2 + m)

    # Restrict FFMC values to range between 0 and 101
    ffmc[ffmc > 101] = 101
    ffmc[ffmc < 0] = 0

    if return_array:
        return ffmc.data
    else:
        return ffmc.data[0]


def dailyFFMC(_ffmc0: Union[int, float, np.ndarray],
              _temp: Union[int, float, np.ndarray],
              _rh: Union[int, float, np.ndarray],
              _wind: Union[int, float, np.ndarray],
              _precip: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate daily FFMC values per Van Wagner (1987).
    :param _ffmc0: yesterday's FFMC value (unitless code)
    :param _temp: temperature value (C)
    :param _rh: relative humidity value (%)
    :param _wind: wind speed value (km/h)
    :param _precip: precipitation value (mm)
    :return: the daily FFMC value
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_ffmc0, _temp, _rh, _wind, _precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _ffmc0
    if not isinstance(_ffmc0, (int, float, np.ndarray)):
        raise TypeError('_ffmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(_ffmc0, np.ndarray):
        _ffmc0 = np.ma.array(_ffmc0, mask=np.isnan(_ffmc0))
    else:
        _ffmc0 = np.ma.array([_ffmc0], mask=np.isnan([_ffmc0]))

    # Verify _temp
    if not isinstance(_temp, (int, float, np.ndarray)):
        raise TypeError('_temp must be either int, float or numpy ndarray data types')
    elif isinstance(_temp, np.ndarray):
        _temp = np.ma.array(_temp, mask=np.isnan(_temp))
    else:
        _temp = np.ma.array([_temp], mask=np.isnan([_temp]))

    # Verify _rh
    if not isinstance(_rh, (int, float, np.ndarray)):
        raise TypeError('_rh must be either int, float or numpy ndarray data types')
    elif isinstance(_rh, np.ndarray):
        _rh = np.ma.array(_rh, mask=np.isnan(_rh))
    else:
        _rh = np.ma.array([_rh], mask=np.isnan([_rh]))

    # Verify _wind
    if not isinstance(_wind, (int, float, np.ndarray)):
        raise TypeError('_wind must be either int, float or numpy ndarray data types')
    elif isinstance(_wind, np.ndarray):
        _wind = np.ma.array(_wind, mask=np.isnan(_wind))
    else:
        _wind = np.ma.array([_wind], mask=np.isnan([_wind]))

    # Verify _precip
    if not isinstance(_precip, (int, float, np.ndarray)):
        raise TypeError('_precip must be either int, float or numpy ndarray data types')
    elif isinstance(_precip, np.ndarray):
        _precip = np.ma.array(_precip, mask=np.isnan(_precip))
    else:
        _precip = np.ma.array([_precip], mask=np.isnan([_precip]))

    # ### YESTERDAY'S ESTIMATED FINE FUEL MOISTURE CONTENT
    m0 = 147.2 * (101 - _ffmc0) / (59.5 + _ffmc0)

    # ### DRYING PHASE
    # LOG DRYING RATE (k)
    # Calculate wetting rate
    k0d = (0.424 * (1 - (_rh / 100)**1.7) +
           (0.0694 * (_wind**0.5)) * (1 - (_rh / 100)**8))
    kd = k0d * 0.581 * np.exp(0.0365 * _temp)
    # Calculate drying rate
    k0w = (0.424 * (1 - ((100 - _rh) / 100)**1.7) +
           (0.0694 * _wind**0.5) * (1 - ((100 - _rh) / 100)**8))
    kw = k0w * 0.581 * np.exp(0.0365 * _temp)

    # Equilibrium Moisture Content (E)
    # Drying from above
    ed = (0.942 * _rh**0.679 + 11 * np.exp((_rh - 100) / 10) +
          0.18 * (21.1 - _temp) * (1 - np.exp(-0.115 * _rh)))
    # Wetting from below
    ew = (0.618 * _rh**0.753 + 10 * np.exp((_rh - 100) / 10) +
          0.18 * (21.1 - _temp) * (1 - np.exp(-0.115 * _rh)))

    # MOISTURE CONTENT (m)
    m = np.ma.where(m0 > ed,
                    ed + (m0 - ed) * 10**-kd,
                    np.ma.where(m0 < ew,
                                ew - (ew - m0) * 10**-kw,
                                m0))

    # ### RAINFALL PHASE
    # Rainfall Effectiveness (delta_mrf)
    rf = _precip - 0.5
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


def dailyDMC(_dmc0: Union[int, float, np.ndarray],
             _temp: Union[int, float, np.ndarray],
             _rh: Union[int, float, np.ndarray],
             _precip: Union[int, float, np.ndarray],
             _month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate today's DMC per Van Wagner (1987).
    :param _dmc0: yesterday's DMC value (unitless code)
    :param _temp: today's temperature value (C)
    :param _rh: today's relative humidity value (%)
    :param _precip: today's precipitation value (mm)
    :param _month: the current month (e.g., 9, '09', 'September')
    :return: the current DMC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_dmc0, _temp, _rh, _precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _dmc0
    if not isinstance(_dmc0, (int, float, np.ndarray)):
        raise TypeError('_dmc0 must be either int, float or numpy ndarray data types')
    elif isinstance(_dmc0, np.ndarray):
        _dmc0 = np.ma.array(_dmc0, mask=np.isnan(_dmc0))
    else:
        _dmc0 = np.ma.array([_dmc0], mask=np.isnan([_dmc0]))

    # Verify _temp
    if not isinstance(_temp, (int, float, np.ndarray)):
        raise TypeError('_temp must be either int, float or numpy ndarray data types')
    elif isinstance(_temp, np.ndarray):
        _temp = np.ma.array(_temp, mask=np.isnan(_temp))
    else:
        _temp = np.ma.array([_temp], mask=np.isnan([_temp]))

    # Verify _rh
    if not isinstance(_rh, (int, float, np.ndarray)):
        raise TypeError('_rh must be either int, float or numpy ndarray data types')
    elif isinstance(_rh, np.ndarray):
        _rh = np.ma.array(_rh, mask=np.isnan(_rh))
    else:
        _rh = np.ma.array([_rh], mask=np.isnan([_rh]))

    # Verify _precip
    if not isinstance(_precip, (int, float, np.ndarray)):
        raise TypeError('_precip must be either int, float or numpy ndarray data types')
    elif isinstance(_precip, np.ndarray):
        _precip = np.ma.array(_precip, mask=np.isnan(_precip))
    else:
        _precip = np.ma.array([_precip], mask=np.isnan([_precip]))

    # Verify _month
    if not isinstance(_month, (int, str)):
        raise TypeError('_month must be either int or string data types')
    elif isinstance(_month, int):
        _month = str(_month).zfill(2)

    # ### YESTERDAYS MOISTURE CONTENT
    np.seterr(over='ignore')
    m0 = 20 + np.exp((244.72 - _dmc0) / 43.43)
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
    le = dmc_daylength_dict.get(_month, None)
    if le is None:
        raise ValueError(f'Month value is invalid: {_month}')
    k = 1.894 * (_temp + 1.1) * (100 - _rh) * le * 10**-6

    # ### RAINFALL PHASE
    np.seterr(divide='ignore')
    b = np.ma.where(_dmc0 <= 33,
                    100 / (0.5 + 0.3 * _dmc0),
                    np.ma.where(_dmc0 <= 65,
                                14 - 1.3 * np.log(_dmc0),
                                6.2 * np.log(_dmc0) - 17.2))
    np.seterr(divide='warn')

    # Effective rain (re)
    re = np.ma.where(_precip > 1.5,
                     (0.92 * _precip) - 1.27,
                     0)

    # Moisture content after rain (mr)
    mr = m0 + 1000 * re / (48.77 + b * re)
    mr[mr < 0] = 0

    # ### RETURN FINAL DMC VALUES
    dmc = np.ma.where(_precip > 1.5,
                      (244.72 - 43.43 * np.log(mr - 20)) + 100 * k,
                      _dmc0 + 100 * k)

    # Ensure DMC >= 0
    dmc[dmc < 0] = 0

    if return_array:
        return dmc.data
    else:
        return dmc.data[0]


def dailyDC(_dc0: Union[int, float, np.ndarray],
            _temp: Union[int, float, np.ndarray],
            _precip: Union[int, float, np.ndarray],
            _month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate today's DMC per Van Wagner (1987).
    :param _dc0: yesterday's DC value (unitless code)
    :param _temp: today's temperature value (C)
    :param _precip: today's precipitation value (mm)
    :param _month: the current month (e.g., 9, '09', 'September')
    :return: the current DC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_dc0, _temp, _precip]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _dc0
    if not isinstance(_dc0, (int, float, np.ndarray)):
        raise TypeError('_dc0 must be either int, float or numpy ndarray data types')
    elif isinstance(_dc0, np.ndarray):
        _dc0 = np.ma.array(_dc0, mask=np.isnan(_dc0))
    else:
        _dc0 = np.ma.array([_dc0], mask=np.isnan([_dc0]))

    # Verify _temp
    if not isinstance(_temp, (int, float, np.ndarray)):
        raise TypeError('_temp must be either int, float or numpy ndarray data types')
    elif isinstance(_temp, np.ndarray):
        _temp = np.ma.array(_temp, mask=np.isnan(_temp))
    else:
        _temp = np.ma.array([_temp], mask=np.isnan([_temp]))

    # Verify _precip
    if not isinstance(_precip, (int, float, np.ndarray)):
        raise TypeError('_precip must be either int, float or numpy ndarray data types')
    elif isinstance(_precip, np.ndarray):
        _precip = np.ma.array(_precip, mask=np.isnan(_precip))
    else:
        _precip = np.ma.array([_precip], mask=np.isnan([_precip]))

    # Verify _month
    if not isinstance(_month, (int, str)):
        raise TypeError('_month must be either int or string data types')
    elif isinstance(_month, int):
        _month = str(_month).zfill(2)

    # ### YESTERDAYS MOISTURE EQUIVALENT VALUE
    q0 = 800 / np.exp(_dc0 / 400)

    # Potential Evapotranspiration (v)
    lf = dc_daylength_dict.get(_month, None)
    if lf is None:
        raise ValueError(f'Month value is invalid: {_month}')
    v = 0.36 * (_temp + 2.8) + lf

    # ### RAINFALL PHASE
    # Effective rainfall (rd)
    rd = np.ma.where(_precip > 2.8,
                     (0.83 * _precip) - 1.27,
                     0)
    # Moisture content after rain (mr)
    qr = q0 + 3.937 * rd

    # ### RETURN FINAL DMC VALUES
    np.seterr(divide='ignore')
    dc = np.ma.where(_precip > 2.8,
                     400 * np.log(800 / qr) + 0.5 * v,
                     _dc0 + 0.5 * v)
    np.seterr(divide='warn')

    # Ensure DC >= 0
    dc[dc < 0] = 0

    if return_array:
        return dc.data
    else:
        return dc.data[0]


def dailyISI(_wind: Union[int, float, np.ndarray],
             _ffmc: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate ISI per Van Wagner (1987).
    The daily ISI equation is used for both hourly and daily ISI calculations.\n
    -- For hourly ISI, use the prior hour's wind and FFMC values.\n
    -- For daily ISI, use the current day's noon-time (1200) wind, and the prior day's noon-time (1200) FFMC value.\n
    :param _wind: 10-m wind speed (km/h)
    :param _ffmc: current FFMC value (unitless code)
    :return: the current ISI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_wind, _ffmc]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _wind
    if not isinstance(_wind, (int, float, np.ndarray)):
        raise TypeError('_wind must be either int, float or numpy ndarray data types')
    elif isinstance(_wind, np.ndarray):
        _wind = np.ma.array(_wind, mask=np.isnan(_wind))
    else:
        _wind = np.ma.array([_wind], mask=np.isnan([_wind]))

    # Verify _ffmc
    if not isinstance(_ffmc, (int, float, np.ndarray)):
        raise TypeError('_ffmc must be either int, float or numpy ndarray data types')
    elif isinstance(_ffmc, np.ndarray):
        _ffmc = np.ma.array(_ffmc, mask=np.isnan(_ffmc))
    else:
        _ffmc = np.ma.array([_ffmc], mask=np.isnan([_ffmc]))

    # ### CURRENT ESTIMATED FINE FUEL MOISTURE CONTENT
    m = 147.2 * (101 - _ffmc) / (59.5 + _ffmc)

    # ### WIND COMPONENT OF ISI
    fw = np.exp(0.05039 * _wind)

    # ### FFMC COMPONENT OF ISI
    ff = 91.9 * np.exp(-0.1386 * m) * (1 + ((m**5.31) / 49300000))

    # ### RETURN FINAL ISI VALUE
    isi = 0.208 * fw * ff
    if return_array:
        return isi.data
    else:
        return isi.data[0]


def dailyBUI(_dmc: Union[int, float, np.ndarray],
             _dc: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate daily Build Up Index values per Van Wagner (1987).
    :param _dmc: current DMC value (unitless code)
    :param _dc: current DC value (unitless code)
    :return: current BUI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_dmc, _dc]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _dmc
    if not isinstance(_dmc, (int, float, np.ndarray)):
        raise TypeError('_dmc must be either int, float or numpy ndarray data types')
    elif isinstance(_dmc, np.ndarray):
        _dmc = np.ma.array(_dmc, mask=np.isnan(_dmc))
    else:
        _dmc = np.ma.array([_dmc], mask=np.isnan([_dmc]))

    # Verify _dc
    if not isinstance(_dc, (int, float, np.ndarray)):
        raise TypeError('_dc must be either int, float or numpy ndarray data types')
    elif isinstance(_dc, np.ndarray):
        _dc = np.ma.array(_dc, mask=np.isnan(_dc))
    else:
        _dc = np.ma.array([_dc], mask=np.isnan([_dc]))

    # ### RETURN FINAL BUI VALUE
    bui = np.ma.where(_dmc == 0,
                      0,
                      np.ma.where(_dmc <= 0.4 * _dc,
                                  0.8 * _dmc * _dc / (_dmc + 0.4 * _dc),
                                  _dmc - (1 - (0.8 * _dc / (_dmc + 0.4 * _dc))) * (0.92 + (0.0114 * _dmc)**1.7)))
    if return_array:
        return bui.data
    else:
        return bui.data[0]


def dailyFWI(_isi: Union[int, float, np.ndarray],
             _bui: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate FWI per Van Wagner (1987).
    The daily FWI equation is used for both hourly and daily FWI calculations.\n
    -- For hourly FWI, use the current hour's ISI and BUI values.\n
    -- For daily FWI, use the current day's noon-time (1200) ISI and BUI values.\n
    :param _isi: the current ISI value (unitless code)
    :param _bui: the current BUI value (unitless code)
    :return: the current FWI value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [_isi, _bui]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _isi
    if not isinstance(_isi, (int, float, np.ndarray)):
        raise TypeError('_isi must be either int, float or numpy ndarray data types')
    elif isinstance(_isi, np.ndarray):
        _isi = np.ma.array(_isi, mask=np.isnan(_isi))
    else:
        _isi = np.ma.array([_isi], mask=np.isnan([_isi]))

    # Verify _bui
    if not isinstance(_bui, (int, float, np.ndarray)):
        raise TypeError('_bui must be either int, float or numpy ndarray data types')
    elif isinstance(_bui, np.ndarray):
        _bui = np.ma.array(_bui, mask=np.isnan(_bui))
    else:
        _bui = np.ma.array([_bui], mask=np.isnan([_bui]))

    # ### DUFF MOISTURE FUNCTION (fD)
    np.seterr(over='ignore')
    fd = np.ma.where(_bui <= 80,
                     0.626 * (_bui**0.809) + 2,
                     1000 / (25 + 108.64 * np.exp(-0.023 * _bui)))
    np.seterr(over='warn')

    # ### INTERMEDIATE FWI (B)
    b = 0.1 * _isi * fd

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


def dailyDSR(_fwi: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function to calculate the Daily Severity Rating (DSR) per Van Wagner (1987)
    :param _fwi: current FWI value (unitless code)
    :return: current DSR value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if isinstance(_fwi, np.ndarray):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify _fwi
    if not isinstance(_fwi, (int, float, np.ndarray)):
        raise TypeError('_fwi must be either int, float or numpy ndarray data types')
    elif isinstance(_fwi, np.ndarray):
        _fwi = np.ma.array(_fwi, mask=np.isnan(_fwi))
    else:
        _fwi = np.ma.array([_fwi], mask=np.isnan([_fwi]))

    # ### RETURN DSR VALUE
    dsr = 0.0272 * _fwi**1.77
    if return_array:
        return dsr.data
    else:
        return dsr.data[0]


def startupDC(dc_f: Union[int, float, np.ndarray],
              moist_f: Union[int, float, np.ndarray],
              moist_s: Union[int, float, np.ndarray],
              precip_ow: Union[int, float, np.ndarray],
              temp: Union[int, float, np.ndarray],
              month: Union[int, str]) -> Union[float, np.ndarray]:
    """
    Function to calculate the DC startup values after overwintering.\n
    This function implements new procedures outlined in Hanes and Wotton (2024).
    :param dc_f: DC value of the last day of FWI System calculation from the previous fall (unitless code)
    :param moist_f: moisture  of the last day of FWI System calculation from the previous fall ()
    :param moist_s: moisture value for the first day of FWI System calculation from the current spring ()
    :param precip_ow: total precipitation between the DC overwinter and startup date (mm)
    :param temp: today's temperature value (C)
    :param month: the current month (e.g., 9, '09', 'September')
    :return: startup DC value (unitless code)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, np.ndarray) for data in [dc_f, moist_f, moist_s, precip_ow]):
        return_array = True
    else:
        return_array = False

    # ### CONVERT ALL INPUTS TO MASKED NUMPY ARRAYS
    # Verify dc_f
    if not isinstance(dc_f, (int, float, np.ndarray)):
        raise TypeError('dc_f must be either int, float or numpy ndarray data types')
    elif isinstance(dc_f, np.ndarray):
        dc_f = np.ma.array(dc_f, mask=np.isnan(dc_f))
    else:
        dc_f = np.ma.array([dc_f], mask=np.isnan([dc_f]))

    # Verify moist_f
    if not isinstance(moist_f, (int, float, np.ndarray)):
        raise TypeError('moist_f must be either int, float or numpy ndarray data types')
    elif isinstance(moist_f, np.ndarray):
        moist_f = np.ma.array(moist_f, mask=np.isnan(moist_f))
    else:
        moist_f = np.ma.array([moist_f], mask=np.isnan([moist_f]))

    # Verify moist_s
    if not isinstance(moist_s, (int, float, np.ndarray)):
        raise TypeError('moist_s must be either int, float or numpy ndarray data types')
    elif isinstance(moist_s, np.ndarray):
        moist_s = np.ma.array(moist_s, mask=np.isnan(moist_s))
    else:
        moist_s = np.ma.array([moist_s], mask=np.isnan([moist_s]))

    # Verify precip_ow
    if not isinstance(precip_ow, (int, float, np.ndarray)):
        raise TypeError('p_ow must be either int, float or numpy ndarray data types')
    elif isinstance(precip_ow, np.ndarray):
        precip_ow = np.ma.array(precip_ow, mask=np.isnan(precip_ow))
    else:
        precip_ow = np.ma.array([precip_ow], mask=np.isnan([precip_ow]))

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
    b = np.ma.where(moist_s < moist_f,
                    0,
                    (moist_s - moist_f) / moist_f)

    # Final fall moisture equivalent
    q_f = 800 * np.exp(-dc_f / 400)

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
