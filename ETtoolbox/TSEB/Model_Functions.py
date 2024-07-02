#!/usr/bin/python
"""
Created on August 1 2019
@author: Bruno Aragon (bruno.aragonsolorio@kaust.edu.sa)

DESCRIPTION
===========
This package contains supporting functions to run TSEB.
PACKAGE CONTENTS
===========
* estimate_std_meridian: Estimates the standard meridian.
* sunset_sunrise: Estimates sun rise, sunset, solar noon times and the solar azimuth angle.
* compute_G: Estimates Soil Heat Flux as function of time and net radiation.
* compute_Tc_Series: Function to estimate the soil, canopy and air canopy temperatures
  from the canopy sensible heat flux and resistance network in series.
"""
def estimate_std_meridian(Lon):
    '''
    Estimates the standard meridian.
    Inputs:       
        - float Lon: the longitude (degrees from -180 to 180)
    Outputs:
        - int Std_Meridian: the standard meridian of the time zone (degrees)
    References
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.    
    '''

    import numpy as np

    Std_Meridian = np.round(Lon / 15) * 15

    return Std_Meridian

def sunset_sunrise(DoY, Lon, Lat, Time_t, Std_meridian):
    '''
    Estimates sun rise, sunset, solar noon times and the solar azimuth angle.
    Inputs:
        - int DoY: the day of the year (1 to 366)        
        - float Lon: the longitude (degrees)
        - float Lat: the latitude (degrees)
        - float Time_t: the time of the observation (decimal time)
        - int Std_meridian: the standard meridian of the time zone (degrees)
    Outputs:
        - float t_rise: biological time of sunrise (decimal time)
        - float t_end: biological time of sunset (decimal time)
        - float zs: azimuth angle of the sun (radians)
        - float t_noon: time of solar noon (decimal time)
    References
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    # Note that the equations are in degrees so conversion is needed.

    from numpy import sin, cos, pi, deg2rad, arcsin, arccos, rad2deg

    dtor = deg2rad(1)
    radeg = rad2deg(1)

    # Estimate the solar declination in radians eq (11.2)
    #; note that 0.39785 = sin(23.44 degrees)
    solar_declination = arcsin(sin(23.45*dtor)*sin(dtor*(278.97 + 0.9856*DoY + 1.9165*sin((356.6 + 0.9856*DoY)*dtor))))

    # Compute the value of the equation of time in hours eq (11.4)
    f = (279.575 + 0.9856*DoY)*dtor # convert f to radians
    equation_time = (-104.7*sin(f) + 596.2*sin(2*f) + 4.3*sin(3*f) - 12.7*sin(4*f) - 429.3*cos(f) - 2.0*cos(2*f) + 19.3*cos(3*f))/3600.0

    # Compute the time of solar noon eq (11.3)
    LC = (4*(Lon - Std_meridian))/60.0 # compute in hours
    t_noon = 12 - LC - equation_time

    # Compute the zenith angle of the sun eq (11.1) time_t must be in standard time (local time disregarding daylight savings adjustment)
    cos_zs = sin(Lat*dtor)*sin(solar_declination) +cos(Lat*dtor)*cos(solar_declination)*cos((15*(Time_t - t_noon))*dtor)

    # Solar angle is in radians.
    zs = arccos(cos_zs)

    # Compute the halfday length considering twilight (set zs = 96 degrees) eq (11.6)
    halfday = arccos((cos(96*dtor)-sin(Lat*dtor)*sin(solar_declination))/(cos(Lat*dtor)*cos(solar_declination)))
    halfday_h = halfday*radeg/15.0 # converting to hours

    # Compute sunrise and sunset time eq (11.7)
    t_rise = t_noon - halfday_h
    t_end = t_noon + halfday_h

    return t_rise, t_end, zs, t_noon

def compute_G(Rn, DoY, Lat, Lon, Time_t, Std_meridian, G_param=[0.31, 3.0, 24.0]):
    ''' 
    Estimates Soil Heat Flux as function of time and net radiation.
    Inputs:
        - float Rn: Net radiation (W m-2)
        - int DoY: the day of the year (1 to 366)
        - float Lat: the latitude (degrees)
        - float Lon: the longitude (degrees)
        - float Time_t: the local time of the observation (decimal time)
        - int Std_meridian: the standard meridian of the time zone (degrees)
        - list [float, float, float] G_Param: parameters required
            - Amplitude: maximum value of G/Rn, amplitude, default=0.31
            - Phase_Shift: shift of peak G relative to solar noon (default 3hrs after noon)
            - Shape: shape of G/Rn, default 24 hrs
    Outputs:
        - float G: Soil heat flux (W m-2)
    References
    ----------
    Santanello2003 Joseph A. Santanello Jr. and Mark A. Friedl, 2003: Diurnal Covariation in
        Soil Heat Flux and Net Radiation. J. Appl. Meteor., 42, 851-862,
        http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2.
    Note: All parameters are in hours, Santanello's parameters are in seconds, convert by
    multiplying by 3600.
    '''
    
    from numpy import sin, cos, pi as sin, cos, pi

    # Compute the solar noon
    t_rise, t_end, zs, t_noon = sunset_sunrise(DoY, Lon, Lat, Time_t, Std_meridian)
    time_g = (Time_t - t_noon) * 3600

    A = G_param[0]
    phase_shift = G_param[1] * 3600
    B = G_param[2] * 3600
    G_ratio = A*cos(2.0*pi*(time_g+phase_shift)/B)

    G = Rn * G_ratio

    return G

def compute_Tc_Series(Tr,Ta, Ra, Rx, Rs, F_theta, H_C, Rho, Cp, Hs_LEs0):
    '''
    Function to estimate the soil, canopy and air canopy temperatures
    from the canopy sensible heat flux and resistance network in series.
    Inputs:
        - float Tr: Radiometric temperature in K.
        - float Ta: the air temperature at reference height in K.
        - float Ra: the aerodynamic resistance to heat transport in s/m.
        - float Rx: the aerodynamic resistance to heat transport 
          at the canopy boundary in s/m.
        - float Rs: the aerodynamic resistance to heat transport 
          at the soil boundary in s/m.
        - float F_theta: fraction of vegetation observed (unitless).
        - float H_C: the sensible heat flux in the canopy in W/m2.
        - float Rho: the density of air in kg/m3.
        - float Cp: the heat capacity of moist air in J/(kg*K).
    Outputs:
        - float Tc: the canopy temperature in K.
        - float Ts: the soil temperature in K.
        - float Tw: the air canopy temperature in K.
    References
    ----------
    J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
        Eqs. A5-A13
    '''

    import numpy as np
    
    n = 4
    Tr4=Tr**n
    # equation A7 from Norman 1995, linear approximation of temperature of the canopy
    Tc_lin = (( Ta/Ra + Tr/(Rs*(1.0-F_theta)) 
        + H_C*Rx/(Rho*Cp)*(1.0/Ra + 1.0/Rs + 1.0/Rx)) 
        /(1.0/Ra + 1.0/Rs + F_theta/(Rs*(1.0 - F_theta))))
    # equation A12 from Norman 1995
    T_D = (Tc_lin*(1+Rs/Ra) - H_C*Rx/(Rho*Cp)*(1.0 + Rs/Rx + Rs/Ra)
            - Ta*Rs/Ra)
    # equation A11 from Norman 1995
    delta_Tc = ((Tr4 - F_theta*Tc_lin**4.0 - (1.0-F_theta)*T_D**4) 
        / (4.0* (1.0-F_theta)* T_D**3.0* (1.0 + Rs/Ra) + 4.0*F_theta*Tc_lin**3))
    # get canopy temperature in Kelvin
    Tc = Tc_lin + delta_Tc

    # Succesfull inversion
    Ts = np.zeros_like(Tc)
    T_temp = Tr**n - F_theta*Tc**n
    Ts[T_temp>=0] = ( T_temp[T_temp>=0] / (1.0 - F_theta[T_temp>=0]))**0.25

    # Unsuccesfull inversion
    Ts[T_temp<0] = Tr[T_temp<0] /(1.0 - F_theta[T_temp<0]) #1e-6
    #Tw = (( Ta/Ra + Ts/Rs + Tc/Rx ) /(1.0/Ra + 1.0/Rs + 1.0/Rx))
    
    Tw_lin = ((Ta/Ra) + Tr/(F_theta*Rx) - ((1-F_theta)/(F_theta*Rx))*(Hs_LEs0*Rs/(Rho*Cp))+Hs_LEs0/(Rho*Cp))/(1/Ra + 1/Rx + (1-F_theta)/(F_theta*Rx))
    Te = Tw_lin*(1 + Rx/Ra) - (Hs_LEs0*Rx)/(Rho*Cp) - Ta*Rx/Ra
    Te4 = Te ** n
    Tw_delta = (Tr4 - (1-F_theta)*((Hs_LEs0*Rs/(Rho*Cp) + Tw_lin) ** n) - F_theta*Te4)/(n*F_theta*(Te ** (n -1))*(1 + Rx/Ra) + n*(1-F_theta)*((Hs_LEs0*Rs/(Rho*Cp) + Tw_lin) ** (n - 1)))

    Tw = Tw_lin + Tw_delta

    return Tc, Ts, Tw