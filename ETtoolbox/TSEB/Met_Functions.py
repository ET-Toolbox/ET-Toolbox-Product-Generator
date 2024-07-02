#!/usr/bin/python

"""
Created on August 1 2019
@author: Bruno Aragon (bruno.aragonsolorio@kaust.edu.sa)

DESCRIPTION
===========
This package contains the functions for computing variables related to
meteorology.
PACKAGE CONTENTS
===========
* compute_rho: Function to estimate the air density.
* compute_Cp: Function to estimate the heat capacity of moist air.
* compute_lambda: Function to estimate the latent heat of vaporaization. 
  Baed on Harrison (1963).
* compute_delta: Function to compute the slope of the relationship between 
  saturation vapour pressure and air temperature.
* compute_gamma: Function to compute the psychrometric constant.
* compute_delta: Function to compute the slope of the relationship between 
  saturation vapour pressure and air temperature.
"""

def compute_rho(P, Ta):
    '''
    Function to estimate the air density.
    Inputs:
        - float P: the air pressure in Pa.
        - float Ta: the air temperature at reference height in K.
    Outputs:
        - float Rho: the density of air in kg/m3.
    '''

    Rd = 287.058 #J/(kg·K) # gas constant for dry air

    Rho = P/(Ta*Rd)

    return Rho

def compute_Cp(P, Ta):
    '''
    Function to estimate the heat capacity of moist air.
    Inputs:
        - float P: the air pressure in Pa.
        - float Ta: the air temperature at reference height in K.
    Outputs:
        - float Cp: the heat capacity of moist air in J/(kg*K).
    References
    ----------
    Idso SB, Jackson RD, Thermal Radiation from the Atmosphere (1969), J Geophys Res, 74 (23), 
        pp. 5397-5403. 
        http://dx.doi.org/10.1029/JC074i023p05397
    '''

    from numpy import exp

    Cpd = 1003.5 # J/(kg*K) specific heat for dry air at constant pressure
    Cpv = 1870 # J/(kg/K) specific heat capacity of water vapor
    # Estimate Ea in mbar using Idso 1969 eq 10, then convert to Pa
    Ea = 6.11*exp(16.9*(1 - 273.0/Ta)) * 100
    E = 0.622 # ratio of molecular weight of water vapour to dry air
   
    # first calculate specific humidity, rearanged eq (5.22) from Maarten Ambaum (2010), (pp 100)
    Q = E * Ea / (P + (E - 1.0)*Ea) # the specific humidity
    Cp = (1.0-Q)*Cpd + Q*Cpv
    
    return Cp    

def compute_lambda(Ta):
    '''
    Function to estimate the latent heat of vaporaization. Baed on
    Harrison (1963). 
    Inputs:
        - float Ta: the air temperature at reference height in K. 
    Outputs:
        - float L: the latent heat of vaporization in MJ/kg
    See:
    http://www.fao.org/3/X0490E/x0490e0k.htm
    '''

    L = 2.501 - (2.361e-3*(Ta - 273.15)) 
    
    return L

def compute_delta(Ta):
    '''
    Function to compute the slope of the relationship between 
    saturation vapour pressure and air temperature.
    Inputs:
        - float Ta: the air temperature at reference height in K. 
    Outputs: 
        - float Delta: the slope of saturation vapour 
        pressure curve in kPa/K.
    References
    ----------
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. 
        Crop evapotranspiration —guidelines for computing crop water requirements
        (1998) FAO Irrigation and drainage paper 56. Food and Agriculture 
        Organization, Rome, pp. 37. 
        http://www.fao.org/docrep/x0490e/x0490e00.htm
    '''

    import numpy as np

    # FAO56 eq. 13
    Temp_C = Ta - 273.16
    Delta = 4098*0.6108*np.exp(17.27*Temp_C/(Temp_C + 237.3)) / ((Temp_C + 237.3) ** 2)

    return Delta

def compute_gamma(P, L, Cp):
    '''
    Function to compute the psychrometric constant.
    Inputs:
        - float P: the air pressure in Pa.
        - float L: the latent heat of vaporization in MJ/kg.
        - float Cp: the heat capacity of moist air in J/(kg*K).              
    Outputs: 
        - float Gamma: the psychrometric constant kPa/K. 
    References
    ----------
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. 
        Crop evapotranspiration —guidelines for computing crop water requirements
        (1998) FAO Irrigation and drainage paper 56. Food and Agriculture 
        Organization, Rome, pp. 32. 
        http://www.fao.org/docrep/x0490e/x0490e00.htm
    '''

    e = 0.622 # ratio of molecular weight of water vapour to dry air
    Pressure = P/1000

    Gamma = (Cp/1000)*(10 ** -3)*Pressure/(e*L)

    return Gamma

def compute_delta(Ta):
    '''
    Function to compute the slope of the relationship between 
    saturation vapour pressure and air temperature.
    Inputs:
        - float Ta: the air temperature at reference height in K.
    Outputs: 
        - float Delta: the slope of saturation vapour 
          pressure curve in KPa/K.
    References
    ----------
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. 
        Crop evapotranspiration —guidelines for computing crop water requirements
        (1998) FAO Irrigation and drainage paper 56. Food and Agriculture 
        Organization, Rome, pp. 37. 
        http://www.fao.org/docrep/x0490e/x0490e00.htm
    '''

    import numpy as np

    Temp_C = Ta - 273.16

    # FAO56 eq. 13
    Delta = 4098*0.6108*np.exp(17.27*Temp_C/(Temp_C + 237.3)) / ((Temp_C + 237.3) ** 2)

    return Delta