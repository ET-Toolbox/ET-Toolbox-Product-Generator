#!/usr/bin/python

"""
Created on August 1 2019
@author: Bruno Aragon (bruno.aragonsolorio@kaust.edu.sa)

DESCRIPTION
===========
This package contains the functions for computing variables related to 
the MOST theory.
PACKAGE CONTENTS
===========
* compute_psi_m: Estimates the integrated stability correction term for wind.
* compute_psi_h(Zt, Z_0H, L): Function to compute the integrated stability correction.
* compute_rm: Function to estimate the momentum transfer resistance.
* compute_tau: Function to compute the surface stress.
* compute_tstar: Function to estimate the frictional temperature.
* compute_L: Function to estimate the Monin-Obukhov Length.
* compute_ustar: Function to calculate the friction velocity.
"""

def compute_psi_m(Zu, Z_0M, L):
    '''
    Estimates the integrated stability correction term for wind.
    Inputs:
        - float Zu: the wind speed measurement height in m.
        - float Z_0M: the aerodynamic roughness length for momentum
          transport in m.
        - float L: the obukhov stability length in m.
    Outputs:
        - float Psi_M: the integrated stability correction term for 
          wind (unitless).
    References
    ----------
    Högström, U., Review of some basic characteristics of the atmospheric surface 
        layer (1996), Bound.-Layer Meteor., 78, 215–246.
        http://dx.doi.org/10.1007%2FBF00120937
    Paulson, C. A., The mathematical representation of wind speed and temperature profiles 
        in the unstable atmospheric surface layer (1970), J. Appl. Meteor, 9, 857–861.
        http://dx.doi.org/10.1175/1520-0450(1970)009<0857:TMROWS>2.0.CO;2
    '''

    import numpy as np

    if type(L) == np.ndarray:
        Psi_M = np.ones(L.shape)
        xi = Zu/L # Stability parameter
        # Correction term for wind under stable conditions
        Psi_M[xi >= 0] = -5.3*(Zu-Z_0M[xi >= 0])/L[xi >= 0]
        # Correction term for wind under unstable conditions. Only assigne values where xi < 0 to avoid taking negative roots
        x = (1 - 19*Zu/L[xi < 0]) ** 0.25
        x_0 = (1 - 19*Z_0M[xi < 0]/L[xi < 0]) ** 0.25
        Psi_M[xi < 0] = 2*np.log((1 + x)/(1 + x_0)) + np.log((1 + x**2)/(1 + x_0**2)) - 2*np.arctan(x) + 2*np.arctan(x_0)
        
    else:
        xi = Zu/L
        # Correction term for wind under unstable conditions
        if xi < 0:
            x = (1 - 19*Zu/L) ** 0.25
            x_0 = (1 - 19*Z_0M/L) ** 0.25
            Psi_M = 2*np.log((1 + x)/(1 + x_0)) + np.log((1 + x**2)/(1 + x_0**2)) - 2*np.arctan(x) + 2*np.arctan(x_0)
        # Correction term for wind under stable conditions
        else:
            Psi_M = -5.3*(Zu-Z_0M)/L

    return Psi_M

def compute_psi_h(Zt, Z_0H, L):
    '''
    Function to compute the integrated stability correction 
    term for temperature.
    Inputs:
        - float Zt: the temperature measurement height in m.
        - float Z_0H: the aerodynamic roughness lenght for heat 
          transport in m.
        - float L: the obukhov stability length in m.
    Outputs:
        - float Psi_H: the integrated stability correction term 
          for temperature (unitless).
    References
    ----------
    Högström, U., Review of some basic characteristics of the atmospheric surface 
        layer (1996), Bound.-Layer Meteor., 78, 215–246.
        http://dx.doi.org/10.1007%2FBF00120937
    Paulson, C. A., The mathematical representation of wind speed and temperature profiles 
        in the unstable atmospheric surface layer (1970), J. Appl. Meteor, 9, 857–861.
        http://dx.doi.org/10.1175/1520-0450(1970)009<0857:TMROWS>2.0.CO;2
    '''
    import numpy as np

    if type(L) == np.ndarray:
        Psi_H = np.ones(L.shape)
        xi = Zt/L # Stability parameter
        # Correction term for temperature under stable conditions
        Psi_H[xi >= 0] = -8.0*(Zt - Z_0H[xi >= 0])/L[xi >= 0]
        # Correction term for temperature under unstable conditions
        y = (1 - 11.6*Zt/L[xi < 0]) ** 0.5
        y_0 = (1 - 11.6*Z_0H[xi < 0]/L[xi < 0]) ** 0.5
        Psi_H[xi < 0] = 2*np.log((1 + y)/(1 + y_0))

    else:
        xi = Zt/L # Stability parameter
        # Correction term for temperature under stable conditions
        if xi < 0:
            # Correction term for temperature under unstable conditions
            y = (1 - 11.6*Zt/L) ** 0.5
            y_0 = (1 - 11.6*Z_0H/L) ** 0.5
            Psi_H = 2*np.log((1 + y)/(1 + y_0))
        else:
            Psi_H = -8.0*(Zt - Z_0H)/L

    return Psi_H

def compute_rm(Zu, Z_0M, Psi_M, U):
    '''
    Function to estimate the momentum transfer resistance.
    Inputs:
        - float Zu: the wind speed measurement height in m.
        - float Z_0M: the aerodynamic roughness length for momentum
          transport in m.
        - float Psi_M: the integrated stability correction term for 
          wind (unitless).
        - float U: Wind speed above the canopy m/s.
    Outputs:
        - float Rm: the momentum transfer resistance s/m.
    References
    ----------        
    Yang, K., Koike, T., Ishikawa, Turbulent flux transfer over bare-soil 
        surfaces: Characteristics and parameterization (2008), Journal of 
        Applied Meteorology and Climatology, Volume 47, Pages 276-290,
        http://dx.doi.org/10.1175/2007JAMC1547.1.
    ''' 
    import numpy as np

    k = 0.4 # the von Karman constant
    Rm = ((np.log(Zu/Z_0M) - Psi_M) ** 2)/(U * (k**2))

    return Rm

def compute_tau(Rho, U, Rm):
    '''
    Function to compute the surface stress.
    Inptus:
        - float Rho: the density of air in kg/m3.
        - float U: Wind speed above the canopy m/s.
        - float Rm: the momentum transfer resistance s/m.
    Outputs:
        - float Tau: the surface stress in kg/(m s^2).  
    '''

    Tau = Rho*(U/Rm)

    return Tau

def compute_tstar(H, Rho, Cp, Ustar):
    '''
    Function to estimate the frictional temperature.
    Intpus:
        - float H: the sensible heat flux in W/m2
        - float Rho: the density of air in kg/m3.
        - float Cp: the heat capacity of moist air in J/(kg*K).     
        - float Ustar: the friction velocity in m/s.
    Outputs:
        - float Tstar: the frictional temperature in K.
    References
    ----------        
    Yang, K., Koike, T., Ishikawa, Turbulent flux transfer over bare-soil 
        surfaces: Characteristics and parameterization (2008), Journal of 
        Applied Meteorology and Climatology, Volume 47, Pages 276-290,
        http://dx.doi.org/10.1175/2007JAMC1547.1.
    '''

    Tstar = -H/(Rho*Cp*Ustar)

    return Tstar

def compute_L(Ta, Ustar, Tstar):
    '''
    Function to estimate the Monin-Obukhov Length.
    Inputs:
        - float Ta: the air temperature at reference height in K.
        - float Ustar: the friction velocity in m/s.
        - float Tstar: the frictional temperature in K.
    Outputs:
        - float L: the obukhov stability length in m.
    References
    ----------        
    Yang, K., Koike, T., Ishikawa, Turbulent flux transfer over bare-soil 
        surfaces: Characteristics and parameterization (2008), Journal of 
        Applied Meteorology and Climatology, Volume 47, Pages 276-290,
        http://dx.doi.org/10.1175/2007JAMC1547.1.
    '''

    import numpy as np

    k = 0.4 # the Von Karman constant
    g = 9.81 # the acceleration due to gravity

    L = Ta*(Ustar ** 2)/(k*g*Tstar)

    return L

def compute_ustar(Tau, Rho):
    '''
    Function to calculate the friction velocity.
    Inptus:
        - float Tau: the surface stress in kg/(m s^2).
        - float Rho: the density of air in kg/m3.
    Output:
        - float Ustar: the friction velocity in m/s.
    References
    ----------
    J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293.
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    Ustar = (Tau/Rho) ** 0.5 # by definition

    return Ustar