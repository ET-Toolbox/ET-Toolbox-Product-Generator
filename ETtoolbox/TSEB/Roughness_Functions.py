#!/usr/bin/python

"""
Created on August 1 2019
@author: Bruno Aragon (bruno.aragonsolorio@kaust.edu.sa)

DESCRIPTION
===========
This package contains the functions to compute roughness lenghts and the zero-plane
displacement height.
PACKAGE CONTENTS
===========
* compute_Z_0M: Function to estimate the aerodynamic roughness length for momentum
  transport.
* compute_D_0: Function to calculate the zero-plane displacement height from a 
  fixed ratio to canopy height.
* compute_Z_0H: Function to calculate the aerodynamic roughness length
  for heat transport.
"""

def compute_Z_0M(Hc):
    ''' 
    Function to estimate the aerodynamic roughness length for momentum
    transport.
    Inputs:
        - float Hc: the canopy height in m
    Outputs:
        - float Z_0M: the aerodynamic roughness length for momentum
          transport in m.
    '''

    Z_0M = Hc * 0.125

    return Z_0M

def compute_D_0(Hc):
    '''
    Function to calculate the zero-plane displacement height from a 
    fixed ratio to canopy height.
    Inputs:
        - float Hc: the canopy height in m.
    Outpus:
        - float D_0: the zero-plance displacement height m.
    '''
    
    D_0 = Hc * 0.65

    return D_0    

def compute_Z_0H(Z_0M, KB = 2):
    '''
    Function to calculate the aerodynamic roughness length
    for heat transport.
    Inputs:
        - float Z_0M: the aerodynamic roughness length for momentum
          transport in m.
    Optional_Inputs:
        - float KB: parameter.
    OutputS:
        - float Z_0H: the aerodynamic roughness lenght for heat 
          transport in m.
    References
    ----------
    J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    from numpy import exp

    Z_0H = Z_0M/exp(KB)

    return Z_0H