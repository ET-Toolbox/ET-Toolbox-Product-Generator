#!/usr/bin/python
"""
Created on August 1 2019
@author: Bruno Aragon (bruno.aragonsolorio@kaust.edu.sa)

DESCRIPTION
===========
This package contains the main function to call the two source energy balance mdoel.
For bare soil conditions, a one source model would be better in general (not SEBS ;]).
PACKAGE CONTENTS
===========
* TSEB_PT_Series: Priestley-Taylor Two Source Model. Calculates TSEB energy 
  fluxes using a single observation of composite radiometric temperature using
  resistances in series.
"""

import warnings

import numpy as np

from evapotranspiration_pipeline.configuration.logger import Logger
from .MOST_Functions import compute_psi_m, compute_rm, compute_tau, \
    compute_ustar, compute_psi_h, compute_tstar, compute_L
from .Met_Functions import compute_rho, compute_Cp, compute_delta, \
    compute_lambda, compute_gamma
from .Model_Functions import estimate_std_meridian, sunset_sunrise, \
    compute_G, compute_Tc_Series
from .Roughness_Functions import compute_Z_0M, compute_D_0, compute_Z_0H


def calculate_vapor(LE_daily, daylight_hours):
    # convert length of day in hours to seconds
    daylight_seconds = daylight_hours * 3600.0

    # constant latent heat of vaporization for water: the number of joules of energy it takes to evaporate one kilogram
    LATENT_VAPORIZATION_JOULES_PER_KILOGRAM = 2450000.0

    # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
    ET = np.float32(np.clip(LE_daily * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0.0, None))

    return ET


def TSEB(Tr_K, Ta_K, U, LAI, Vza, Sdn, Ldn, P, DoY, Lat, Lon, Time_t, Albedo, Rn, Rnd, daylight_hours, Z_u=2.0, Z_t=2.0, KB=2.0, F_c=1.0, Alpha_PT=1.26,
         Leaf_width=0.2, F_g=1.0, EmisVeg=0.98, EmisGrd=0.93, Hc=0.3, Mask=1, Mask_Val=-9999, logger=None):
    """
    Priestley-Taylor Two Source Model. Calculates TSEB energy fluxes using
    a single observation of composite radiometric temperature using
    resistances in series.
    Inputs:      
        - float Tr_K: Radiometric temperature (Kelvin)
        - float Ta_K: Air temperature (Kelvin)
        - float U: Wind speed above the canopy (m/s)
        - float LAI: Effective Leaf Area Index (m2 m-2)
        - float Vza: Sensor View Zenith Angle (degrees)
        - floar Sdn: Downwelling shortwave radiation at the surface (W m-2)
        - floar Ldn: Downwelling longwave radiation at the surface (W m-2)
        - float P: Atmospheric pressure (Pa)
        - int DoY: the day of the year (1 to 366)
        - float Lat: the latitude (degrees)
        - float Lon: the longitude (degrees)
        - float Time_t: the time of the observation (decimal time)
        - float Albedo: The fraction of the incident radiation that is reflected
          by the mixed surface, default is None        
    Optional Inputs:
        - float Z_u: Height of measurement of windspeed (m)
        - float Z_t: Height of measurement of air temperature (m)    
        - float KB: kB parameter, default value 2.0 Garrat 1992 suggestion
        - float F_c: Fractional cover, default is 1.0
        - float Alpha_PT: Priestley Taylor coeffient for canopy potential
          transpiration, default is 1.36
        - float Leaf_width: average/effective leaf width (m)
        - float F_g: Fraction of vegetation that is green, default is 1.0
        - float EmisVeg: Leaf emissivity, default is 0.98
        - float EmisGrd: Soil emissivity, default is 0.93
        - float Rn: the net radiation (W/m^2), default is None, to be computed
          from Sdn and Ldn.
        - numpy array Mask: an array containing the coded invalid data (as
          a number like -9999) of the same size as the input image. 
        - number Mask_Val: the value that represents invalid data.
    Outputs:
        - 
    References
    ----------
    J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29,
        http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    Colaizzi, P.D., Kustas, W.P., Anderson, M.C., Agam, N., Tolk, J.A., Evett, S.R., Howell, 
        T.A., Gowda, P.H., O'Shaughnessy, S.A.
        Two-source energy balance model estimates of evapotranspiration using component and 
        composite surface temperatures (2012) Advances in Water Resources, 50, pp. 134-151.
        http://dx.doi.org/10.1016/j.advwatres.2012.06.004
    Idso SB, Jackson RD, Thermal Radiation from the Atmosphere (1969), J Geophys Res, 74 (23), 
        pp. 5397-5403. 
        http://dx.doi.org/10.1029/JC074i023p05397

    """
    if logger is None:
        logger = Logger()

    warnings.filterwarnings("ignore")

    # mimimun allowed friction velocity    
    u_friction_min_glob = 0.35
    # minimum z_0H
    Z_0H_min = 0.003
    Z_0M_min = 0.003
    # minimum D_0
    D_0_min = 0.004
    # Maximum number of interations
    ALPHA_DEC = 0.01
    ITERATIONS = int(
        Alpha_PT / ALPHA_DEC)  # this is based on how fine to decrease the alpha parameter on each iteration
    kB = 2.0  # Garrat 1992 suggestion.
    min_LAI = 1

    max_iterations = ITERATIONS
    u_friction_min = np.ones_like(Tr_K) * u_friction_min_glob

    Hc = np.ones_like(Tr_K) * Hc
    Hc[LAI <= min_LAI] = 0.05  # D_0_min*3/2.0

    # If F_g is not a numpy array, make it 
    try:
        test = F_g.shape
    except AttributeError:
        F_g = F_g * np.ones_like(Tr_K)

    # If F_c is not a numpy array, make it
    try:
        test = F_c.shape
    except AttributeError:
        F_c = F_c * np.ones_like(Tr_K)

    # Compute Aerodynamic surface roughness length for momentum transfer (m) and Zero-plane displacement height (m)
    Z_0M = np.maximum(Z_0M_min, compute_Z_0M(Hc))
    D_0 = np.maximum(D_0_min, compute_D_0(Hc))
    Z_0H = np.maximum(Z_0H_min, compute_Z_0H(Z_0M))  # Roughness length for heat transport

    # Compute Solar Zenith Angle (radians)
    Std_meridian = estimate_std_meridian(Lon)
    _, _, Sza, _ = sunset_sunrise(DoY, Lon, Lat, Time_t, Std_meridian)

    # ensure arrays, can be removed.
    P = P * np.ones_like(Tr_K)
    Ta_K = Ta_K * np.ones_like(Tr_K)
    U = U * np.ones_like(Tr_K)

    # Create the output variables
    [Ts, Tc, LEc, H_c, LEs, Hs, G, Rs, Rx, Ra,
     u_friction, L, n_iterations] = [np.zeros_like(Tr_K) for i in range(13)]

    # Initialize support variables
    [U_C, u_S, u_d_zm, H, LE, Tw] = [np.zeros_like(Tr_K) for i in range(6)]

    # Calculate the general parameters    
    rho = compute_rho(P, Ta_K)  # Air density
    cp = compute_Cp(P, Ta_K)  # Heat capacity of air

    f_theta = 1 - np.exp(-0.5 * LAI / np.cos(Vza))
    f_theta[f_theta > 0.9] = 0.9
    f_theta[f_theta < 0.05] = 0.05

    # Assume stable conditions to get U* and L
    t_friction = np.zeros_like(Tr_K)
    L[:] = np.inf
    psim = compute_psi_m(Z_u, Z_0M, L)
    rm = compute_rm(Z_u, Z_0M, psim, U)
    tau = compute_tau(rho, U, rm)
    u_friction = compute_ustar(tau, rho)
    u_friction = np.maximum(u_friction_min, u_friction)

    # Estimates for the Prietly Taylor coefficients
    s = compute_delta(Ta_K)
    Lambda = compute_lambda(Ta_K)
    gama = compute_gamma(P, Lambda, cp)
    s_gama = s / (s + gama)

    # Compute the net radiation according to: Estimating Evapotranspiration from an Improved 
    # Two-Source Energy Balance Model Using ASTER Satellite Imagery

    # if type(Rn) != np.ndarray:
    #     sb = 5.670373e-8
    #     EmisSurf = (f_theta * EmisVeg + (1 - f_theta) * EmisGrd)
    #     Rn = (1 - Albedo) * Sdn + Ldn - EmisSurf * sb * Tr_K ** 4

    # Compute the canopy net radiation according to: Estimating Evapotranspiration from an 
    # Improved Two-Source Energy Balance Model Using ASTER Satellite Imagery

    Rnc = Rn * (1 - np.exp(-0.45 * LAI / np.sqrt(2 * np.cos(Sza))))
    Rns = Rn - Rnc

    # G = 0.35*Rns
    G = compute_G(Rns, DoY, Lat, Lon, Time_t, Std_meridian)

    # Calculate wind speed at the soil surface and in the canopy assuming stable conditions
    a1 = 0.004
    b1 = 0.012
    C = 90

    U_C = U * (np.log((Hc - D_0) / Z_0M) / (np.log((Z_u - D_0) / Z_0M) - compute_psi_m(Z_u, Z_0M, L)))
    a = 0.28 * LAI ** (2 / 3.) * Hc ** (1 / 3.) * Leaf_width ** (-1 / 3.)
    u_Ss = U_C * np.exp(-1 * a * (1 - (0.2 / Hc)))
    u_d_zm = U_C * np.exp(-1 * a * (1 - ((D_0 + Z_0M) / Hc)))

    # Calculate resistances assuming stable conditions
    psih = compute_psi_h(Z_t, Z_0H, L)
    Ra = (np.log((Z_u - D_0) / Z_0M) - psim) * (np.log((Z_t - D_0) / Z_0M) - psih) / (0.16 * U)

    Rs = 1 / (a1 + (b1 * u_Ss))
    Rx = (C / LAI) * ((s / u_d_zm) ** 0.5)

    # Initialize loop for water stress correction
    alpha_PT_loop = (Alpha_PT) * np.ones_like(Tr_K)
    LEs[:] = -1
    i = LEs < 0

    # Perform first Ts and Tc estimates using the PT-Equation
    Tc = np.minimum(Tr_K, Ta_K) + ((Rn * Ra) / (rho * cp)) * (1 - Alpha_PT * F_g * s_gama)  # Nieto + Adjustment of Colazzi
    LEc = alpha_PT_loop * F_g * s * Rnc / (s + s_gama)
    # LEc[LEc < 0] = 0.0
    H_c = Rnc - LEc

    Tc = Ta_K.copy() + H_c * Ra / (rho * cp)
    Ts = (Tr_K - f_theta * Tc) / (1 - f_theta)

    # Same constraints as in DisALEXI
    Ts[f_theta >= 0.9] = Tr_K[f_theta >= 0.9]
    Ts[f_theta <= 0.1] = Tr_K[f_theta <= 0.1]
    Tc[f_theta >= 0.9] = Tr_K[f_theta >= 0.9]
    Tc[f_theta <= 0.1] = Tr_K[f_theta <= 0.1]

    # Estimate Hs where LEs is zero
    Hs_LEs0 = Rns - G

    previous_negative_pixels = None
    LE_previous = None

    # on subsequent loops, only the pixels where LEs < 0 will run
    for iteration in range(1, max_iterations + 1):
        logger.info(f"running TSEB iteration: {iteration}")
        # New Estimate of temperatures
        Tc[i], Ts[i], Tw[i] = compute_Tc_Series(Tr_K[i], Ta_K[i], Ra[i], Rx[i], Rs[i], f_theta[i], H_c[i], rho[i],
                                                cp[i], Hs_LEs0[i])

        # Same constraints as in DisALEXI
        Ts[f_theta >= 0.9] = Tr_K[f_theta >= 0.9]
        Ts[f_theta <= 0.1] = Tr_K[f_theta <= 0.1]
        Tc[f_theta >= 0.9] = Tr_K[f_theta >= 0.9]
        Tc[f_theta <= 0.1] = Tr_K[f_theta <= 0.1]
        Tw[f_theta <= 0.1] = Ta_K[f_theta <= 0.1]

        # Tw[i] = ((Ta_K[i]/Ra[i]) + (Ts[i]/Rs[i]) + (Tc[i]/Rx[i])) / ((1/Ra[i]) + (1/Rs[i]) + (1/Rx[i])) #CalcT_AC(Ta_K[i], Ra[i], Ts[i], Rs[i], Tc[i], Rx[i])#
        H_c[i] = rho[i] * cp[i] * (Tc[i] - Tw[i]) / Rx[i]
        Hs[i] = rho[i] * cp[i] * (Ts[i] - Tw[i]) / Rs[i]

        # Estimate latent heat fluxes as residual of energy balance at the soil and the canopy
        LEs[i] = Rns[i] - G[i] - Hs[i]
        LEc[i] = Rnc[i] - H_c[i]

        # Calculate total fluxes
        H[i] = H_c[i] + Hs[i]
        LE[i] = LEc[i] + LEs[i]

        H[H == 0] = 10  # same as in disalexi IDL...
        Ra[Ra == 0] = 10

        # Now Land and friction velocity can be recalculated
        t_friction[i] = compute_tstar(H[i], rho[i], cp[i], u_friction[i])
        L[i] = compute_L(Ta_K[i], u_friction[i], t_friction[i])
        psim[i] = compute_psi_m(Z_u, Z_0M[i], L[i])
        rm[i] = compute_rm(Z_u, Z_0M[i], psim[i], U[i])
        tau[i] = compute_tau(rho[i], U[i], rm[i])

        u_friction[i] = compute_ustar(tau[i], rho[i])
        # Avoid very low friction velocity values
        u_friction = np.maximum(u_friction_min, u_friction)

        # Calculate wind speed at the soil surface and in the canopy
        U_C[i] = U[i] * (np.log((Hc[i] - D_0[i]) / Z_0M[i]) / (np.log((Z_u - D_0[i]) / Z_0M[i]) - psim[i]))
        u_S[i] = U_C[i] * np.exp(-1 * a[i] * (1 - (0.01 / Hc[i])))

        Ts_m_Tc = Ts[i] - Tc[i]
        abs_Temp = np.abs(Ts_m_Tc) < 1
        Rs[i][abs_Temp] = 1.0 / (0.004 + (0.012 * u_S[i][abs_Temp]))
        Rs[i][~abs_Temp] = 1 / (0.0025 * (Ts[i][~abs_Temp] - Tc[i][~abs_Temp]) ** (1 / 3.) + (b1 * u_S[i][~abs_Temp]))
        u_d_zm[i] = U_C[i] * np.exp(-1 * a[i] * (1 - ((D_0[i] + Z_0M[i]) / Hc[i])))
        Rx[i] = (C / LAI[i]) * ((Leaf_width / u_d_zm[i]) ** 0.5)
        psih[i] = compute_psi_h(Z_t, Z_0H[i], L[i])
        Ra[i] = (np.log((Z_u - D_0[i]) / Z_0M[i]) - psim[i]) * (np.log((Z_t - D_0[i]) / Z_0M[i]) - psih[i]) / (0.16 * U[i])

        alpha_PT_loop[i] -= ALPHA_DEC

        # Check that Alpha_PT doesn't reach negative values        
        alpha_PT_loop[alpha_PT_loop <= 0.0] = 0.0

        # If LEs is still less than zero when Alpha_PT <= 0.0, set it to zero and force closure  i = np.logical_and(LEs < 0, ~np.isnan(LEs))
        i = LEs < 0

        # Recompute using the PT equation as there is negative LEs
        LEc[i] = alpha_PT_loop[i] * F_g[i] * s[i] * Rnc[i] / (s[i] + s_gama[i])
        # LEc[LEc < 0] = 0.0
        H_c[i] = Rnc[i] - LEc[i]

        negative_pixel_count = np.nansum(i)

        logger.info(f"iteration {iteration} found {negative_pixel_count} negative pixels ({(negative_pixel_count / LEs.size * 100.0):0.2f}%)")

        if previous_negative_pixels is None:
            previous_negative_pixels = i
        else:
            if np.all(previous_negative_pixels == i):
                logger.info("no change in negative LEs, breaking")
                break

            previous_negative_pixels = i

        # If no negative LEs remains, exit the loop as it was successful
        if np.all(i == False):
            logger.info(f"all was good at iteration: {iteration}")
            break

        neg_LE_S = LEs < 0
        if np.any(neg_LE_S == True):
            LEs[neg_LE_S] = 0.0
            Hs[neg_LE_S] = Rns[neg_LE_S] - G[neg_LE_S]  # - LEs[neg_LE_S]

            Ts[neg_LE_S] = (Hs[neg_LE_S] * (Rs[neg_LE_S] + Ra[neg_LE_S]) / (rho[neg_LE_S] * cp[neg_LE_S])) + Ta_K[neg_LE_S]
            Tc[neg_LE_S] = (((Tr_K[neg_LE_S] ** 4) - (1 - f_theta[neg_LE_S]) * (Ts[neg_LE_S] ** 4)) / f_theta[neg_LE_S]) ** 0.25

            H_c[neg_LE_S] = rho[neg_LE_S] * cp[neg_LE_S] * (Tc[neg_LE_S] - Ta_K[neg_LE_S]) / Ra[neg_LE_S]
            LEc[neg_LE_S] = Rnc[neg_LE_S] - H_c[neg_LE_S]

            # Calculate total fluxes
            H[neg_LE_S] = H_c[neg_LE_S] + Hs[neg_LE_S]
            LE[neg_LE_S] = LEc[neg_LE_S] + LEs[neg_LE_S]

            hc_gt_rnc = H_c > Rnc
            if np.any(hc_gt_rnc == True):
                LEc[hc_gt_rnc] = 0
                H_c[hc_gt_rnc] = Rnc[hc_gt_rnc]
                Tc[hc_gt_rnc] = H_c[hc_gt_rnc] * Ra[hc_gt_rnc] / (rho[hc_gt_rnc] * cp[hc_gt_rnc]) + Ta_K[hc_gt_rnc]
                Ts[hc_gt_rnc] = (((Tr_K[hc_gt_rnc] ** 4) - f_theta[hc_gt_rnc] * (Tc[hc_gt_rnc] ** 4)) / (1 - f_theta[hc_gt_rnc])) ** 0.25
                Hs[hc_gt_rnc] = rho[hc_gt_rnc] * cp[hc_gt_rnc] * (Ts[hc_gt_rnc] - Ta_K[hc_gt_rnc]) / (Rs[hc_gt_rnc] + Ra[hc_gt_rnc])
                G[hc_gt_rnc] = Rns[hc_gt_rnc] - Hs[hc_gt_rnc]

                # Calculate total fluxes
                H[hc_gt_rnc] = H_c[hc_gt_rnc] + Hs[hc_gt_rnc]
                LE[hc_gt_rnc] = LEc[hc_gt_rnc] + LEs[hc_gt_rnc]

        if LE_previous is None:
            LE_previous = LE
            LE_diff = None
        else:
            LE_diff = LE - LE_previous
            LE_previous = LE

        if LE_diff is not None:
            logger.info(f"mean change in LE: {np.nanmean(LE_diff)} watts per square meter")

    # floor latent heat flux
    LEs = np.clip(LEs, 0, None)
    LEc = np.clip(LEc, 0, None)
    LE = np.clip(LE, 0, None)

    # calculate daily ET in mm per day or kg per square meter per day
    EF = LE / (Rn - G)
    EF = np.where((Rn - G) < 0.0001, 1.0, EF)
    EF = np.float32(np.clip(EF, 0.0, 1.0))
    LE_daily = EF * Rnd
    LE_daily = np.clip(LE_daily, 0.0, None)
    ET = calculate_vapor(LE_daily, daylight_hours)

    results = {'LE': LE, 'ET': ET, 'H': H, 'G': G, 'LEc': LEc, 'LEs': LEs, 'Hs': Hs, 'Hc': H_c, 'Rs': Rs, 'Rx': Rx, 'Ra': Ra, 'Ts': Ts, 'Tc': Tc, 'Tw': Tw}

    return results
