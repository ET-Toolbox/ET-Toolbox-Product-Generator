# ET Toolbox High-Resolution Evapotranspiration 7-Day Hindcast & 7-Day Forecast

This repository contains the code for the ET Toolbox 7-day hindcast and 7-day forecast data production system.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Copyright

Copyright 2022, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Requirements

This system was designed to work in a Linux-like environment and macOS using a conda environment.

### Amazon Linux 2 AMI

```bash

sudo yum update

sudo yum install git docker

sudo systemctl start docker

sudo systemctl enable docker

wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

bash Mambaforge-Linux-x86_64.sh

mamba init bash 
``` 

### `conda`

The ECOSTRESS Collection 2 PGEs are designed to run in a Python 3 [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) To use this environment, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Make sure that your shell has been initialized for `conda`.

You should see the base environment name `(base)` when running a shell with conda active.

## Installation

Use `make install` to produce the `ETtoolbox` environment:

```bash
(base) $ make install
```

This should produce a conda environment called `ETtoolbox` in your [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation.

## Activation

To use the pipeline, you must activate the `ETtoolbox` environment:

```bash
(base) $ conda activate ETtoolbox
```

You should see the environment name `(ETtoolbox)` in parentheses prepended to the command line prompt.

## Credentials

This system requires credentials from the [EROS Registration System](https://ers.cr.usgs.gov/register) and [Spacetrack](https://www.space-track.org/auth/createAccount). To check or store credentials for the servers accessed by this system, use this command:

```bash
(ETtoolbox) $ ET-Toolbox-Credentials
```

## Tile Runs

This system organizes raster processing by Sentinel tiles. To run the 7-day hindcast/forecast system on a single Sentinel tile, run the `ET-Toolbox-Tile` command with the namne of the tile and optional directory parameters:

```bash
(ETtoolbox) $ ET-Toolbox-Tile 13SDA --working working_directory --static static_directory --SRTM SRTM_directory --LANCE LANCE_directory --GEOS5FP GEOS5FP_directory
```

## Rio Grande Operation

To run all of the tiles covering the Rio Grande river in New Mexico, run the `ET-Toolbox-Rio-Grande` command:

```bash
(ETtoolbox) $ ET-Toolbox-Rio-Grande --working working_directory --static static_directory --SRTM SRTM_directory --LANCE LANCE_directory --GEOS5FP GEOS5FP_directory
```

![Map of Rio Grande Sentinel Tiles](./Rio%20Grande%20Sentinel%20Tiles.png)

## Deactivation

When you are done using the pipeline, you can deactivate the `ETtoolbox` environment:

```bash
(STARS) $ conda deactivate ETtoolbox
```

You should see the environment name on the command line prompt change to `(base)`.

## Updating

To update your installation of the `ETtoolbox` environment, rebuild with this command:

```bash
(base) $ make reinstall-hard
```

## Uninstallation

```bash
(base) $ make remove
```

