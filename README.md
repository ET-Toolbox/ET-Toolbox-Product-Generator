# ET Toolbox High-Resolution Evapotranspiration 7-Day Hindcast & 7-Day Forecast

This repository contains the code for the ET Toolbox 7-day hindcast and 7-day forecast data production system. This repository contains both code to deploy the container as well as the scientific code to run within the deployed container. Changes to either the container configuration or the scientific code should be redeployed to update the production products.

Caution should be given to any modifications of either the container or scientific code. The scientific code was provided by an external contractor and is not robust against all potential conditions. Modification of the scientific code is likely to result in unexpected outcomes. If modifications are necessary, an A/B test should be done to confirm that there are no changes to the scientific products. The intent is to improve the code provided by the contractor as it becomes necessary to modify sections of the code. While this accepts the current state of the code, it minimizes the effort needed to bring the system into production.

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

The ECOSTRESS Collection 2 PGEs are designed to run in a Python 3 [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To use this environment, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Make sure that your shell has been initialized for `conda`.

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

## Credentials Configuration (REQUIRED BEFORE BUILD)

**⚠️ CRITICAL: The `.credentials` file must be created and properly configured before building the Docker container. The build will fail without valid credentials.**

The system requires authentication credentials for various external data sources. These credentials are stored in a `.credentials` file that serves as a shell script to set environment variables.

### Creating the Credentials File

1. **Copy the template** to create your credentials file:
   ```bash
   cp .credentials.template .credentials
   ```

2. **Edit the `.credentials` file** and replace all placeholder values with your actual credentials:
   ```bash
   nano .credentials
   ```

3. **Secure the file** - Set appropriate permissions:
   ```bash
   chmod 600 .credentials
   ```

### Required Credentials

#### USGS EROS Credentials
- `EROS_USERNAME` - USGS EROS login username
- `EROS_PASSWORD` - USGS EROS login password  
- `EROS_TOKEN` - Machine-to-Machine (M2M) API token
- **Purpose**: Access USGS satellite data and Landsat imagery
- **Setup**: Register at https://ers.cr.usgs.gov/register, then request M2M access at https://m2m.cr.usgs.gov/

#### NASA Earthdata Credentials
- `EARTHDATA_USERNAME` - NASA Earthdata username
- `EARTHDATA_PASSWORD` - NASA Earthdata password
- **Purpose**: Access NASA Earth science datasets (MODIS, VIIRS, etc.)
- **Setup**: Register at https://urs.earthdata.nasa.gov/

#### Spacetrack Credentials
- `SPACETRACK_USERNAME` - Space-Track.org username
- `SPACETRACK_PASSWORD` - Space-Track.org password
- **Purpose**: Access satellite orbital data for VIIRS orbit calculations
- **Setup**: Register at https://www.space-track.org/auth/login

### Credential Maintenance

- **Periodic Updates**: Some credentials expire and need refresh (e.g., GitLab tokens annually, EROS access periodically)
- **Container Updates**: When credentials expire, either:
  1. Rebuild the container with updated `.credentials` file, OR  
  2. Update both the repository `.credentials` file AND `/root/.credentials` in the production container
- **Monitoring**: If authentication errors occur, check credential expiration first

## Host Configuration

The application is fully containerized and set up to run as a RedHat Enterprise Linux (RHEL) 9 container. As Reclamation does not currently provide a container hosting platform like Kubernetes, the container is typically run within a host machine. It does not matter to the container itself if it runs in a containerized environment or on a host machine in terms of the internal workflow. However, additional setup is required on a host machine compared to a container hosting environment.

The container has been tested running in Podman on a RHEL 8 host machine. It is possible to use other host/container platforms with the current template, but testing would be necessary to ensure host configuration and container deployment is successful. The Reclamation security environment requires that users be root when working with containers on a host machine.

### Dependencies

1. Ensure the host machine is up-to-date with all required packages.
2. Install Podman:

   ```bash
   sudo dnf install podman
   ```

3. Configure Podman storage and SELinux settings as needed (see alternate README for detailed instructions).

4. Configure the network mount location on the host. Start by making the mount location in the host:

   ```bash
   mkdir /mnt/jpl
   ```

   Then update the `/etc/fstab` file and remount the drives:

   ```bash
   mount -o remount -a
   ```

### Building the Container

1. Clone the repository to the host machine.
2. Configure credentials (see above).
3. Build the image:

   ```bash
   podman build --format=docker --layers=false -t ettoolbox -f Dockerfile
   ```

4. Create and start the container:

   ```bash
   podman run -dit --name etcontainer -v /mnt/jpl/:/mnt/export/ ettoolbox
   ```

5. Enter the container:

   ```bash
   podman exec -it etcontainer /bin/bash
   ```

## Tile Runs

This system organizes raster processing by Sentinel tiles. To run the 7-day hindcast/forecast system on a single Sentinel tile, run the `ET-Toolbox-Tile` command with the name of the tile and optional directory parameters:

```bash
(ETtoolbox) $ ET-Toolbox-Tile 13SDA --working working_directory --static static_directory --SRTM SRTM_directory --LANCE LANCE_directory --GEOS5FP GEOS5FP_directory
```

## Rio Grande Operation

To run all of the tiles covering the Rio Grande river in New Mexico, run the `ET-Toolbox-Rio-Grande` command:

```bash
(ETtoolbox) $ ET-Toolbox-Rio-Grande --working working_directory --static static_directory --SRTM SRTM_directory --LANCE LANCE_directory --GEOS5FP GEOS5FP_directory
```

![Map of Rio Grande Sentinel Tiles](./Rio%20Grande%20Sentinel%20Tiles.png)

## Certificates

Certificates can be problematic in the Reclamation security environment. We use self-signed certificates that are not universally accepted by the data providers. This can result in SSL errors that keep the workflow from functioning correctly. Although SSL verification can be disabled, this is not considered best security practice. The approach used here instead is to patch Reclamation certificates into the container and workflow with the necessary connection adjustments to use our certificates.

The certificates used in building the container are in the `certs` repository folder and consist of two files: `ca-bundle.crt` and `ca-bundle.trust.crt`. Ensure these files are updated periodically to avoid SSL issues.

## Deactivation

When you are done using the pipeline, you can deactivate the `ETtoolbox` environment:

```bash
(ETtoolbox) $ conda deactivate
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

