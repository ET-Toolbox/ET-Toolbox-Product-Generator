# ET Toolbox High-Resolution Evapotranspiration 7-Day Hindcast & 7-Day Forecast

This repository contains the code for the ET Toolbox 7-day hindcast and 7-day forecast data production system. This repository contains both code to deploy the container as
well as the scientific code to run within the deployed container. Changes to either the container configuration or the scientific code should be redeployed to update
the production products.

Caution should be given to any modifications of either the container or scientific code. The scientific code was provided by an external contractor and is not robust against
all potential conditions. Modification of the scientific code is likely to result in unexpected outcomes. If modifications are necessary, an A/B test should be done to 
confirm that there are no changes to the scientific products. The intent is to improve the code provided by the contractor as it becomes necessary to modify sections of the 
code. While this accepts the current state of the code, it minimizes the effort needed to bring the system into production.

## Host Configuration 
The application is fully containerized and setup to run as a RedHat Enterprise Linux (RHEL) 9 container. As Reclamation does not currently provide a container hosting 
platform like Kubernetes, the container is typically ran within a host machine. It does not matter to the container itself if it runs in a containerized environment or on a 
host machine in terms of the internal workflow. However, additional setup is required on a host machine compared to a container hosting environment.

The container has been tested running in Podman on a RHEL 8 host machine. It is possible to use other host/container platforms with the current template, but testing would 
be necessary to ensure host configuration and container deployment is successful. The Reclamation security environment requires that users be root when working with 
containers on a host machine.

### Dependencies
1. Ensure the host machine is up-to-date with all required packages
2. Install podman:

   ```bash
   sudo dnf install podman
   ```
   
3. Podman installs with a set of defaults, some of which may need to be changed based on the configuration of the host machine. 
    * Podman typically has a folder in the /var/temp space for containers that can get full quickly, particularly once scientific data starts to come into the container. The 
      best way to deal with this is to change the location of the container storage folder. In /etc/containers/storage.conf, change graphroot variable to the desired storage 
      location, such as: /mnt/scratch-medium/containers.
   * If you change the default container location above, it will confuse the SELinux configuration and prevent podman from building containers correctly. You will need to 
     update the SELinux configuration with the following commands:
   
   ```bash
   semanage fcontext -a -e /var/lib/containers/storage /NEWSTORAGEPATH
   restorecon -R -v /NEWSTORAGEPATH
   ```

     The new storage path in the above commands corresponds to the desired storage path from the first commands. This will result in a change to where the containers are 
     permanently stored on the host machine.
   * The location of the container build is different from that of permanent container location. When building a large container, it is therefore possible to run out of space
     during the build before the container completes the build. The best way to manage this is to repoint the temporary build space to the same folder that was used above. 
     This gives Podman enough space to build the container and then commits it into the same space. Every time a new shell session is created, the following environment 
     variable should be set:

     ```bash
     export TMPDIR=/mnt/scratch-medium/containers
     ```
   
4. Configure the network mount location on the host. Start by making the mount location in the host:

    `mkdir /mnt/jpl`
   
    Then the /etc/fstab file needs to be updated with the following line. The first portion of the line is the host machine for the network location:

    `IBR8DROFP001.bor.doi.net:/JPL /mnt/jpl          nfs defaults 0 0`

    Then remount the drives in fstab file using the following line:

    `mount -o remount -a`

    It is good practice to write a test file into the folder to confirm that it is setup correct.

### Building the container
The container deployment follows the standard Podman process. Ensure that the final step from the dependencies is done to change the temporary storage location.

1. Clone the repository to the host machine.
2. Update the .credentials and certificates following the proceedures in the subsequent sections.
3. Build the image.

   `podman build --format=docker --layers=false -t ettoolbox -f Dockerfile`

4. Create a container from the image. This requires the that network share be mounted on the container host at /mnt/jpl. This is the location that the front end will
   attempt to find the raster data.

   `podman run -dit --name etcontainer -v /mnt/jpl/:/mnt/export/ ettoolbox`

5. Start and enter the container.

   `podman exec -it etcontainer /bin/bash`


## Data Feed Registrations
The system requires authentication from various data feeds to both configure the container build and the data sources for the scientific calculations. These values should be
set in the .credentials file within the repository before completing a build. Periodically, tokens and passwords may need to refreshed within the .credentials file of the 
repository. To get the refreshed keys into the container, the user can either rebuild the container or also update the .credentials file within the production system. 

### GitLab
The GitLab token allows the system to access the Reclamation Gitlab server to pull in the necessary package dependencies. It is therefore utilized only during the initial
build of the container. The token is setup to expire on an annual basis, and should be refreshed at least every January 1st to ensure the token remains active. 
The username does not need to be changed. Instructions for creating a token are available here: https://docs.gitlab.com/user/project/settings/project_access_tokens/

### EROS 
The Earth Resources Observation and Science (EROS) system is a USGS system that provides automated data access to USGS data resources. It authentications both with 
username/password as well as a token. To create an account, go to this website: 

https://ers.cr.usgs.gov/register

The username/password will need to go into the .credentials file. These are static and do not need to change. The access request expires periodically and needs to get 
renewed. Next expiration is 02/04/2026. There also needs to be an API token issued to programmatically access the system. That comes from the machine-2-machine (M2M) system 
and needs a separate authorization from https://m2m.cr.usgs.gov/. Request access by going to:

https://ers.cr.usgs.gov/profile/access

Access type is “Access to EE’s Machine to Machine interface (MACHINE)”, and request access to the MODIS and GEOS-5 datasets. You can create an Application token that needs .
to be put into the .credentials file as the EROS_TOKEN. 

### Earthdata
Go to the following site and create a username/password combination:

https://urs.earthdata.nasa.gov/

Those should be entered into the .credentials file. Using the credentials information, the system will automatically request and managing tokens within the EarthData 
environment.

### Spacetrack
Create an account at the below link, and add the username/password into the credentials file.

https://www.space-track.org/auth/login

## Certificates
Certificates can be ... problematic ... in the Reclamation security environment. We use self-signed certificates that are not universally accepted by the data providers. 
This can result in ssl errors that keep the workflow from functioning correctly. Although ssl verification can be disabled, this is not considered best security practice. 
The approach used here instead is to patch Reclamation certificates into the container and workflow with the necessary connection adjustments to use our certificates.

The bulk of the changes are in the workflow itself and does not require user intervention. Broadly, these modifications involved placing the certificates in the proper 
locations in the folder, forcing python to use the system certificates, and using curl instead of wget. However, the user must ensure that the certificates within the 
repository are current for the build script to incorporate the correct information into the build process.

The certificates used in building the container are in the `certs` repository folder and consist of two files `ca-bundle.crt` and `ca-bundle.trust.crt`. Both files are
necessary for the system authenticate correctly. The most straightforward way to obtain the files is from the Linux host or another Linux machine. These are typically stored
in the `/etc/ssl/certs/` or `/etc/ssl/` folder on RHEL systems. These can be copied from the donor system and updated in the repo. Certificates refresh periodically in 
random intervals, so updating the files is pretty impossible on a known schedule. If the container begins to fail and throw SSL certificate issues everywhere, most likely
these files need to be updated. 

