FROM registry.access.redhat.com/ubi9/ubi-init

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Copy files from the host to the container
RUN mkdir /home/app
COPY ETtoolbox /home/app/ETtoolbox
COPY environment.yml /home/app
COPY run.sh /home/app
COPY pyproject.toml /home/app
COPY Rio_Grande_Sentinel_Tiles.png /home/app/
COPY ETtoolbox_riogrande.py /home/app
COPY postprocess.py /home/app
COPY .credentials /root/

RUN mkdir -p /mnt/export/et_rasters
COPY mrg_shapefile/ /home/app/mrg_shapefile

# Repermission the run script
RUN chmod 777 /home/app/run.sh

# Fix ssl encryption
RUN mkdir -p /etc/ssl/certs
COPY certs/ca-bundle.crt /etc/ssl/certs/
COPY certs/ca-bundle.trust.crt /etc/ssl/certs/
RUN update-ca-trust

# Set global SSL variables
RUN echo export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt >> /root/.bashrc
RUN echo export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt >> /root/.bashrc

RUN echo export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt >> /root/.bash_profile
RUN echo export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt >> /root/.bash_profile

# Fix SSL issues that with the University of Maryland server
RUN true | openssl s_client -connect gladweb1.umd.edu:443 2>/dev/null | openssl x509 >> /etc/ssl/certs/ca-bundle.crt
RUN true | openssl s_client -connect gladweb1.umd.edu:443 2>/dev/null | openssl x509 >> /etc/ssl/certs/ca-bundle.crt

# Update the container from the RedHat repos
RUN yum reinstall -y ca-certificates
RUN yum install -y cronie
RUN yum install -y wget
RUN yum install -y gcc
RUN yum install -y nano
RUN yum install -y git
RUN yum update -y

# Refix the ssl certificates
# Using ca-certificates is good, because it updates the system certs, but it moves the initial ca-bundle to .rpm save extensions, which does not follow the intended
# application. We need to move them back to the original locations.
RUN mv /etc/ssl/certs/ca-bundle.crt.rpmsave /etc/ssl/certs/ca-bundle.crt
RUN mv /etc/ssl/certs/ca-bundle.trust.crt.rpmsave /etc/ssl/certs/ca-bundle.trust.crt
RUN update-ca-trust

# Disable pam authentication
RUN sed -i 's/account    required   pam_access.so/#account    required   pam_access.so/g' /etc/pam.d/crond
RUN sed -i 's/session    required   pam_loginuid.so/#session    required   pam_loginuid.so/g' /etc/pam.d/crond

# Install conda
RUN wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-Linux-x86_64.sh --no-check-certificate
RUN chmod 777 Miniforge3-Linux-x86_64.sh
RUN ./Miniforge3-Linux-x86_64.sh -b
RUN rm -f Miniforge3-Linux-x86_64.sh

# Configure pip to prevent SSL issues
RUN mkdir -p ~/.config/pip
RUN printf '[global]\ntrusted-host = pypi.python.org\n\tpypi.org\n\tfiles.pythonhosted.org\n' >> ~/.config/pip/pip.conf

# Copy packages from Gitlab
RUN mkdir -p /home/app/packages
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/soil_capacity_wilting.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/harmonized-landsat-sentinel.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/gedi_canopy_height.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/rasters.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/geos5fp.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/modisci.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/koppengeiger.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/solar-apparent-time.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/sentinel-tiles.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/colored-logging.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/modland.git
RUN source ~/.credentials && cd /home/app/packages && git clone https://$GIT_USERNAME:$GIT_TOKEN@gitlab.bor.doi.net/aao-et-toolbox/sun-angles.git


# Create the python environment
RUN /root/miniforge3/bin/conda init bash
RUN cd /home/app && conda env create -f environment.yml
RUN echo "conda activate ettoolbox" >> ~/.bashrc

# Additionally, Python wil lnot use the system certs without adding in another project dependency. The cert files need to be copied into the certificate path of the
# Python ssl environment
RUN cp /etc/ssl/certs/ca-bundle.crt /root/miniforge3/envs/ettoolbox/ssl/certs/
RUN cp /etc/ssl/certs/ca-bundle.trust.crt /root/miniforge3/envs/ettoolbox/ssl/certs/

# Fix SSL issues that with the University of Maryland server within the Python ssl environment
RUN true | openssl s_client -connect gladweb1.umd.edu:443 2>/dev/null | openssl x509 >> /root/miniforge3/envs/ettoolbox/ssl/cert.pem

# Install app
RUN cd /home/app/ && pip install .

# Setup folders and permission. todo: Likely need to do this with all of them
RUN mkdir -p /home/app/ptjpl_static/GEDI_download
RUN chmod -R 777 /home/app/ptjpl_static

RUN mkdir -p /home/app/GEOS5FP_download_directory
RUN chmod -R 777 /home/app/GEOS5FP_download_directory

RUN mkdir -p /home/app/srtm_download_directory
RUN chmod -R 777 /home/app/srtm_download_directory

RUN mkdir -p /home/app/lance_download_directory
RUN chmod -R 777 /home/app/lance_download_directory
