FROM condaforge/mambaforge

ENV APP_ROOT /ET-Toolbox-Product-Generator

# update Ubuntu package manager
RUN apt-get update
RUN apt-get -y install gcc
# install fish shell
RUN apt-add-repository ppa:fish-shell/release-3; apt-get -y install fish; chsh -s /usr/local/bin/fish; mamba init fish
# install dependencies
RUN mamba update -y mamba
RUN mamba update -y --all
RUN mamba install -y "python=3.10"
RUN mamba install -y pygrib
RUN mamba install -y python-dateutil
RUN mamba install -y wget
RUN mamba install -y xtensor xtensor-python
RUN mamba install -y "gdal>3.1" "rasterio>1.0.0" "setuptools!=58" "shapely<2.0.0" "tensorflow!=2.11.0"
RUN pip install astropy beautifulsoup4 cmake descartes ephem geopandas h5py imageio imageio-ffmpeg jupyter keras matplotlib mgrs netcdf4 nose pip pycksum pygeos pyhdf pyresample pysolar pystac-client requests scikit-image sentinelsat spacetrack termcolor untangle urllib3 xmltodict

# install app
RUN mkdir /ET-Toolbox-Product-Generator
ADD . /ET-Toolbox-Product-Generator
WORKDIR /ET-Toolbox-Product-Generator
RUN python setup.py install
