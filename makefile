.ONESHELL:
SHELL=bash
VERSION := $(shell cat ETtoolbox/version.txt)

default:
	make install

version:
	$(info ETtoolbox Collection 2 pipeline version ${VERSION})

mamba:
ifeq ($(word 1,$(shell mamba --version)),mamba)
	@echo "mamba already installed"
else
	-conda deactivate; conda install -y -c conda-forge "mamba>=0.23"
endif

create-blank-env:
	$(info creating blank ETtoolbox environment)
	-conda run -n base mamba create -n ETtoolbox

update-env-mamba:
	mamba env update -n ETtoolbox -f ETtoolbox.yml

environment:
	make mamba
	-conda deactivate; pip install pybind11_cmake
	make create-blank-env
	make update-env-mamba

refresh-env:
	make remove
	make environment

clean-python:
	$(info cleaning python)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

clean:
	$(info cleaning build)
	-rm -rvf build
	-rm -rvf dist
	-rm -rvf *.egg-info
	-rm -rvf CMakeFiles
	-rm CMakeCache.txt
	-find . -type d -name __pycache__ -exec rm -r {} +


uninstall:
	$(info uninstalling ETtoolbox package)
	-conda run -n ETtoolbox pip uninstall ETtoolbox -y

unit-tests:
	$(info running unit tests)
	conda run -n ETtoolbox nosetests -v -w tests

unit-tests-docker:
	nosetests -v -w tests

setuptools:
	-conda run -n ETtoolbox python setup.py install

install-package:
	$(info installing ETtoolbox package)
	-make setuptools
	make clean
	make unit-tests

install-package-docker:
	python setup.py install
	make clean
	make unit-tests-docker

install:
	make environment
	make clean
	make uninstall
	make install-package

install-docker:
	make clean
	cp ERS_credentials.txt ERS_credentials
	cp spacetrack_credentials.txt spacetrack_credentials
	make install-package-docker

remove:
	# conda run -n base conda env remove -n ETtoolbox
	mamba env remove -n ETtoolbox

reinstall-hard:
	make remove
	make install

reinstall-soft:
	make uninstall
	make install-package

docker-build:
	docker build -t ettoolbox .
