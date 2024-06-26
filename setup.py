from os.path import join, abspath, dirname

from setuptools import setup, find_packages


def version():
    with open(join(abspath(dirname(__file__)), "ETtoolbox", "version.txt"), "r") as file:
        return file.read()


setup(
    version=version(),
    zip_safe=False,
    packages=find_packages(),
    package_data={'': ["*"]}
)
