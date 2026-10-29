"""
configure.py
Takes user input file and returns settings namespace object

Modified from isEE_framework: https://github.com/emmeburgin/isEE_framework/blob/main/isee/infrastructure/configure.py
"""

import argparse
import pytraj       # to support pytraj calls in input file
import mdtraj       # to support mdtraj calls in the input file
import numpy        # to support numpy  calls in input file
import numpy as np  # to support numpy  calls even if called as np
import sys
import os
import shutil
import pickle
from jinja2 import Environment, FileSystemLoader
import typing
import pydantic

def configure(input_file, user_working_directory=''):
    """
    Configure the settings namespace based on the config file.

    Parameters
    ----------
    input_file : str
        Name of the configuration file to read
    user_working_directory : str
        User override for working directory (overrides value in input_file), ignored if set to ''

    Returns
    -------
    settings : argparse.Namespace
        Settings namespace object

    """

    class Settings(pydantic.BaseModel):
        # This class initializes the settings object with type hints. After being built, it gets exported as an
        # argparse.Namelist object, just for convenience.

        md_engine: str
        working_directory: str
        feature_data_directory: str
        log_directory: str
        md_data_directory: str
        # etc, etc, ...

    # Import config file line-by-line using exec()
    lines = open(input_file, 'r').readlines()
    line_index = 0
    for line in lines:      # each line in the input file is just python code setting a variable...
        line_index += 1
        try:
            exec(line)      # ... this means that comments are supported using '#' and whitespace is ignored.
        except Exception as e:
            raise ValueError('error raised while reading line ' + str(int(line_index)) + ' of configuration file '
                             + input_file + ': ' + str(e))

    # Define settings namespace to store all these variables
    config_dict = {}
    config_dict.update(locals())
    settings = argparse.Namespace()
    settings.__dict__.update(Settings(**config_dict))

    # Override working directory if provided with user_working_directory
    if user_working_directory:
        settings.working_directory = user_working_directory


    return settings

if __name__ == "__main__":
    configure('','')
