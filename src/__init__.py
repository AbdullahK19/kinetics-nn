"""
Kinetics Neural Network Package

A data-driven machine learning pipeline for identifying chemical reaction
mechanisms from concentration-time profiles.
"""

__version__ = "0.2.0"
__author__ = "Abdullah Kashif"

from . import ode_solver
from . import excel_driver
from . import dataset_builder
from . import preprocess
from . import model
from . import train
from . import predict

__all__ = [
    'ode_solver',
    'excel_driver',
    'dataset_builder',
    'preprocess',
    'model',
    'train',
    'predict'
]
