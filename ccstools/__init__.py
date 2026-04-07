# CCS EEG Toolbox (ccstools)
# (c) 2024 Arun Sasidharan

from . import eegfeatures
from . import sigproc
from . import fileio
from . import plot

# Expose some common functions for easier access
from .eegfeatures import compute_psd, compute_fooof, compute_irasa, compute_nonlinear, compute_acw
from .sigproc import bootstrapERP, pca, smooth, detecterppeak

__version__ = "0.1.0"
