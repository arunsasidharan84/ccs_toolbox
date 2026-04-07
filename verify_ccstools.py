import ccstools
print(f"Successfully imported ccstools version {ccstools.__version__}")

try:
    from ccstools import compute_psd, compute_irasa, compute_acw
    print("Successfully imported core functions from ccstools")
except ImportError as e:
    print(f"ImportError: {e}")

try:
    from ccstools.ccs_eeg.pipeline import run_ccs_pipeline
    print("Successfully imported run_ccs_pipeline")
except ImportError as e:
    print(f"ImportError in pipeline: {e}")
