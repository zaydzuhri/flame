# run_eval.py
print("Importing fla explicitly before running lm_eval...")
import fla  # Ensure registration happens
print("Imported fla.")

# Now import and run the lm_eval main entry point
# Adjust this import based on how lm_eval is installed/structured
# This is a common pattern, but might need tweaking:
from lm_eval.__main__ import cli_evaluate

print("Running lm_eval cli_evaluate...")
cli_evaluate()