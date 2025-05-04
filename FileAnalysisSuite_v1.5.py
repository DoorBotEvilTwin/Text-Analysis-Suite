# FileAnalysisSuite_v1.5.py
# Shortcut script to launch the main run_suite.py located in a subfolder.

import os
import sys
import subprocess

# --- Configuration ---
# The name of the subfolder containing the actual suite scripts
SUITE_SUBFOLDER_NAME = "FAS_1.5"

# --- Find Paths ---
try:
    # Directory where this shortcut script is located
    shortcut_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the subfolder containing the suite
    suite_folder_path = os.path.join(shortcut_dir, SUITE_SUBFOLDER_NAME)

    # Path to the actual run_suite.py inside the subfolder
    run_suite_script_path = os.path.join(suite_folder_path, "run_suite.py")

    # Get the python executable used to run this shortcut
    python_exe = sys.executable

except Exception as e:
    print(f"Error determining script paths: {e}")
    input("Press Enter to exit.")
    sys.exit(1)

# --- Validate Paths ---
if not os.path.isdir(suite_folder_path):
    print(f"Error: Suite subfolder not found at expected location:")
    print(f"       '{suite_folder_path}'")
    print(f"       Ensure the '{SUITE_SUBFOLDER_NAME}' folder exists in the same directory as this shortcut.")
    input("Press Enter to exit.")
    sys.exit(1)

if not os.path.isfile(run_suite_script_path):
    print(f"Error: Main script 'run_suite.py' not found inside the subfolder:")
    print(f"       '{run_suite_script_path}'")
    input("Press Enter to exit.")
    sys.exit(1)

# --- Execute Main Script ---
# print(f"Launching Text Analysis Suite from: {suite_folder_path}")
# print("-" * 30)

try:
    # Run the main script using the same Python interpreter
    # Crucially, set the current working directory (cwd) to the shortcut's directory,
    # so the modified run_suite.py uses this directory as its default target.
    process = subprocess.run(
        [python_exe, run_suite_script_path],
        check=True,  # Raise an exception if the script fails
        cwd=shortcut_dir # Set the working directory for the launched script
    )
    print("-" * 30)
    print("Suite finished.")

except FileNotFoundError:
     print(f"\nError: Python executable not found: {python_exe}")
except subprocess.CalledProcessError as e:
    print(f"\nError: The main suite script failed with exit code {e.returncode}.")
except Exception as e:
    print(f"\nAn unexpected error occurred while launching the suite: {e}")

# Keep window open until user presses Enter (optional)
# input("\nPress Enter to close this launcher window.")