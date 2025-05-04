# run_suite.py
# Updated with menu and specific actions.
# PATCHED: Removed timed input, automatically uses current working directory.
# PATCHED: Corrected target_folder logic to use the script's directory, not cwd.

import subprocess
import sys
import os
import webbrowser # Added for opening HTML
# Removed time and threading imports as they are no longer needed

# --- Configuration ---
# SCRIPT_DIR is now the directory containing *this* script (run_suite.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable # Use the same python that runs this script

# Define paths relative to THIS script's directory
ANALYZER_SCRIPT = os.path.join(SCRIPT_DIR, "lexicon_analyzer.py")
VISUALIZER_SCRIPT = os.path.join(SCRIPT_DIR, "lexicon_visualizer.py")
TRANSFORMER_SCRIPT = os.path.join(SCRIPT_DIR, "lexicon_transformer.py")
PROCESSOR_SCRIPT = os.path.join(SCRIPT_DIR, "lexicon_processor.py")
OPTIMIZER_SCRIPT = os.path.join(SCRIPT_DIR, "lexicon_optimizer.py")
WORDCOUNTER_SCRIPT = os.path.join(SCRIPT_DIR, "wordcounter.py") # Assuming you have this script
SETTINGS_INI = os.path.join(SCRIPT_DIR, "lexicon_settings.ini") # Settings are alongside the suite scripts
HTML_REPORT_BASE = "_LEXICON.html" # Base name, path constructed later using target_folder

# Sequence for option 1
FULL_SEQUENCE = [
    ANALYZER_SCRIPT,
    VISUALIZER_SCRIPT,
    TRANSFORMER_SCRIPT,
    PROCESSOR_SCRIPT,
    OPTIMIZER_SCRIPT,
]

# Map choices to scripts for individual runs
SCRIPT_MAP = {
    '2': ANALYZER_SCRIPT,
    '3': VISUALIZER_SCRIPT,
    '4': TRANSFORMER_SCRIPT,
    '5': PROCESSOR_SCRIPT,
    '6': OPTIMIZER_SCRIPT,
    '7': WORDCOUNTER_SCRIPT,
}

# --- ASCII Art ---
ASCII_ART = r"""
___  ___     ___                             __     __      __         ___  ___
 |  |__  \_/  |      /\  |\ |  /\  |    \ / /__` | /__`    /__` |  | |  |  |__
 |  |___ / \  |     /~~\ | \| /~~\ |___  |  .__/ | .__/    .__/ \__/ |  |  |___

                        Text Analysis Suite v1.5
                        by Fentible/EvilDoorBotTwin
"""

# --- Helper Function to Run Scripts ---
def run_script(script_path, target_folder_for_data):
    """
    Runs a given Python script, passing the target folder for data.
    The script's execution working directory (cwd) is set to its own location (SCRIPT_DIR)
    to ensure it can find any relative resources it needs (like models, templates).
    """
    script_name = os.path.basename(script_path)
    if not os.path.isfile(script_path):
        print(f"\nError: Script not found: {script_path}")
        return False
    if not os.path.isdir(target_folder_for_data):
         print(f"\nError: Target data folder not found for {script_name}: {target_folder_for_data}")
         return False

    print(f"\n--- Running {script_name} ---")
    print(f"--- Target Data Folder: {target_folder_for_data} ---")
    try:
        # Pass the target_folder_for_data as a command-line argument
        # Set the working directory (cwd) for the child script to be SCRIPT_DIR
        # This ensures relative paths *within* the child scripts work correctly if they
        # need to load models/etc relative to their own location.
        # The child script should use the *argument* for data file locations.
        process = subprocess.run(
            [PYTHON_EXE, script_path, target_folder_for_data],
            check=True,
            text=True,
            cwd=SCRIPT_DIR # Execute the script as if we are in its directory
        )
        print(f"--- {script_name} Completed Successfully ---")
        return True

    except FileNotFoundError:
         print(f"\nError: Python executable not found: {PYTHON_EXE}")
         return False
    except subprocess.CalledProcessError as e:
        print(f"\nError: {script_name} failed with exit code {e.returncode}")
        # Optionally print stdout/stderr from the failed script
        # if e.stdout: print("STDOUT:\n", e.stdout)
        # if e.stderr: print("STDERR:\n", e.stderr)
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred while running {script_name}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # --- Set Target Folder Path ---
    # *** CORRECTED LOGIC ***
    # Use the directory containing *this* script (run_suite.py) as the
    # target folder where data files (_LEXICON.html, .npy, plots/, etc.) are expected.
    # This assumes the data files reside alongside the suite scripts in the subfolder.
    target_folder = SCRIPT_DIR
    # print(f"Using script's directory as target folder: {target_folder}") # Optional debug print

    # Validate target folder
    if not os.path.isdir(target_folder):
        print(f"\nCRITICAL Error: Target folder (script directory) not found or is not a directory: {target_folder}")
        sys.exit(1)

    # Construct path to HTML report within the determined target folder
    html_report_path = os.path.join(target_folder, HTML_REPORT_BASE)
    # Construct path to settings INI relative to *this* script's location (already defined)
    settings_ini_path = SETTINGS_INI # Use the constant defined earlier


# --- Main Menu Loop ---
    while True:
        # Clear console screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Now print the separator and the rest of the menu
        print("\n" + "="*60)
        print(ASCII_ART)
        print("="*60)
        # Display the determined target folder where data files are expected
        print(" Target Data Folder:", target_folder)
        print("-"*60)
        print(" Select an option:")
        print("  1) Run Full Suite (Analyze > Visualize > Transform > Process > Optimize)")
        print("  2) Run Analyzer (Metrics Calculation)")
        print("  3) Run Visualizer (Generate Plots)")
        print("  4) Run Transformer (LLM Dendrogram Analysis + Numbering)")
        print("  5) Run Processor (LLM Interpretations)")
        print("  6) Run Optimizer (HTML Cleanup)")
        print("  7) Run Word Counter (Requires wordcounter.py)")
        print("  8) Edit Settings (Opens lexicon_settings.ini)")
        print("  9) View HTML Report (Opens _LEXICON.html)")
        print("  0) Quit")
        print("-"*60)

        choice = input(" Enter your choice (0-9): ")

        if choice == '0':
            print("Exiting Text Analysis Suite.")
            break
        elif choice == '1':
            print(f"\nStarting Full Suite")
            print(f"Operating on data in folder: {target_folder}")
            print("="*50)
            all_success = True
            for script_path in FULL_SEQUENCE:
                # Pass the determined target_folder to each script
                success = run_script(script_path, target_folder)
                if not success:
                    all_success = False
                    print(f"\nError during Full Suite run. Aborting sequence.")
                    break # Stop sequence on error
            print("="*50)
            if all_success:
                print(f"Full Suite finished successfully.")
                print(f"Check the '{target_folder}' directory for outputs.")
            else:
                print(f"Full Suite finished with errors.")
            print("="*50)

        elif choice in SCRIPT_MAP:
            script_to_run = SCRIPT_MAP[choice]
            # Pass the determined target_folder to the script
            run_script(script_to_run, target_folder)
            print(f"\nCheck the '{target_folder}' directory for outputs/updates.")

        elif choice == '8':
            # Use the path relative to this script's location
            if not os.path.isfile(settings_ini_path):
                print(f"\nError: Settings file not found: {settings_ini_path}")
            else:
                print(f"\nAttempting to open '{os.path.basename(settings_ini_path)}' with notepad...")
                try:
                    # Use startfile on Windows for default editor, more robust than assuming notepad
                    if os.name == 'nt':
                        os.startfile(settings_ini_path)
                    else: # Basic fallback for Linux/Mac - might need 'xdg-open' or 'open'
                         opener = "open" if sys.platform == "darwin" else "xdg-open"
                         subprocess.run([opener, settings_ini_path])
                except FileNotFoundError:
                    print(f"Error: Default editor command not found. Cannot open settings file automatically.")
                    print(f"Please open manually: {settings_ini_path}")
                except Exception as e:
                    print(f"Error opening settings file: {e}")
                    print(f"Please open manually: {settings_ini_path}")


        elif choice == '9':
            # Use the path constructed within the target_folder
            if not os.path.isfile(html_report_path):
                print(f"\nError: HTML report not found: {html_report_path}")
                print("       Run Analyzer (2) and Visualizer (3) first.")
            else:
                print(f"\nAttempting to open '{HTML_REPORT_BASE}' in web browser...")
                try:
                    # Use file:// URI scheme for local files
                    webbrowser.open(f"file://{os.path.realpath(html_report_path)}")
                except Exception as e:
                    print(f"Error opening HTML report: {e}")

        else:
            print("\nInvalid choice. Please enter a number between 0 and 9.")

        input("\nPress Enter to return to the menu...") # Pause before showing menu again