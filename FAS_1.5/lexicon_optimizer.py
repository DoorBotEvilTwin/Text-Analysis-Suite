# lexicon_optimizer.py
# Purpose: Performs LLM Meta-Analysis on the initial HTML state.
# Focuses on:
#   1. Estimating LLM context size needed.
#   2. Performing LLM Meta-Analysis on the initial HTML state.
#   3. Appending the raw LLM Meta-Analysis output to the HTML.
#   4. Adding a TOC link for the LLM Meta-Analysis section.
# REMOVED: All formatting cleanup rules (**/* bold, list conversion, br collapse, empty p removal).
# REMOVED: Markdown rendering functionality.
# REMOVED: Styling for Synthesis section.
# REMOVED: TOC link addition for Processor sections.
# PATCHED: Changed TOKENS_PER_IMAGE estimate.
# PATCHED: Use <pre> tag for LLM Meta-Analysis output to ensure raw text display. # <<< THIS FIX

import os
import sys
import logging
import re # Import regex (might be needed for TOC search)
import time
import threading
import configparser # Added for LLM config
import json # Added for LLM payload
import math # Added for ceiling function in token estimation

# --- Dependency Check ---
try:
    # <<< Ensure NavigableString is imported >>>
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    print("ERROR: 'beautifulsoup4' library not found. Install using: py -m pip install beautifulsoup4")
    print("       HTML optimization cannot be performed.")
    BS4_AVAILABLE = False
    sys.exit(1)

# --- Try importing lxml for potentially better parsing ---
try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    lxml = None
    LXML_AVAILABLE = False

# --- Requests library needed for local LLM calls ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    # Log warning later if LLM is enabled but requests is missing

# --- Markdown library REMOVED ---


# --- Configuration ---
HTML_FILENAME = "_LEXICON.html"
LOG_FILENAME = "_LEXICON_errors.log" # Appends to the same log
INI_FILENAME = "lexicon_settings.ini" # For LLM config

# --- LLM Configuration ---
MAX_RETRIES = 2
RETRY_DELAY = 3
DEFAULT_MAX_TOKENS = 256 # Default if not in INI
# <<< Adjusted multiplier back from 64, 16 seems more reasonable for meta-analysis >>>
SUMMARY_MAX_TOKENS_MULTIPLIER = 16.0 # Multiplier for final summary token limit
# Heuristic values for token estimation
CHARS_PER_TOKEN = 4
TOKENS_PER_IMAGE = 175 # <<< FIX: Changed from 750
PROMPT_OVERHEAD_TOKENS = 200 # Estimate for instructions, structure, etc.

# --- Basic Logging Setup ---
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
log_level = logging.INFO # Change to logging.DEBUG for verbose cleanup steps
# Remove existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'), # Append mode 'a'
        logging.StreamHandler()
    ]
)
logging.info(f"\n{'='*10} Starting HTML Optimization Run (LLM Meta-Analysis Only) {'='*10}")


# --- START: Added Helper Function for Timed Input ---
def get_input_with_timeout(prompt, timeout):
    """Gets input from the user with a timeout and countdown."""
    user_input = [None] # Use a list to allow modification from thread
    interrupted = [False] # Flag to signal if user provided input

    def _get_input_target(prompt_str, result_list, interrupt_flag):
        try:
            result_list[0] = input(prompt_str)
            interrupt_flag[0] = True # Signal that input was received
        except EOFError: # Handle case where input stream is closed unexpectedly
            pass

    thread = threading.Thread(target=_get_input_target, args=(prompt, user_input, interrupted))
    thread.daemon = True # Allow program to exit even if thread is stuck
    thread.start()

    # Countdown loop
    for i in range(timeout, 0, -1):
        if interrupted[0]: break
        print(f"\rNo path provided. Using current directory in {i}s... Press Enter or type path to override: ", end="", flush=True)
        time.sleep(1)
        if interrupted[0]: break
    print("\r" + " " * 100 + "\r", end="", flush=True) # Overwrite with spaces
    if interrupted[0] and user_input[0] is not None: return user_input[0]
    else: return None
# --- END: Added Helper Function ---


# --- START: LLM Helper Functions ---
def load_config(ini_path):
    """Loads configuration from the INI file."""
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        logging.error(f"Configuration file '{INI_FILENAME}' not found at '{ini_path}'. Cannot use LLM features.")
        return None
    try:
        config.read(ini_path)
        logging.info(f"Loaded configuration from '{INI_FILENAME}'.")
        return config
    except configparser.Error as e:
        logging.error(f"Error parsing configuration file '{INI_FILENAME}': {e}")
        return None

def get_llm_config(config):
    """Extracts LLM configuration and checks if enabled."""
    if not config or 'LocalLLM' not in config:
        logging.warning("[LocalLLM] section not found in INI file. LLM meta-analysis disabled.")
        return None, False
    local_llm_enabled = config.getboolean('LocalLLM', 'enabled', fallback=False)
    if not local_llm_enabled:
        logging.info("Local LLM mode is disabled in configuration. LLM meta-analysis disabled.")
        return None, False

    local_llm_config = dict(config.items('LocalLLM'))
    if not local_llm_config.get('api_base'):
        logging.error("Local LLM mode enabled, but 'api_base' is not set in [LocalLLM] section. LLM meta-analysis disabled.")
        return None, False
    if 'max_tokens' not in local_llm_config:
        local_llm_config['max_tokens'] = str(DEFAULT_MAX_TOKENS)
        logging.info(f"Setting default max_tokens to {DEFAULT_MAX_TOKENS}")

    # Check for requests library if LLM is enabled
    if not REQUESTS_AVAILABLE:
        logging.error("'requests' library not found (pip install requests). LLM meta-analysis disabled.")
        return None, False

    logging.info(f"Local LLM mode is enabled for meta-analysis. Config: {local_llm_config}")
    return local_llm_config, True

def call_local_llm_api_text(prompt, llm_config, max_tokens_override=None):
    """
    Calls a local LLM API (e.g., KoboldCpp) via HTTP POST for text generation.
    Allows overriding max_tokens. Uses 'processor_model' or 'model'.
    """
    if not REQUESTS_AVAILABLE or requests is None:
        return "Error: 'requests' library not installed."

    api_base = llm_config.get('api_base')
    if not api_base:
        return "Error: 'api_base' URL not configured."

    current_max_tokens = max_tokens_override if max_tokens_override is not None else int(llm_config.get('max_tokens', DEFAULT_MAX_TOKENS))

    # Use processor_model if available, otherwise fallback to model, then a default
    model_name = llm_config.get('processor_model', llm_config.get('model', 'local-text-model'))

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(llm_config.get('temperature', 0.0)),
        "max_tokens": current_max_tokens,
    }
    if 'top_p' in llm_config: payload['top_p'] = float(llm_config['top_p'])
    if 'top_k' in llm_config: payload['top_k'] = int(llm_config['top_k'])

    payload = {k: v for k, v in payload.items() if v is not None}
    endpoint = api_base # Assumes endpoint is for chat completions

    logging.debug(f"Attempting local LLM text call to: {endpoint} with model {model_name}")
    logging.debug(f"Payload: {json.dumps(payload, indent=2)}")

    last_error = None
    for attempt in range(MAX_RETRIES):
        response = None
        try:
            logging.info(f"Sending request to LLM for meta-analysis (Attempt {attempt+1}/{MAX_RETRIES})...") # Updated log message
            response = requests.post(endpoint, json=payload)
            logging.debug(f"LLM request sent. Status Code: {response.status_code if response else 'N/A'}")
            if response is None: raise requests.exceptions.RequestException("Request returned None unexpectedly.")
            response.raise_for_status()

            raw_response_text = response.text
            logging.debug(f"Raw response text received: {raw_response_text[:500]}...")
            response_json = response.json()
            logging.debug(f"LLM Response JSON: {json.dumps(response_json, indent=2)}")

            if 'choices' in response_json and len(response_json['choices']) > 0:
                 message = response_json['choices'][0].get('message', {})
                 content = message.get('content')
                 if content:
                      logging.info("Local LLM call successful.")
                      return content.strip()
                 else: last_error = "Error: 'content' field missing in response message."; logging.warning(last_error)
            else: last_error = "Error: Expected 'choices' array missing or empty in response."; logging.warning(last_error)

        except requests.exceptions.ConnectionError as e: last_error = f"Error: Connection failed to {endpoint} ({e}). Is the backend running?"; logging.error(last_error); break
        except requests.exceptions.RequestException as e: response_text_on_error = response.text[:500] if response is not None else "N/A"; status_code = response.status_code if response is not None else "N/A"; last_error = f"Error: Request failed (Status: {status_code}, Error: {e}). Response: {response_text_on_error}"; logging.error(f"Local LLM request failed (Attempt {attempt + 1}/{MAX_RETRIES}): {last_error}")
        except json.JSONDecodeError as e: raw_text_on_decode_error = response.text if response is not None else "N/A"; last_error = f"Error: Failed to decode JSON response from {endpoint}."; logging.error(f"{last_error} Raw Response: '{raw_text_on_decode_error}'", exc_info=False); break
        except Exception as e: last_error = f"Error: Unexpected error during local LLM call ({type(e).__name__}: {e})"; logging.error(last_error, exc_info=True); break

        if attempt < MAX_RETRIES - 1: logging.info(f"Retrying in {RETRY_DELAY}s..."); time.sleep(RETRY_DELAY)
        else: logging.error(f"LLM call failed after {MAX_RETRIES} attempts.")

    return last_error if last_error else "Error: LLM call failed after multiple retries."
# --- END: LLM Helper Functions ---


# --- Core Optimization Logic ---

# --- REMOVED: _process_text_node_formatting function ---
# --- REMOVED: convert_markdown_in_node function ---

def optimize_html_formatting(html_path, llm_config, llm_enabled):
    """Reads HTML, performs LLM meta-analysis, adds TOC link, and saves."""
    if not BS4_AVAILABLE:
        logging.error("BeautifulSoup4 not available. Cannot optimize HTML.")
        return False

    if not os.path.exists(html_path):
        logging.error(f"HTML file not found for optimization: {html_path}")
        return False

    logging.info(f"Optimizing HTML file (Meta-Analysis Only): {html_path}")
    parser_to_use = 'lxml' if LXML_AVAILABLE else 'html.parser'
    toc_links_added = 0
    made_changes = False # Flag to track if any changes were made
    llm_meta_analysis_added = False # Flag for new meta-analysis

    try:
        # --- Read and Parse ---
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, parser_to_use)
        body = soup.body
        if not body:
            logging.error("HTML file is missing a <body> tag. Cannot proceed.")
            return False

        # --- START: LLM Context Size Estimation ---
        logging.info("Estimating LLM context size requirements...")
        main_h1 = soup.find('h1')
        anchor_tag = None
        if main_h1:
            last_p_after_h1 = None; sibling = main_h1.find_next_sibling()
            while sibling:
                if sibling.name == 'p': last_p_after_h1 = sibling
                elif sibling.name != 'p' and sibling.name is not None: break
                sibling = sibling.find_next_sibling()
            anchor_tag = last_p_after_h1 if last_p_after_h1 else main_h1 # Fallback to H1
        else:
            logging.warning("Could not find main H1 tag. Context estimator insertion might be suboptimal.")
            anchor_tag = body.find('p') # Fallback to first p tag in body

        estimator_p_tag = None # Initialize

        if anchor_tag:
            try:
                page_text = body.get_text(separator=' ', strip=True); num_chars = len(page_text)
                images = body.find_all('img'); num_images = len(images)
                text_tokens = math.ceil(num_chars / CHARS_PER_TOKEN); image_tokens = num_images * TOKENS_PER_IMAGE # Uses updated TOKENS_PER_IMAGE
                estimated_total_tokens = text_tokens + image_tokens + PROMPT_OVERHEAD_TOKENS
                logging.info(f"Context Estimation: Chars={num_chars}, TextTokens={text_tokens}, Images={num_images}, ImageTokens={image_tokens}, Overhead={PROMPT_OVERHEAD_TOKENS} -> TotalEst={estimated_total_tokens}")

                estimator_p_tag = soup.new_tag('p', style="font-weight: bold; color: #006400;") # Dark green
                estimator_p_tag.string = f"LLM Context Size Estimator: ~{estimated_total_tokens:,} tokens (based on {num_chars:,} chars and {num_images} images)"
                anchor_tag.insert_after(estimator_p_tag) # Insert after the anchor
                made_changes = True # Mark change

                print("\n" + "="*30)
                print(f" LLM Context Size Estimate: ~{estimated_total_tokens:,} tokens")
                print(f" (Based on {num_chars:,} text characters and {num_images} images)")
                print(" Make sure your LLM context window is large enough for the meta-analysis step.")
                print("="*30)
                for i in range(5, 0, -1):
                    print(f"\r Starting LLM meta-analysis in {i}s... (Press Ctrl+C to cancel) ", end="", flush=True)
                    time.sleep(1)
                print("\r" + " " * 70 + "\r", end="", flush=True) # Clear countdown line
                print(" Proceeding with LLM meta-analysis...")

            except Exception as e:
                logging.error(f"Error during context estimation or countdown: {e}", exc_info=True)
                error_p = soup.new_tag('p', style="color: red; font-weight: bold;"); error_p.string = "Error calculating LLM context size estimate."
                if anchor_tag: anchor_tag.insert_after(error_p) # Try inserting error message
                made_changes = True
        else:
            logging.warning("Could not find a suitable anchor tag (H1 or P). Cannot insert context estimate.")
        # --- END: LLM Context Size Estimation ---


        # --- START: LLM Page Meta-Analysis (RUNS FIRST on initial soup) ---
        if llm_enabled and llm_config:
            logging.info("Attempting LLM meta-analysis of the entire page...")
            try:
                # 1. Extract Content for Prompt (Capture everything currently in body)
                page_text_content = body.get_text(separator='\n', strip=True)
                image_tags = body.find_all('img')
                image_references = [f"[Image: {img.get('alt', f'Image {i+1}')}]" for i, img in enumerate(image_tags)]
                image_ref_string = "\n".join(image_references)
                full_content_for_prompt = f"Report Text Content:\n{page_text_content}\n\nVisualizations Present:\n{image_ref_string}"

                # 2. Construct Prompt (Refocused for High-Level Synthesis)
                summary_prompt = f"""1. Please perform a high-level meta-analysis of the following comprehensive text analysis report. This report includes calculated metrics, metric descriptions, generated plots (referenced by [Image: Alt Text]), and potentially LLM-generated interpretations/synthesis. There are also 12 images to review.

2. Your task is to provide a detailed, elaborate overview focusing *primarily* on the big picture revealed by the *entire* report. Synthesize all sections togther. Instead, focus *extensively* on the following aspects:
3.  What are some conclusions that can be drawn about the linguistic profile, style, or thematic content of the analyzed text dataset as a whole, based on the combined evidence from metrics, plots, and interpretations? Elaborate on the reasoning for each conclusion. Maybe give examples.
4. Identify and discuss in detail any notable consistencies (e.g., multiple metrics pointing to high complexity) or contradictions (e.g., high lexical diversity but simple sentence structure, or conflicting sentiment signals between different methods) observed across the different analysis sections (numerical metrics, visualizations, LLM interpretations). Explore potential reasons for these patterns.
5. Briefly compare the insights gained from different types of meta-analysis presented in the report. For example, did the visual plots reveal patterns not obvious from the tables? Did the LLM interpretations offer insights beyond the numerical metrics or rule-based analysis? What unique perspective did each major analysis component (numerical, visual, LLM-based) contribute to the overall understanding? How does this relate to the actual content summaries of each file? What insights can we gain from this?
6. Aim for a thoughtful synthesis that integrates the various pieces of info into a cohesive understanding of the dataset's characteristics. Use clear paragraphs. Write extensively yet concisely.

7. Report Content to Analyze:
--- START REPORT CONTENT ---
{full_content_for_prompt}
--- END REPORT CONTENT ---

8. Dive deep into thematic content and how it correlates with the metrics to generate the detailed meta-analysis:""" # Removed Markdown instruction

                # 3. Determine Max Tokens
                base_max_tokens = int(llm_config.get('max_tokens', DEFAULT_MAX_TOKENS))
                summary_tokens = int(base_max_tokens * SUMMARY_MAX_TOKENS_MULTIPLIER)
                logging.info(f"Requesting max_tokens={summary_tokens} for page meta-analysis.")

                # 4. Call LLM
                summary_text = call_local_llm_api_text(
                    summary_prompt,
                    llm_config,
                    max_tokens_override=summary_tokens
                )

                # 5. Append Meta-Analysis Section (Using <pre> tag)
                if summary_text and not summary_text.startswith("Error:"):
                    summary_section_id = "llm_meta_analysis_section" # Renamed ID
                    existing_summary = soup.find('div', id=summary_section_id)
                    if existing_summary: existing_summary.decompose()

                    summary_div = soup.new_tag('div', id=summary_section_id, style="margin-top: 30px; padding: 15px; border: 1px solid #006400; background-color: #f0fff0; border-radius: 4px;")
                    summary_title = soup.new_tag('h2', style="margin-top: 0; margin-bottom: 10px; color: #006400;")
                    summary_title.string = "Comprehensive LLM Meta-Analysis" # Renamed Title
                    summary_div.append(summary_title)

                    # --- START FIX: Use <pre> tag for raw text display ---
                    # Create a <pre> tag
                    summary_content_pre = soup.new_tag('pre', style="white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: inherit; line-height: 1.6;")
                    # Append the raw summary text as a NavigableString inside the <pre> tag
                    summary_content_pre.append(NavigableString(summary_text))
                    summary_div.append(summary_content_pre)
                    # --- END FIX ---

                    body.append(summary_div) # Append to the end of the body
                    llm_meta_analysis_added = True
                    made_changes = True
                    logging.info("Successfully generated and appended LLM meta-analysis section (using <pre> tag) to HTML.")

                    # TOC link added later after formatting

                else:
                    logging.error(f"LLM meta-analysis failed: {summary_text}")
                    error_div = soup.new_tag('div', id="llm_meta_analysis_section", style="margin-top: 30px; padding: 15px; border: 1px solid #cc0000; background-color: #fff0f0; border-radius: 4px;")
                    error_title = soup.new_tag('h2', style="margin-top: 0; margin-bottom: 10px; color: #cc0000;"); error_title.string = "LLM Meta-Analysis (Failed)"; error_div.append(error_title)
                    error_p = soup.new_tag('p'); error_p.string = f"Failed to generate meta-analysis. Error: {summary_text}"; error_div.append(error_p)
                    body.append(error_div); made_changes = True

            except Exception as e:
                logging.error(f"Error during LLM meta-analysis process: {e}", exc_info=True)

        elif not llm_enabled:
            logging.info("LLM meta-analysis skipped (disabled in INI).")
        else: # llm_config is None or requests missing
             logging.warning("LLM meta-analysis skipped (configuration error or missing 'requests' library).")
        # --- END: LLM Page Meta-Analysis ---


        # --- REMOVED: Formatting Cleanup Rules Section ---
        logging.info("Skipping formatting cleanup rules.")


        # --- START: TOC Update (Only for Meta-Analysis) ---
        toc_div = soup.find('div', id='toc')
        if toc_div:
            toc_list = toc_div.find('ul')
            if toc_list:
                # Add TOC link for Meta-Analysis if it was added
                meta_analysis_div = soup.find('div', id='llm_meta_analysis_section')
                if meta_analysis_div and not toc_list.find('a', href='#llm_meta_analysis_section'):
                    summary_toc_li = soup.new_tag('li')
                    summary_toc_a = soup.new_tag('a', href='#llm_meta_analysis_section')
                    summary_toc_a.string = "LLM Meta-Analysis" # Renamed Link Text
                    summary_toc_li.append(summary_toc_a)
                    toc_list.append(summary_toc_li)
                    toc_links_added += 1
                    made_changes = True # Mark change
                    logging.info("Added TOC link for LLM meta-analysis.")
                elif meta_analysis_div:
                    logging.debug("TOC link for LLM meta-analysis already exists.")
                else:
                    logging.debug("LLM meta-analysis section not found, skipping TOC link.")
            else: logging.warning("Could not find UL tag within TOC div.")
        else: logging.warning("Could not find TOC div (id='toc').")
        # --- END: TOC Update ---


        # --- REMOVED: Style Synthesis Section ---
        logging.info("Skipping Synthesis section styling.")


        # --- REMOVED: Final Markdown Rendering Section ---
        logging.info("Skipping final Markdown rendering step.")


        # --- Save Optimized HTML ---
        if made_changes:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(str(soup.prettify()))
            # Updated log message
            logging.info(f"HTML optimization complete. Added {toc_links_added} TOC links. LLM Meta-Analysis Added: {llm_meta_analysis_added}.")
        else:
            logging.info("HTML optimization complete. No changes made (context estimate, meta-analysis, or TOC link).")

        return True

    except Exception as e:
        logging.error(f"Error during HTML optimization: {e}", exc_info=True)
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Get Target Folder
    if len(sys.argv) < 2:
        prompt = f"Enter path to folder containing '{HTML_FILENAME}' (or wait 5s for current: {os.getcwd()}): "
        user_path = get_input_with_timeout(prompt, 5)
        if user_path is None: target_folder = os.getcwd(); print(f"\nTimeout! Using current directory: {target_folder}")
        else:
            user_path_stripped = user_path.strip()
            if not user_path_stripped: target_folder = os.getcwd(); print(f"\nNo path entered. Using current directory: {target_folder}")
            else: target_folder = user_path_stripped; print(f"\nUsing provided path: {target_folder}")
    else:
        target_folder = sys.argv[1]
        print(f"Using folder path from command line argument: {target_folder}")

    # --- Validate Folder Path ---
    if not os.path.isdir(target_folder):
        log_func = logging.critical if 'logging' in sys.modules else print
        log_func(f"CRITICAL Error: Target folder not found or is not a directory: {target_folder}")
        sys.exit(1)

    html_path = os.path.join(target_folder, HTML_FILENAME)
    ini_path = os.path.join(target_folder, INI_FILENAME) # Assume INI is in the target folder too

    # --- Load LLM Config ---
    config = load_config(ini_path)
    llm_config, llm_enabled = get_llm_config(config) # Returns None, False if config fails or LLM disabled

    # --- Run Optimization ---
    success = optimize_html_formatting(html_path, llm_config, llm_enabled) # Pass LLM config

    if success:
        logging.info("HTML optimization script finished successfully.")
    else:
        logging.error("HTML optimization script finished with errors.")
        sys.exit(1)

    logging.info("--- Script Finished ---")