import os
import re
import sys
import time
import csv
import math
from collections import Counter
import itertools
import warnings
import logging

# --- Basic Logging Setup ---
LOG_FILENAME = '_LEXICON_errors.log'
# Ensure logs are written relative to the script's directory or a specific path
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'), # Start fresh log for analysis run
        logging.StreamHandler()
    ]
)
logging.info(f"Logging initialized. Log file: {log_path}")


# --- Dependency Imports with Error Handling & Logging ---
try:
    from lexical_diversity import lex_div as ld
    logging.info(f"Successfully imported lex_div from lexical_diversity")
except ImportError:
    logging.critical("CRITICAL: 'lexical-diversity' library not found or lex_div submodule missing.")
    sys.exit(1)

# Removed corpus-toolkit import

try:
    import spacy
    # Check if model is available without loading it fully here to save time/memory
    if not spacy.util.is_package("en_core_web_sm"):
        raise OSError("spaCy model 'en_core_web_sm' not found.")
    # spacy.load('en_core_web_sm') # Don't load globally, load if needed? No, POS tagging needs it.
    nlp_spacy = spacy.load('en_core_web_sm') # Load once if needed for POS tagging alternative
    logging.info(f"Successfully imported and loaded spacy model 'en_core_web_sm' (spaCy version: {spacy.__version__})")
except ImportError:
     logging.critical("CRITICAL: 'spacy' library not found. Install: pip install spacy")
     sys.exit(1)
except OSError:
     logging.critical("CRITICAL: spaCy model 'en_core_web_sm' not found. Download using: python -m spacy download en_core_web_sm")
     sys.exit(1)

try:
    import nltk
    from nltk.util import ngrams
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk import word_tokenize, sent_tokenize, pos_tag
    logging.info(f"Successfully imported nltk (version: {nltk.__version__})")
    nltk_data_packages = {'punkt': 'tokenizers/punkt', 'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'}
    nltk_data_available = {}
    for pkg_id, pkg_path in nltk_data_packages.items():
        try:
            nltk.data.find(pkg_path)
            nltk_data_available[pkg_id] = True
            logging.info(f"NLTK data '{pkg_id}' found.")
        except LookupError:
            logging.warning(f"NLTK data '{pkg_id}' not found. Attempting download...")
            try:
                nltk.download(pkg_id, quiet=True)
                nltk.data.find(pkg_path)
                nltk_data_available[pkg_id] = True
                logging.info(f"NLTK data '{pkg_id}' downloaded successfully.")
            except Exception as e:
                 logging.error(f"Failed to download or verify NLTK data '{pkg_id}': {e}", exc_info=False)
                 nltk_data_available[pkg_id] = False
except ImportError:
    logging.critical("CRITICAL: 'nltk' library not found. Install: pip install nltk")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    logging.info("Successfully imported sentence_transformers")
except ImportError:
    logging.critical("CRITICAL: 'sentence-transformers' library not found. Install: pip install sentence-transformers")
    sys.exit(1)
try:
    import numpy as np
    logging.info(f"Successfully imported numpy (version: {np.__version__})")
except ImportError:
    logging.critical("CRITICAL: 'numpy' library not found. Install: pip install numpy")
    sys.exit(1)
try:
    import scipy
    from scipy.spatial.distance import cosine as cosine_distance
    logging.info(f"Successfully imported scipy (version: {scipy.__version__})")
except ImportError:
    logging.critical("CRITICAL: 'scipy' library not found. Install: pip install scipy")
    sys.exit(1)
try:
    import pandas as pd
    logging.info(f"Successfully imported pandas (version: {pd.__version__})")
except ImportError:
     logging.critical("CRITICAL: 'pandas' library not found. Install: pip install pandas")
     sys.exit(1)
try:
    import textstat
    logging.info("Successfully imported textstat")
except ImportError:
     logging.critical("CRITICAL: 'textstat' library not found. Install: pip install textstat")
     sys.exit(1)


# --- Configuration ---
NGRAM_N_VALUES = [2, 3]
MATTR_WINDOW_SIZE = 100
MSTTR_WINDOW_SIZE = 100
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_FILENAME_BASE = "_LEXICON"
FAILED_METRIC_PLACEHOLDER = None
CONTENT_POS_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
# Using Universal POS tags now for broader compatibility if switching tagger
UPOS_CONTENT_TAGS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
PENN_TO_UPOS = {
    # Nouns
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
    # Pronouns (Not content)
    'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
    # Verbs
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB', 'MD': 'AUX',
    # Adjectives
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    # Adverbs
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
    # Adpositions (Not content)
    'IN': 'ADP', 'TO': 'PART', # 'TO' can be ADP or PART, NLTK usually tags infinitive 'TO' as TO
    # Conjunctions (Not content)
    'CC': 'CCONJ', 'SC': 'SCONJ', # Using SC for subordinating conjunctions if IN isn't used
    # Determiners (Not content)
    'DT': 'DET', 'WDT': 'DET', 'PDT': 'DET', 'EX': 'DET', # EX existential there often DET
    # Particles (Not content)
    'RP': 'PART', 'POS': 'PART', # Possessive ending 's
    # Numbers (Could be content depending on analysis, excluding for now)
    'CD': 'NUM',
    # Punctuation/Other (Not content)
    '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', '(': 'PUNCT', ')': 'PUNCT', '"': 'PUNCT',
    "''": 'PUNCT', '``': 'PUNCT', '#': 'SYM', '$': 'SYM', 'SYM': 'SYM',
    'FW': 'X', 'LS': 'X', 'UH': 'INTJ', # Foreign word, List item marker, Interjection
    # Add other specific tags if needed
}


# --- Header Order ---
HEADER_ORDER = [
    # A
    'Filename', 'Total Words', 'Total Characters', 'Unique Words',
    # B
    'Average Sentence Length', 'Flesch Reading Ease',
    # C
    'TTR', 'RTTR', 'Herdan\'s C (LogTTR)',
    f'MATTR (W={MATTR_WINDOW_SIZE})', f'MSTTR (W={MSTTR_WINDOW_SIZE})',
    'VOCD', 'MTLD',
    'Yule\'s K', 'Simpson\'s D',
    'Lexical Density',
    # D
    'Distinct-2', 'Repetition-2', 'Distinct-3', 'Repetition-3'
]
header_d_part = []
for n in NGRAM_N_VALUES: header_d_part.extend([f'Distinct-{n}', f'Repetition-{n}'])
HEADER_ORDER = HEADER_ORDER[:16] + header_d_part

# --- Float Columns ---
FLOAT_COLUMNS = [
    'Average Sentence Length', 'Flesch Reading Ease',
    'TTR', 'RTTR', 'Herdan\'s C (LogTTR)',
    f'MATTR (W={MATTR_WINDOW_SIZE})', f'MSTTR (W={MSTTR_WINDOW_SIZE})',
    'VOCD', 'MTLD',
    'Yule\'s K', 'Simpson\'s D',
    'Lexical Density'
] + [f'Distinct-{n}' for n in NGRAM_N_VALUES] + [f'Repetition-{n}' for n in NGRAM_N_VALUES]


# --- Helper Functions ---
def safe_tokenize_words(text, filename=""):
    if not text: return []
    try: return word_tokenize(text)
    except Exception as e: logging.error(f"Word tokenization failed for {filename}: {e}", exc_info=True); return None

def safe_tokenize_sentences(text, filename=""):
    if not text: return []
    if not nltk_data_available.get('punkt'): logging.warning(f"Skipping sentence tokenization for {filename}: NLTK 'punkt' data unavailable."); return None
    try: return sent_tokenize(text)
    except Exception as e: logging.error(f"Sentence tokenization failed for {filename}: {e}", exc_info=True); return None

def safe_pos_tag(tokens, filename=""):
    if not tokens: return []
    if not nltk_data_available.get('averaged_perceptron_tagger'): logging.warning(f"Skipping POS tagging for {filename}: NLTK 'averaged_perceptron_tagger' data unavailable."); return None
    try: return pos_tag(tokens)
    except Exception as e: logging.error(f"POS tagging failed for {filename}: {e}", exc_info=True); return None

def calculate_ttr(tokens):
    if not tokens: return 0.0, 0, 0
    total_count = len(tokens); unique_count = len(set(tokens))
    return (unique_count / total_count) if total_count > 0 else 0.0, unique_count, total_count

def calculate_generic_metric(metric_func, tokens, *args, filename="", **kwargs):
    if not tokens: return FAILED_METRIC_PLACEHOLDER
    try:
        score = metric_func(tokens, *args, **kwargs)
        if isinstance(score, (float, np.floating)) and math.isnan(score):
             logging.debug(f"Metric '{metric_func.__name__}' resulted in NaN for {filename}. Returning 0.0.")
             return 0.0
        if isinstance(score, float) and math.isinf(score):
             logging.warning(f"Metric '{metric_func.__name__}' resulted in infinity for {filename}. Returning FAIL.")
             return FAILED_METRIC_PLACEHOLDER
        return score
    except Exception as e:
        logging.warning(f"Metric '{metric_func.__name__}' failed for {filename}: {e}")
        return FAILED_METRIC_PLACEHOLDER

def calculate_yules_k_manual(tokens, filename=""):
    if not tokens: return FAILED_METRIC_PLACEHOLDER
    N = len(tokens)
    if N < 2: return 0.0
    try:
        freq_dist = Counter(tokens)
        M2 = sum(freq * freq for freq in freq_dist.values())
        numerator = M2 - N
        denominator_ct = N * (N - 1)
        if denominator_ct == 0: return 0.0
        K = 10000 * (numerator / denominator_ct)
        return K if not math.isinf(K) else float('inf')
    except Exception as e:
        logging.warning(f"Manual Yule's K calculation failed for {filename}: {e}")
        return FAILED_METRIC_PLACEHOLDER

def calculate_simpsons_d_manual(tokens, filename=""):
    if not tokens: return FAILED_METRIC_PLACEHOLDER
    N = len(tokens)
    if N < 2: return 0.0
    try:
        freq_dist = Counter(tokens)
        numerator = sum(n * (n - 1) for n in freq_dist.values())
        denominator = N * (N - 1)
        return (numerator / denominator) if denominator > 0 else 0.0
    except Exception as e:
        logging.warning(f"Manual Simpson's D calculation failed for {filename}: {e}")
        return FAILED_METRIC_PLACEHOLDER

def calculate_ngram_stats(tokens, n, filename=""):
    if not tokens or len(tokens) < n: return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER
    try:
        str_tokens = [str(t) for t in tokens]; generated_ngrams = list(ngrams(str_tokens, n))
        if not generated_ngrams: return 0.0, 0.0
        total_ngrams = len(generated_ngrams); unique_ngrams = set(generated_ngrams)
        distinct_n = len(unique_ngrams) / total_ngrams
        ngram_counts = Counter(generated_ngrams)
        repeated_tokens_count = sum(count for count in ngram_counts.values() if count > 1)
        repetition_n = repeated_tokens_count / total_ngrams if total_ngrams > 0 else 0.0
        return repetition_n, distinct_n
    except Exception as e: logging.warning(f"N-gram stats (n={n}) failed for {filename}: {e}"); return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER

def calculate_avg_sentence_length(total_words, sentences, filename=""):
    if sentences is None or total_words is None: return FAILED_METRIC_PLACEHOLDER
    num_sentences = len(sentences); return (total_words / num_sentences) if num_sentences > 0 else 0.0

def calculate_flesch_reading_ease(text, filename=""):
    if not text: return FAILED_METRIC_PLACEHOLDER
    try: return textstat.flesch_reading_ease(str(text))
    except Exception as e: logging.warning(f"Flesch Reading Ease calculation failed for {filename}: {e}"); return FAILED_METRIC_PLACEHOLDER

def calculate_lexical_density(tagged_tokens, total_words, filename=""):
    if tagged_tokens is None or total_words is None or total_words == 0: return FAILED_METRIC_PLACEHOLDER
    try:
        content_word_count = 0
        for _, tag in tagged_tokens:
            # Convert Penn tag to UPOS tag if possible, otherwise use original tag's first letter
            upos_tag = PENN_TO_UPOS.get(tag, tag[:1].upper()) # Use first letter as fallback category
            if upos_tag in UPOS_CONTENT_TAGS:
                 content_word_count += 1
        return (content_word_count / total_words)
    except Exception as e: logging.warning(f"Lexical Density calculation failed for {filename}: {e}"); return FAILED_METRIC_PLACEHOLDER

def calculate_self_bleu(list_of_token_lists):
    if list_of_token_lists is None or len(list_of_token_lists) < 2: return FAILED_METRIC_PLACEHOLDER
    total_bleu = 0.0; pair_count = 0; chencherry = SmoothingFunction()
    for i, j in itertools.combinations(range(len(list_of_token_lists)), 2):
        candidate = [str(tok) for tok in list_of_token_lists[i]]; reference = [[str(tok) for tok in list_of_token_lists[j]]]
        if not candidate or not reference[0]: continue
        try: score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1); total_bleu += score; pair_count += 1
        except Exception as e: logging.warning(f"BLEU calculation failed for a pair: {e}"); continue
    return (total_bleu / pair_count) if pair_count > 0 else 0.0

def calculate_embedding_metrics(texts, model):
    if texts is None or len(texts) < 2: return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER
    try:
        logging.info("Calculating embeddings..."); embeddings = model.encode(texts, show_progress_bar=False)
        num_texts = len(embeddings); total_similarity = 0.0; pair_count = 0
        if num_texts < 2: return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER
        for i in range(num_texts):
            for j in range(i + 1, num_texts):
                try: similarity = 1 - cosine_distance(embeddings[i], embeddings[j]); total_similarity += similarity; pair_count += 1
                except Exception as e_pair: logging.warning(f"Cosine similarity calculation failed for a pair: {e_pair}"); continue
        avg_pairwise_similarity = (total_similarity / pair_count) if pair_count > 0 else FAILED_METRIC_PLACEHOLDER
        try:
            centroid = np.mean(embeddings, axis=0); total_distance = 0.0; valid_distances = 0
            for i in range(num_texts):
                 try:
                      dist = cosine_distance(embeddings[i], centroid)
                      if not np.isnan(dist): total_distance += dist; valid_distances += 1
                      else: logging.warning(f"NaN distance calculated for embedding {i} from centroid.")
                 except Exception as e_dist: logging.warning(f"Distance calculation failed for embedding {i}: {e_dist}"); continue
            avg_dist_to_centroid = (total_distance / valid_distances) if valid_distances > 0 else FAILED_METRIC_PLACEHOLDER
        except Exception as e_centroid: logging.error(f"Centroid calculation failed: {e_centroid}", exc_info=True); avg_dist_to_centroid = FAILED_METRIC_PLACEHOLDER
        return avg_pairwise_similarity, avg_dist_to_centroid
    except Exception as e: logging.error(f"Embedding metric calculation failed: {e}", exc_info=True); return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER

def format_duration(seconds):
    """Formats seconds into HH:MM:SS string."""
    if seconds is None: return "N/A"
    try:
        sec = int(seconds)
        hours = sec // 3600
        minutes = (sec % 3600) // 60
        secs = sec % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except Exception:
        return "N/A"

# --- Main Processing Function ---

def process_folder_comprehensive(folder_path):
    """Processes folder, calculates all metrics."""
    if not os.path.isdir(folder_path): logging.error(f"Folder not found: {folder_path}"); return None, None, None, 0
    results = []; all_raw_texts = []; all_tokenized_texts_for_bleu = []; filenames_processed = []
    logging.info(f"Scanning folder: {folder_path}")
    try: files_to_process = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))])
    except Exception as e: logging.error(f"Error listing files: {e}", exc_info=True); return None, None, None, 0
    if not files_to_process: logging.warning("No .txt files found in the folder."); return [], {}, 0, 0

    total_files_found = len(files_to_process) # Total found initially
    num_files_processed_successfully = 0 # Counter for files added to results
    start_time = time.time()
    logging.info(f"Found {total_files_found} text files. Processing...")
    model = None
    if total_files_found >= 2: # Load model only if needed
        try: logging.info(f"Loading embedding model ({EMBEDDING_MODEL_NAME})..."); model = SentenceTransformer(EMBEDDING_MODEL_NAME); logging.info("Embedding model loaded.")
        except Exception as e: logging.error(f"Failed to load embedding model: {e}. Embedding metrics will fail.", exc_info=True); model = None

    for i, filename in enumerate(files_to_process):
        file_path = os.path.join(folder_path, filename)
        logging.info(f"Processing file {i+1}/{total_files_found}: {filename}...")
        file_metrics = {hdr: FAILED_METRIC_PLACEHOLDER for hdr in HEADER_ORDER}
        file_metrics['Filename'] = filename
        file_processed_flag = False # Flag to track if this file gets added

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()

            # --- A. Identification & Basic Counts ---
            file_metrics['Total Characters'] = len(content)
            tokens_orig_case = safe_tokenize_words(content, filename)
            if tokens_orig_case is None: logging.error(f"Skipping metrics for {filename} due to word tokenization error."); results.append(file_metrics); continue
            tokens_lower = [t.lower() for t in tokens_orig_case]
            total_words = len(tokens_lower); unique_words = len(set(tokens_lower))
            file_metrics['Total Words'] = total_words; file_metrics['Unique Words'] = unique_words

            # --- B. Readability & Sentence Structure ---
            sentences = safe_tokenize_sentences(content, filename)
            file_metrics['Average Sentence Length'] = calculate_avg_sentence_length(total_words, sentences, filename)
            file_metrics['Flesch Reading Ease'] = calculate_flesch_reading_ease(content, filename)

            # --- C. Lexical Diversity (Word Level) ---
            file_metrics['TTR'] = (unique_words / total_words) if total_words > 0 else 0.0
            file_metrics['RTTR'] = calculate_generic_metric(ld.root_ttr, tokens_lower, filename=filename)
            file_metrics['Herdan\'s C (LogTTR)'] = calculate_generic_metric(ld.log_ttr, tokens_lower, filename=filename)
            file_metrics[f'MATTR (W={MATTR_WINDOW_SIZE})'] = calculate_generic_metric(ld.mattr, tokens_lower, MATTR_WINDOW_SIZE, filename=filename)
            file_metrics[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = calculate_generic_metric(ld.msttr, tokens_lower, MSTTR_WINDOW_SIZE, filename=filename)
            file_metrics['VOCD'] = calculate_generic_metric(ld.hdd, tokens_lower, filename=filename)
            file_metrics['MTLD'] = calculate_generic_metric(ld.mtld, tokens_lower, filename=filename)
            file_metrics['Yule\'s K'] = calculate_yules_k_manual(tokens_lower, filename=filename)
            file_metrics['Simpson\'s D'] = calculate_simpsons_d_manual(tokens_lower, filename=filename)
            tagged_tokens = safe_pos_tag(tokens_orig_case, filename)
            file_metrics['Lexical Density'] = calculate_lexical_density(tagged_tokens, total_words, filename)

            # --- D. N-gram Level Diversity & Repetition (Phrase Level) ---
            for n in NGRAM_N_VALUES:
                rep_n, dist_n = calculate_ngram_stats(tokens_lower, n, filename=filename)
                file_metrics[f'Distinct-{n}'] = dist_n
                file_metrics[f'Repetition-{n}'] = rep_n

            # Store data for collective metrics
            all_raw_texts.append(content)
            all_tokenized_texts_for_bleu.append(tokens_lower)
            filenames_processed.append(filename)
            results.append(file_metrics)
            file_processed_flag = True # Mark as successfully processed

        except Exception as e:
            logging.error(f"Unexpected error processing file {filename}: {e}", exc_info=True)
            # Append dict with placeholders only if not already added due to earlier error
            if not file_processed_flag:
                 results.append(file_metrics)

    # --- E. Calculate Collective Metrics ---
    num_files_processed_successfully = len(filenames_processed) # Count files that made it to the end of try block
    collective_metrics = {}
    if num_files_processed_successfully >= 2:
        logging.info(f"Calculating collective metrics for {num_files_processed_successfully} successfully processed files...")
        logging.info("Calculating Self-BLEU..."); start_bleu = time.time()
        collective_metrics['Self-BLEU'] = calculate_self_bleu(all_tokenized_texts_for_bleu)
        logging.info(f"Self-BLEU calculation took {time.time() - start_bleu:.2f}s")
        if model:
             start_embed = time.time()
             avg_similarity, avg_dist_centroid = calculate_embedding_metrics(all_raw_texts, model)
             collective_metrics['Avg Pairwise Cosine Similarity'] = avg_similarity
             collective_metrics['Avg Distance to Centroid'] = avg_dist_centroid
             logging.info(f"Embedding metric calculation took {time.time() - start_embed:.2f}s")
        else:
             logging.warning("Skipping embedding metrics as model failed to load.")
             collective_metrics['Avg Pairwise Cosine Similarity'] = FAILED_METRIC_PLACEHOLDER
             collective_metrics['Avg Distance to Centroid'] = FAILED_METRIC_PLACEHOLDER
    else:
        logging.warning("Skipping collective metrics (need at least 2 successfully processed files).")
        collective_metrics['Self-BLEU'] = 'N/A (<=1 file)'
        collective_metrics['Avg Pairwise Cosine Similarity'] = 'N/A (<=1 file)'
        collective_metrics['Avg Distance to Centroid'] = 'N/A (<=1 file)'

    end_time = time.time()
    analysis_duration = end_time - start_time
    logging.info(f"Finished processing in {analysis_duration:.2f} seconds.")
    # Return the count of files successfully processed
    return results, collective_metrics, analysis_duration, num_files_processed_successfully


# --- Saving Functions ---

def format_value(value, is_float=False):
    """Formats values for output, converting None to 'FAIL'."""
    if value is None: return "FAIL"
    if is_float:
        try:
            num_val = float(value)
            if math.isinf(num_val): return "INF"
            if math.isnan(num_val): return "NaN"
            return f"{num_val:.4f}"
        except (ValueError, TypeError): return str(value)
    else: return str(value)

def save_results_csv(results, collective_metrics, output_folder):
    """Saves results to a CSV file."""
    output_path = os.path.join(output_folder, f"{OUTPUT_FILENAME_BASE}.csv")
    logging.info(f"Saving CSV results to: {output_path}")
    if not results: logging.warning("No results to save."); return
    try:
        data_for_df = []
        for res_dict in results:
            formatted_row = {}
            # Handle potential key renaming
            if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict: res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
            if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict: res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

            for header in HEADER_ORDER:
                is_float = header in FLOAT_COLUMNS
                formatted_row[header] = format_value(res_dict.get(header, FAILED_METRIC_PLACEHOLDER), is_float)
            data_for_df.append(formatted_row)
        df = pd.DataFrame(data_for_df, columns=HEADER_ORDER)
        df.to_csv(output_path, index=False, encoding='utf-8')
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow([]); writer.writerow(["--- Collective Metrics (E) ---"])
            for key, value in collective_metrics.items():
                is_coll_float = isinstance(value, (float, np.floating))
                writer.writerow([key, format_value(value, is_coll_float)])
        logging.info("CSV Results saved successfully.")
    except Exception as e: logging.error(f"Error writing CSV output file '{output_path}': {e}", exc_info=True)


def save_results_txt(results, collective_metrics, output_folder):
    """Saves results to a formatted TXT file."""
    output_path = os.path.join(output_folder, f"{OUTPUT_FILENAME_BASE}.txt")
    logging.info(f"Saving TXT results to: {output_path}")
    if not results: logging.warning("No results to save."); return

    # Define fixed widths (Adjusted for new metrics)
    widths = {
        'Filename': 30, 'Total Words': 12, 'Total Characters': 15, 'Unique Words': 12, # A
        'Average Sentence Length': 15, 'Flesch Reading Ease': 15, # B
        'TTR': 10, 'RTTR': 10, 'Herdan\'s C (LogTTR)': 18,
        f'MATTR (W={MATTR_WINDOW_SIZE})': 15, f'MSTTR (W={MSTTR_WINDOW_SIZE})': 15,
        'VOCD': 10, 'MTLD': 10,
        'Yule\'s K': 10, 'Simpson\'s D': 12,
        'Lexical Density': 15, # C
        'Distinct-2': 10, 'Repetition-2': 12, 'Distinct-3': 10, 'Repetition-3': 12 # D
    }
    for h in HEADER_ORDER:
        if h not in widths: widths[h] = 15 # Default

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # --- Write Header ---
            header_line = ""; separator_line = ""
            section_headers = {'A': 'A. Identification & Counts', 'B': 'B. Readability & Sentence', 'C': 'C. Lexical Diversity (Word)', 'D': 'D. N-gram Stats'}
            section_map = {i: 'A' for i in range(4)}
            section_map.update({i: 'B' for i in range(4, 6)})
            section_map.update({i: 'C' for i in range(6, 16)}) # Section C now up to index 15
            section_map.update({i: 'D' for i in range(16, len(HEADER_ORDER))})

            col_widths_in_section = {}
            for i, h in enumerate(HEADER_ORDER):
                 section = section_map.get(i)
                 col_widths_in_section.setdefault(section, 0)
                 is_last_in_section = (i == len(HEADER_ORDER) - 1) or (section_map.get(i+1) != section)
                 col_widths_in_section[section] += widths[h] + (0 if is_last_in_section else 1)

            section_title_line = f"{section_headers['A']:<{col_widths_in_section['A']}} | "
            section_title_line += f"{section_headers['B']:<{col_widths_in_section['B']}} | "
            section_title_line += f"{section_headers['C']:<{col_widths_in_section['C']}} | "
            section_title_line += f"{section_headers['D']:<{col_widths_in_section['D']}}"
            f.write(section_title_line + "\n")

            for i, h in enumerate(HEADER_ORDER):
                section = section_map.get(i); next_section = section_map.get(i+1); width = widths[h]
                header_line += f"{h:<{width}}"; separator_line += "-" * width
                if i < len(HEADER_ORDER) - 1:
                    if section == next_section: header_line += " "; separator_line += "-"
                    else: header_line += " | "; separator_line += "---"
            f.write(header_line + "\n"); f.write(separator_line + "\n")

            # --- Write Data Rows ---
            for res_dict in results:
                # Handle potential key renaming
                if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict: res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
                if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict: res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

                row_line = ""
                for i, h in enumerate(HEADER_ORDER):
                    section = section_map.get(i); next_section = section_map.get(i+1); width = widths[h]
                    is_float = h in FLOAT_COLUMNS
                    value_str = format_value(res_dict.get(h, FAILED_METRIC_PLACEHOLDER), is_float)
                    align = ">" if h not in ['Filename'] else "<"
                    formatted_val = f"{value_str:{align}{width}}"
                    if len(formatted_val) > width: formatted_val = formatted_val[:width-1] + "."
                    row_line += formatted_val
                    if i < len(HEADER_ORDER) - 1:
                        if section == next_section: row_line += " "
                        else: row_line += " | "
                f.write(row_line + "\n")

            # --- Write Collective Metrics ---
            f.write("\n" + "="*60 + "\n"); f.write("E. Collective Metrics (Calculated across all processed files)\n"); f.write("="*60 + "\n")
            for key, value in collective_metrics.items():
                is_coll_float = isinstance(value, (float, np.floating))
                value_str = format_value(value, is_coll_float)
                f.write(f"{key:<40}: {value_str}\n")

        logging.info("TXT Results saved successfully.")
    except Exception as e:
        logging.error(f"Error writing TXT output file '{output_path}': {e}", exc_info=True)

# --- UPDATED HTML Saving Function (in lexicon_analyzer.py) ---
def save_results_html(results, collective_metrics, output_folder, analysis_duration, num_files_processed):
    """Saves results to an HTML file with a sortable table structure."""
    output_path = os.path.join(output_folder, f"{OUTPUT_FILENAME_BASE}.html")
    logging.info(f"Saving initial HTML results to: {output_path}")
    if not results: logging.warning("No results to save."); return

    try:
        # Prepare data for DataFrame, keeping numeric types where possible for sorting
        data_for_df = []
        for res_dict in results:
            row_data = {}
            # Handle potential key renaming
            if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict:
                 res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
            if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict:
                 res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

            for header in HEADER_ORDER:
                raw_value = res_dict.get(header, FAILED_METRIC_PLACEHOLDER)
                if raw_value is None:
                    row_data[header] = np.nan # Use NaN for sorting
                else:
                    # Attempt conversion for numeric columns, keep others as string
                    if header in FLOAT_COLUMNS or header in ['Total Words', 'Total Characters', 'Unique Words']:
                         try: row_data[header] = pd.to_numeric(raw_value)
                         except (ValueError, TypeError): row_data[header] = str(raw_value) # Keep as string if conversion fails
                    else: row_data[header] = str(raw_value) # Keep Filename as string
            data_for_df.append(row_data)

        df = pd.DataFrame(data_for_df, columns=HEADER_ORDER)

        # Generate HTML table with specific attributes for sorting
        html_table = df.to_html(
            index=False,
            border=1,
            classes='dataframe sortable results_table', # Add 'sortable' class
            table_id='resultsTable', # Add ID for JS targeting
            justify='center',
            na_rep='FAIL', # How NaNs (originally FAIL) are displayed
            float_format='{:.4f}'.format # Format floats in the HTML output
        )

        # Prepare collective metrics HTML part
        collective_html = "<h2>E. Collective Metrics (Calculated across all processed files)</h2>\n<ul>\n"
        for key, value in collective_metrics.items():
            is_coll_float = isinstance(value, (float, np.floating))
            value_str = format_value(value, is_coll_float) # Use existing format helper
            collective_html += f"  <li><b>{key}:</b> {value_str}</li>\n"
        collective_html += "</ul>"

        # Format duration
        formatted_duration = format_duration(analysis_duration)

        # Basic HTML structure (JavaScript/CSS will be added by visualizer script)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lexical Analysis Results</title>
    <!-- CSS and JS for sorting will be added by lexicon_visualizer.py -->
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.5; }}
        h1, h2, h3 {{ color: #333; margin-top: 1.5em; }}
        p {{ margin-bottom: 0.5em; }}
        table.dataframe {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }}
        table.dataframe th, table.dataframe td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
        table.dataframe th {{ background-color: #f0f0f0; font-weight: bold; text-align: center; }}
        table.dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
        table.dataframe tbody tr:hover {{ background-color: #e5e5e5; }}
        table.dataframe td:first-child {{ text-align: left; }} /* Left-align filename */
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 5px; }}
        b {{ color: #555; }}
        /* Basic sortable styles (will be enhanced by visualizer) */
        th.sortable-header {{ cursor: pointer; position: relative; padding-right: 20px !important; }} /* Add padding for arrows */
        th.sortable-header::after, th.sortable-header::before {{
            content: ""; position: absolute; right: 8px; opacity: 0.2; border: 5px solid transparent;
        }}
        th.sortable-header::before {{ border-bottom-color: #666; bottom: 55%; }} /* Up arrow */
        th.sortable-header::after {{ border-top-color: #666; top: 55%; }} /* Down arrow */
        th.sort-asc::before {{ opacity: 1; }}
        th.sort-desc::after {{ opacity: 1; }}
        /* Style for appended sections */
        #metric_descriptions dl {{ margin-left: 20px; }}
        #metric_descriptions dt {{ font-weight: bold; margin-top: 10px; color: #444; }}
        #metric_descriptions dd {{ margin-left: 20px; margin-bottom: 8px; font-size: 0.95em; }}
        #generated_plots h3 {{ border-top: 1px solid #eee; padding-top: 15px; margin-top: 30px; }}
        #generated_plots p {{ margin-left: 10px; font-size: 0.95em; }}
        #generated_plots img {{ max-width: 95%; height: auto; border: 1px solid #ccc; margin-top: 10px; display: block; margin-left: auto; margin-right: auto; margin-bottom: 20px; background-color: white; padding: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}}
    </style>
</head>
<body>
    <h1>Lexical Analysis Results</h1>
    <p><b>Target Folder:</b> {output_folder}</p>
    <p><b>Analysis Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><b>Total Files Analyzed:</b> {num_files_processed}</p> <!-- ADDED FILE COUNT HERE -->
    <p><b>Total Time Spent Analyzing:</b> {formatted_duration}</p> <!-- ADDED DURATION HERE -->

    <h2>Individual File Metrics</h2>
    {html_table}

    {collective_html}

    <!-- Descriptions, Plots and JS will be added by lexicon_visualizer.py -->
</body>
</html>
"""
        # Write initial HTML content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info("Initial HTML Results saved successfully.")
    except Exception as e:
        logging.error(f"Error writing initial HTML output file '{output_path}': {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='nltk.translate.bleu_score')
    warnings.filterwarnings("ignore", message="The sentence constituent `.+` does not exist in the dictionary")

    target_folder = input("Enter the path to the folder containing text files: ")

    if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
         logging.critical(f"CRITICAL: Invalid folder path '{target_folder}'")
         sys.exit(1)

    logging.info(f"\n{'='*20} Starting New Analysis Run: {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}")
    logging.info(f"Target Folder: {target_folder}")

    # --- UPDATED to receive num_files_processed ---
    individual_results, collective_results, analysis_duration, num_files_processed = process_folder_comprehensive(target_folder)

    if individual_results is not None:
        save_results_csv(individual_results, collective_results, target_folder)
        save_results_txt(individual_results, collective_results, target_folder)
        # --- UPDATED to pass num_files_processed ---
        save_results_html(individual_results, collective_results, target_folder, analysis_duration, num_files_processed)
    else:
        logging.error("Processing failed at a high level. No results saved.")
        sys.exit(1)

    logging.info("Script finished.")