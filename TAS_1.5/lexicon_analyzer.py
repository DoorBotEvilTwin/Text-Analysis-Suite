# lexicon_analyzer.py
# MODIFIED: Reads source .txt files from a 'txt' subdirectory
# MODIFIED: Input prompt updated to reflect subdirectory requirement
# MODIFIED: Added 5-second countdown timer for folder input
# MODIFIED: Added new metrics (Readability, Sentiment, Emotion, NMF Topics, Keywords)
# PATCHED: Fixed CSV separator string to match visualizer/processor expectation.
# PATCHED: Modified save_results_html to split output into two tables.
# PATCHED: Added more specific error logging in NRCLex/Keyword functions.
# PATCHED: Moved VADER/TextBlob Sentiment columns to the secondary HTML table.

import os
import re
import sys
import time # Added for sleep
import csv
import math
from collections import Counter
import itertools
import warnings
import logging
import threading # Added for timed input
import json # For NRCLex JSON output

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
# (Imports remain the same as the previous version)
try:
    from lexical_diversity import lex_div as ld
    logging.info(f"Successfully imported lex_div from lexical_diversity")
except ImportError:
    logging.critical("CRITICAL: 'lexical-diversity' library not found or lex_div submodule missing. Install: pip install lexical-diversity")
    sys.exit(1)

try:
    import spacy
    if not spacy.util.is_package("en_core_web_sm"):
        raise OSError("spaCy model 'en_core_web_sm' not found.")
    nlp_spacy = spacy.load('en_core_web_sm')
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
    nltk_data_packages = {'punkt': 'tokenizers/punkt', 'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger', 'stopwords': 'corpora/stopwords'} # Added stopwords for NMF/TFIDF
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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    logging.info("Successfully imported and initialized VADER")
except ImportError:
     logging.critical("CRITICAL: 'vaderSentiment' library not found. Install: pip install vaderSentiment")
     sys.exit(1)

try:
    from textblob import TextBlob
    logging.info("Successfully imported TextBlob")
except ImportError:
     logging.critical("CRITICAL: 'textblob' library not found. Install: pip install textblob")
     logging.warning("TextBlob requires NLTK corpora. Attempting download...")
     try:
         nltk.download('brown', quiet=True)
         nltk.download('punkt', quiet=True) # Already checked, but ensure it's here
         from textblob import TextBlob # Try importing again
         logging.info("Downloaded TextBlob corpora and imported successfully.")
     except Exception as e:
         logging.critical(f"Failed to download TextBlob corpora or import TextBlob: {e}")
         sys.exit(1)

try:
    from nrclex import NRCLex
    logging.info("Successfully imported NRCLex")
except ImportError:
     logging.critical("CRITICAL: 'NRCLex' library not found. Install: pip install NRCLex")
     sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    logging.info("Successfully imported sklearn components (TfidfVectorizer, NMF)")
except ImportError:
     logging.critical("CRITICAL: 'scikit-learn' library not found. Install: pip install scikit-learn")
     sys.exit(1)

try:
    import yake
    logging.info("Successfully imported YAKE!")
except ImportError:
     logging.critical("CRITICAL: 'yake' library not found. Install: pip install yake")
     sys.exit(1)


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


# --- Configuration ---
NGRAM_N_VALUES = [2, 3]
MATTR_WINDOW_SIZE = 100
MSTTR_WINDOW_SIZE = 100
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_FILENAME_BASE = "_LEXICON"
FAILED_METRIC_PLACEHOLDER = None
CONTENT_POS_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
UPOS_CONTENT_TAGS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
PENN_TO_UPOS = {
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN', 'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB', 'MD': 'AUX',
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ', 'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
    'IN': 'ADP', 'TO': 'PART', 'CC': 'CCONJ', 'SC': 'SCONJ', 'DT': 'DET', 'WDT': 'DET', 'PDT': 'DET', 'EX': 'DET',
    'RP': 'PART', 'POS': 'PART', 'CD': 'NUM', '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', '(': 'PUNCT', ')': 'PUNCT', '"': 'PUNCT',
    "''": 'PUNCT', '``': 'PUNCT', '#': 'SYM', '$': 'SYM', 'SYM': 'SYM', 'FW': 'X', 'LS': 'X', 'UH': 'INTJ',
}
# --- NEW Configuration ---
N_TOPICS = 5 # Number of topics for NMF
N_KEYWORDS = 5 # Number of keywords to extract

# --- Header Order (Updated) ---
HEADER_ORDER = [
    # Basic Info
    'Filename', 'Total Words', 'Total Characters', 'Unique Words', 'Average Sentence Length',
    # Readability
    'Flesch Reading Ease', 'Gunning Fog', 'SMOG Index', 'Dale-Chall Score',
    # Lexical Diversity
    'TTR', 'RTTR', "Herdan's C (LogTTR)", f'MATTR (W={MATTR_WINDOW_SIZE})', f'MSTTR (W={MSTTR_WINDOW_SIZE})',
    'VOCD', 'MTLD', "Yule's K", "Simpson's D",
    # Syntactic/Phrase Diversity
    'Lexical Density', 'Distinct-2', 'Repetition-2', 'Distinct-3', 'Repetition-3',
    # Sentiment/Emotion
    'Sentiment (VADER Comp)', 'Sentiment (VADER Pos)', 'Sentiment (VADER Neu)', 'Sentiment (VADER Neg)',
    'Sentiment (TextBlob Pol)', 'Sentiment (TextBlob Subj)',
    'Emotion (NRCLex Dominant)', 'Emotion (NRCLex Scores JSON)',
    # Topic Modeling
    'Topic (NMF ID)', 'Topic (NMF Prob)',
    # Keyword Extraction
    'Keywords (TF-IDF Top 5)', 'Keywords (YAKE! Top 5)',
    # Collective Metrics (Added Separately)
    # 'Self-BLEU', 'Avg Pairwise Cosine Similarity', 'Avg Distance to Centroid'
]

# --- Float Columns (Updated) ---
FLOAT_COLUMNS = [
    'Average Sentence Length',
    # Readability
    'Flesch Reading Ease', 'Gunning Fog', 'SMOG Index', 'Dale-Chall Score',
    # Diversity
    'TTR', 'RTTR', "Herdan's C (LogTTR)", f'MATTR (W={MATTR_WINDOW_SIZE})', f'MSTTR (W={MSTTR_WINDOW_SIZE})',
    'VOCD', 'MTLD', "Yule's K", "Simpson's D",
    # Syntactic/Phrase
    'Lexical Density', 'Distinct-2', 'Repetition-2', 'Distinct-3', 'Repetition-3',
    # Sentiment/Emotion
    'Sentiment (VADER Comp)', 'Sentiment (VADER Pos)', 'Sentiment (VADER Neu)', 'Sentiment (VADER Neg)',
    'Sentiment (TextBlob Pol)', 'Sentiment (TextBlob Subj)',
    # Topics
    'Topic (NMF Prob)',
    # Collective (Added Separately)
    # 'Self-BLEU', 'Avg Pairwise Cosine Similarity', 'Avg Distance to Centroid'
]

# --- Columns for the Secondary HTML Table (Updated) ---
SECONDARY_HTML_COLUMNS = [
    'Filename', # Keep filename for reference
    # Sentiment
    'Sentiment (VADER Comp)', 'Sentiment (VADER Pos)', 'Sentiment (VADER Neu)', 'Sentiment (VADER Neg)',
    'Sentiment (TextBlob Pol)', 'Sentiment (TextBlob Subj)',
    # Emotion
    'Emotion (NRCLex Dominant)',
    'Emotion (NRCLex Scores JSON)',
    # Keywords
    'Keywords (TF-IDF Top 5)',
    'Keywords (YAKE! Top 5)'
]


# --- Helper Functions (Existing + New) ---
# (Helper functions remain the same as the previous version)
def safe_tokenize_words(text, filename=""):
    if not text: return []
    try: return word_tokenize(text)
    except Exception as e: logging.error(f"Word tokenization failed for {filename}: {e}", exc_info=False); return None # Less verbose log

def safe_tokenize_sentences(text, filename=""):
    if not text: return []
    if not nltk_data_available.get('punkt'): logging.warning(f"Skipping sentence tokenization for {filename}: NLTK 'punkt' data unavailable."); return None
    try: return sent_tokenize(text)
    except Exception as e: logging.error(f"Sentence tokenization failed for {filename}: {e}", exc_info=False); return None

def safe_pos_tag(tokens, filename=""):
    if not tokens: return []
    if not nltk_data_available.get('averaged_perceptron_tagger'): logging.warning(f"Skipping POS tagging for {filename}: NLTK 'averaged_perceptron_tagger' data unavailable."); return None
    try: return pos_tag(tokens)
    except Exception as e: logging.error(f"POS tagging failed for {filename}: {e}", exc_info=False); return None

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
        # Log specific metric failure with filename
        logging.warning(f"Metric '{metric_func.__name__}' failed for {filename}: {e}", exc_info=False)
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
        logging.warning(f"Manual Yule's K calculation failed for {filename}: {e}", exc_info=False)
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
        logging.warning(f"Manual Simpson's D calculation failed for {filename}: {e}", exc_info=False)
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
    except Exception as e: logging.warning(f"N-gram stats (n={n}) failed for {filename}: {e}", exc_info=False); return FAILED_METRIC_PLACEHOLDER, FAILED_METRIC_PLACEHOLDER

def calculate_avg_sentence_length(total_words, sentences, filename=""):
    if sentences is None or total_words is None: return FAILED_METRIC_PLACEHOLDER
    num_sentences = len(sentences); return (total_words / num_sentences) if num_sentences > 0 else 0.0

def calculate_lexical_density(tagged_tokens, total_words, filename=""):
    if tagged_tokens is None or total_words is None or total_words == 0: return FAILED_METRIC_PLACEHOLDER
    try:
        content_word_count = 0
        for _, tag in tagged_tokens:
            upos_tag = PENN_TO_UPOS.get(tag, tag[:1].upper())
            if upos_tag in UPOS_CONTENT_TAGS:
                 content_word_count += 1
        return (content_word_count / total_words)
    except Exception as e: logging.warning(f"Lexical Density calculation failed for {filename}: {e}", exc_info=False); return FAILED_METRIC_PLACEHOLDER

def calculate_self_bleu(list_of_token_lists):
    if list_of_token_lists is None or len(list_of_token_lists) < 2: return FAILED_METRIC_PLACEHOLDER
    total_bleu = 0.0; pair_count = 0; chencherry = SmoothingFunction()
    for i, j in itertools.combinations(range(len(list_of_token_lists)), 2):
        candidate = [str(tok) for tok in list_of_token_lists[i]]; reference = [[str(tok) for tok in list_of_token_lists[j]]]
        if not candidate or not reference[0]: continue
        try: score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1); total_bleu += score; pair_count += 1
        except Exception as e: logging.warning(f"BLEU calculation failed for a pair: {e}", exc_info=False); continue
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
                except Exception as e_pair: logging.warning(f"Cosine similarity calculation failed for a pair: {e_pair}", exc_info=False); continue
        avg_pairwise_similarity = (total_similarity / pair_count) if pair_count > 0 else FAILED_METRIC_PLACEHOLDER
        try:
            centroid = np.mean(embeddings, axis=0); total_distance = 0.0; valid_distances = 0
            for i in range(num_texts):
                 try:
                      dist = cosine_distance(embeddings[i], centroid)
                      if not np.isnan(dist): total_distance += dist; valid_distances += 1
                      else: logging.warning(f"NaN distance calculated for embedding {i} from centroid.")
                 except Exception as e_dist: logging.warning(f"Distance calculation failed for embedding {i}: {e_dist}", exc_info=False); continue
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

# --- NEW Helper Functions for New Metrics ---

def calculate_textstat_metrics(text, filename=""):
    """Calculates multiple readability scores using textstat."""
    scores = {
        'Flesch Reading Ease': FAILED_METRIC_PLACEHOLDER,
        'Gunning Fog': FAILED_METRIC_PLACEHOLDER,
        'SMOG Index': FAILED_METRIC_PLACEHOLDER,
        'Dale-Chall Score': FAILED_METRIC_PLACEHOLDER,
    }
    if not text: return scores
    try: scores['Flesch Reading Ease'] = textstat.flesch_reading_ease(text)
    except Exception as e: logging.warning(f"Textstat Flesch Reading Ease failed for {filename}: {e}", exc_info=False)
    try: scores['Gunning Fog'] = textstat.gunning_fog(text)
    except Exception as e: logging.warning(f"Textstat Gunning Fog failed for {filename}: {e}", exc_info=False)
    try: scores['SMOG Index'] = textstat.smog_index(text)
    except Exception as e: logging.warning(f"Textstat SMOG Index failed for {filename}: {e}", exc_info=False)
    try: scores['Dale-Chall Score'] = textstat.dale_chall_readability_score(text)
    except Exception as e: logging.warning(f"Textstat Dale-Chall failed for {filename}: {e}", exc_info=False)
    return scores

def calculate_vader_sentiment(text, filename=""):
    """Calculates VADER sentiment scores."""
    if not text: return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    try:
        vs = vader_analyzer.polarity_scores(text)
        return {
            'Sentiment (VADER Comp)': vs.get('compound', FAILED_METRIC_PLACEHOLDER),
            'Sentiment (VADER Pos)': vs.get('pos', FAILED_METRIC_PLACEHOLDER),
            'Sentiment (VADER Neu)': vs.get('neu', FAILED_METRIC_PLACEHOLDER),
            'Sentiment (VADER Neg)': vs.get('neg', FAILED_METRIC_PLACEHOLDER),
        }
    except Exception as e:
        logging.warning(f"VADER sentiment calculation failed for {filename}: {e}", exc_info=False)
        return {
            'Sentiment (VADER Comp)': FAILED_METRIC_PLACEHOLDER,
            'Sentiment (VADER Pos)': FAILED_METRIC_PLACEHOLDER,
            'Sentiment (VADER Neu)': FAILED_METRIC_PLACEHOLDER,
            'Sentiment (VADER Neg)': FAILED_METRIC_PLACEHOLDER,
        }

def calculate_textblob_sentiment(text, filename=""):
    """Calculates TextBlob sentiment scores."""
    if not text: return {'polarity': 0.0, 'subjectivity': 0.0}
    try:
        blob = TextBlob(text)
        return {
            'Sentiment (TextBlob Pol)': blob.sentiment.polarity,
            'Sentiment (TextBlob Subj)': blob.sentiment.subjectivity,
        }
    except Exception as e:
        logging.warning(f"TextBlob sentiment calculation failed for {filename}: {e}", exc_info=False)
        return {
            'Sentiment (TextBlob Pol)': FAILED_METRIC_PLACEHOLDER,
            'Sentiment (TextBlob Subj)': FAILED_METRIC_PLACEHOLDER,
        }

# --- PATCHED: Added specific error logging ---
def calculate_nrclex_emotions(text, filename=""):
    """Calculates NRCLex emotions."""
    if not text: return {'dominant': 'neutral', 'scores_json': '{}'}
    try:
        nrc = NRCLex(text)
        top_emotion = nrc.top_emotions[0][0] if nrc.top_emotions else 'neutral'
        # Convert raw scores to JSON string for single column storage
        scores_json = json.dumps(nrc.raw_emotion_scores)
        return {
            'Emotion (NRCLex Dominant)': top_emotion,
            'Emotion (NRCLex Scores JSON)': scores_json,
        }
    except Exception as e:
        # Log the specific error encountered by NRCLex
        logging.error(f"NRCLex emotion calculation failed for {filename}: {type(e).__name__} - {e}", exc_info=False)
        return {
            'Emotion (NRCLex Dominant)': FAILED_METRIC_PLACEHOLDER,
            'Emotion (NRCLex Scores JSON)': FAILED_METRIC_PLACEHOLDER,
        }
# --- END PATCH ---

def calculate_nmf_topics(texts, n_topics=N_TOPICS):
    """Performs NMF topic modeling on a list of texts."""
    if not texts or len(texts) < n_topics:
        logging.warning(f"Not enough documents ({len(texts)}) for NMF with {n_topics} topics. Skipping NMF.")
        return None, None, None, None, None # Return None for assignments, probs, vectorizer, model, matrix

    logging.info(f"Performing NMF Topic Modeling with {n_topics} topics...")
    try:
        # Use NLTK stopwords if available
        stop_words = None
        if nltk_data_available.get('stopwords'):
            from nltk.corpus import stopwords
            stop_words = list(stopwords.words('english'))
            logging.info("Using NLTK English stopwords for TF-IDF.")
        else:
            logging.warning("NLTK stopwords not available. TF-IDF/NMF might be less effective.")
            stop_words = 'english' # Use sklearn's built-in list as fallback

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(texts)
        logging.info(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")

        # NMF Model
        nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=300, init='nndsvda') # Added init for stability
        W = nmf_model.fit_transform(tfidf_matrix) # Document-topic matrix
        H = nmf_model.components_ # Topic-term matrix

        # Get dominant topic and probability for each document
        dominant_topic = np.argmax(W, axis=1)
        topic_probability = np.max(W, axis=1)

        logging.info("NMF Topic Modeling completed.")
        # Return assignments, probabilities, and the fitted vectorizer/model for keyword extraction
        return dominant_topic, topic_probability, vectorizer, nmf_model, tfidf_matrix

    except ValueError as ve:
        logging.error(f"ValueError during NMF (check vectorizer/NMF parameters or input data): {ve}", exc_info=True)
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"NMF Topic Modeling failed: {e}", exc_info=True)
        return None, None, None, None, None

# --- PATCHED: Added specific error logging ---
def extract_tfidf_keywords(tfidf_matrix, vectorizer, n_keywords=N_KEYWORDS):
    """Extracts top TF-IDF keywords for each document."""
    keywords_list = []
    if tfidf_matrix is None or vectorizer is None:
        logging.warning("TF-IDF matrix or vectorizer not available. Skipping TF-IDF keywords.")
        # Determine expected number of docs if possible
        num_docs_expected = tfidf_matrix.shape[0] if hasattr(tfidf_matrix, 'shape') else 0
        return [FAILED_METRIC_PLACEHOLDER] * num_docs_expected

    logging.info(f"Extracting Top {n_keywords} TF-IDF Keywords...")
    num_docs = tfidf_matrix.shape[0]
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        for i in range(num_docs):
            try:
                # Get the row corresponding to the document
                row = tfidf_matrix[i].toarray().flatten()
                # Get indices of top N scores
                top_indices = row.argsort()[-n_keywords:][::-1]
                # Get corresponding feature names (keywords)
                top_keywords = feature_names[top_indices]
                keywords_list.append(";".join(top_keywords)) # Join with semicolon
            except IndexError as ie:
                logging.error(f"IndexError extracting TF-IDF keywords for document index {i}: {ie}. Assigning FAIL.", exc_info=False)
                keywords_list.append(FAILED_METRIC_PLACEHOLDER)
            except Exception as doc_e:
                 logging.error(f"Error extracting TF-IDF keywords for document index {i}: {doc_e}. Assigning FAIL.", exc_info=False)
                 keywords_list.append(FAILED_METRIC_PLACEHOLDER)

        logging.info("TF-IDF Keyword Extraction completed.")
        return keywords_list
    except Exception as e:
        # Log the general error if it happens outside the loop
        logging.error(f"TF-IDF Keyword Extraction failed: {type(e).__name__} - {e}", exc_info=False)
        return [FAILED_METRIC_PLACEHOLDER] * num_docs
# --- END PATCH ---

# --- PATCHED: Added specific error logging ---
def extract_yake_keywords(text, n_keywords=N_KEYWORDS, filename=""):
    """Extracts keywords using YAKE!"""
    if not text: return FAILED_METRIC_PLACEHOLDER
    try:
        # Configure YAKE! (language, max ngram size, deduplication threshold, num keywords)
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=n_keywords, features=None)
        keywords = kw_extractor.extract_keywords(text)
        # Extract just the keyword strings
        keyword_strings = [kw[0] for kw in keywords]
        return ";".join(keyword_strings) # Join with semicolon
    except Exception as e:
        # Log the specific error encountered by YAKE!
        logging.error(f"YAKE! Keyword Extraction failed for {filename}: {type(e).__name__} - {e}", exc_info=False)
        return FAILED_METRIC_PLACEHOLDER
# --- END PATCH ---


# --- Main Processing Function (Updated) ---

def process_folder_comprehensive(folder_path):
    """Processes folder, calculates all metrics."""
    if not os.path.isdir(folder_path): logging.error(f"Folder not found: {folder_path}"); return None, None, None, 0
    results = []; all_raw_texts = []; all_tokenized_texts_for_bleu = []; filenames_processed = []

    text_subdir_name = "txt"
    text_folder_path = os.path.join(folder_path, text_subdir_name)
    if not os.path.isdir(text_folder_path):
        logging.error(f"Required subdirectory '{text_subdir_name}' not found inside: {folder_path}")
        logging.error("Please ensure your .txt files are placed inside a 'txt' folder within the target directory.")
        return None, None, None, 0
    logging.info(f"Scanning for .txt files in: {text_folder_path}")

    try:
        files_to_process = sorted([
            f for f in os.listdir(text_folder_path)
            if f.lower().endswith(".txt") and os.path.isfile(os.path.join(text_folder_path, f))
        ])
    except Exception as e: logging.error(f"Error listing files in '{text_folder_path}': {e}", exc_info=True); return None, None, None, 0
    if not files_to_process: logging.warning(f"No .txt files found in the '{text_folder_path}' subdirectory."); return [], {}, 0, 0

    total_files_found = len(files_to_process)
    num_files_processed_successfully = 0
    start_time = time.time()
    logging.info(f"Found {total_files_found} text files in '{text_subdir_name}'. Processing...")
    model = None
    if total_files_found >= 2:
        try: logging.info(f"Loading embedding model ({EMBEDDING_MODEL_NAME})..."); model = SentenceTransformer(EMBEDDING_MODEL_NAME); logging.info("Embedding model loaded.")
        except Exception as e: logging.error(f"Failed to load embedding model: {e}. Embedding metrics will fail.", exc_info=True); model = None

    # --- Process Individual Files ---
    for i, filename in enumerate(files_to_process):
        file_path = os.path.join(text_folder_path, filename)
        logging.info(f"Processing file {i+1}/{total_files_found}: {filename} (from '{text_subdir_name}' folder)...")
        file_metrics = {hdr: FAILED_METRIC_PLACEHOLDER for hdr in HEADER_ORDER}
        file_metrics['Filename'] = filename
        file_processed_flag = False

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            all_raw_texts.append(content) # Store raw text for NMF/TFIDF

            # --- Basic Counts ---
            file_metrics['Total Characters'] = len(content)
            tokens_orig_case = safe_tokenize_words(content, filename)
            if tokens_orig_case is None: logging.error(f"Skipping metrics for {filename} due to word tokenization error."); results.append(file_metrics); continue
            tokens_lower = [t.lower() for t in tokens_orig_case]
            total_words = len(tokens_lower); unique_words = len(set(tokens_lower))
            file_metrics['Total Words'] = total_words; file_metrics['Unique Words'] = unique_words

            # --- Sentence Structure ---
            sentences = safe_tokenize_sentences(content, filename)
            file_metrics['Average Sentence Length'] = calculate_avg_sentence_length(total_words, sentences, filename)

            # --- Readability (NEW) ---
            textstat_scores = calculate_textstat_metrics(content, filename)
            file_metrics.update(textstat_scores) # Add all scores from the dict

            # --- Lexical Diversity ---
            file_metrics['TTR'] = (unique_words / total_words) if total_words > 0 else 0.0
            file_metrics['RTTR'] = calculate_generic_metric(ld.root_ttr, tokens_lower, filename=filename)
            file_metrics['Herdan\'s C (LogTTR)'] = calculate_generic_metric(ld.log_ttr, tokens_lower, filename=filename)
            file_metrics[f'MATTR (W={MATTR_WINDOW_SIZE})'] = calculate_generic_metric(ld.mattr, tokens_lower, MATTR_WINDOW_SIZE, filename=filename)
            file_metrics[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = calculate_generic_metric(ld.msttr, tokens_lower, MSTTR_WINDOW_SIZE, filename=filename)
            file_metrics['VOCD'] = calculate_generic_metric(ld.hdd, tokens_lower, filename=filename)
            file_metrics['MTLD'] = calculate_generic_metric(ld.mtld, tokens_lower, filename=filename)
            file_metrics['Yule\'s K'] = calculate_yules_k_manual(tokens_lower, filename=filename)
            file_metrics['Simpson\'s D'] = calculate_simpsons_d_manual(tokens_lower, filename=filename)

            # --- Syntactic/Phrase Diversity ---
            tagged_tokens = safe_pos_tag(tokens_orig_case, filename)
            file_metrics['Lexical Density'] = calculate_lexical_density(tagged_tokens, total_words, filename)
            for n in NGRAM_N_VALUES:
                rep_n, dist_n = calculate_ngram_stats(tokens_lower, n, filename=filename)
                file_metrics[f'Distinct-{n}'] = dist_n
                file_metrics[f'Repetition-{n}'] = rep_n

            # --- Sentiment/Emotion (NEW) ---
            vader_scores = calculate_vader_sentiment(content, filename)
            file_metrics.update(vader_scores)
            textblob_scores = calculate_textblob_sentiment(content, filename)
            file_metrics.update(textblob_scores)
            nrclex_scores = calculate_nrclex_emotions(content, filename)
            file_metrics.update(nrclex_scores)

            # --- Keyword Extraction (YAKE!) (NEW - Per File) ---
            file_metrics['Keywords (YAKE! Top 5)'] = extract_yake_keywords(content, N_KEYWORDS, filename)

            # Store data for collective metrics
            all_tokenized_texts_for_bleu.append(tokens_lower)
            filenames_processed.append(filename)
            results.append(file_metrics)
            file_processed_flag = True

        except Exception as e:
            logging.error(f"Unexpected error processing file {filename}: {e}", exc_info=True)
            if not file_processed_flag: results.append(file_metrics)

    # --- Calculate Corpus-Wide Metrics (NMF Topics, TF-IDF Keywords) ---
    nmf_topic_assignments = None
    nmf_topic_probabilities = None
    tfidf_keywords = None

    if all_raw_texts:
        # NMF Topics
        nmf_topic_assignments, nmf_topic_probabilities, vectorizer, nmf_model, tfidf_matrix = calculate_nmf_topics(all_raw_texts, N_TOPICS)
        # TF-IDF Keywords
        if tfidf_matrix is not None and vectorizer is not None:
             tfidf_keywords = extract_tfidf_keywords(tfidf_matrix, vectorizer, N_KEYWORDS)
        else:
             logging.warning("TF-IDF matrix/vectorizer unavailable from NMF step. Skipping TF-IDF keywords.")
             tfidf_keywords = [FAILED_METRIC_PLACEHOLDER] * len(all_raw_texts)

        # --- Merge Corpus-Wide Metrics back into results ---
        if nmf_topic_assignments is not None and nmf_topic_probabilities is not None and len(nmf_topic_assignments) == len(results):
            for i, res_dict in enumerate(results):
                res_dict['Topic (NMF ID)'] = nmf_topic_assignments[i]
                res_dict['Topic (NMF Prob)'] = nmf_topic_probabilities[i]
        else:
            logging.warning("NMF results length mismatch or calculation failed. Skipping NMF merge.")
            for i, res_dict in enumerate(results): # Still ensure columns exist
                 if 'Topic (NMF ID)' not in res_dict: res_dict['Topic (NMF ID)'] = FAILED_METRIC_PLACEHOLDER
                 if 'Topic (NMF Prob)' not in res_dict: res_dict['Topic (NMF Prob)'] = FAILED_METRIC_PLACEHOLDER

        if tfidf_keywords is not None and len(tfidf_keywords) == len(results):
            for i, res_dict in enumerate(results):
                res_dict['Keywords (TF-IDF Top 5)'] = tfidf_keywords[i]
        else:
            logging.warning("TF-IDF keyword results length mismatch or calculation failed. Skipping TF-IDF keyword merge.")
            for i, res_dict in enumerate(results): # Still ensure column exists
                 if 'Keywords (TF-IDF Top 5)' not in res_dict: res_dict['Keywords (TF-IDF Top 5)'] = FAILED_METRIC_PLACEHOLDER

    else:
        logging.warning("No raw texts collected. Skipping NMF and TF-IDF keyword calculations.")
        # Ensure columns exist even if calculation skipped
        for res_dict in results:
            if 'Topic (NMF ID)' not in res_dict: res_dict['Topic (NMF ID)'] = FAILED_METRIC_PLACEHOLDER
            if 'Topic (NMF Prob)' not in res_dict: res_dict['Topic (NMF Prob)'] = FAILED_METRIC_PLACEHOLDER
            if 'Keywords (TF-IDF Top 5)' not in res_dict: res_dict['Keywords (TF-IDF Top 5)'] = FAILED_METRIC_PLACEHOLDER


    # --- Calculate Collective Metrics (Embeddings, Self-BLEU) ---
    num_files_processed_successfully = len(filenames_processed)
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
    return results, collective_metrics, analysis_duration, num_files_processed_successfully


# --- Saving Functions (Updated for new headers/floats) ---

def format_value(value, is_float=False):
    """Formats values for output, converting None to 'FAIL'."""
    if value is None: return "FAIL"
    # --- MODIFIED: Check for string first to handle JSON ---
    if isinstance(value, str):
        # Check if it looks like a JSON string (basic check)
        if value.startswith('{') and value.endswith('}'):
            return value # Return JSON string as is
        # Fall through to float/string conversion if not JSON-like
    # --- END MODIFICATION ---
    if isinstance(value, (dict, list)): # Handle actual dict/list (shouldn't happen often here)
        try: return json.dumps(value)
        except TypeError: return str(value)
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
        # Add collective metrics to the float columns list if they are numeric
        temp_float_cols = FLOAT_COLUMNS.copy()
        for key, value in collective_metrics.items():
            if isinstance(value, (float, np.floating)):
                temp_float_cols.append(key)

        data_for_df = []
        for res_dict in results:
            formatted_row = {}
            # Handle potential key renaming (e.g., MATTR/MSTTR)
            if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict: res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
            if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict: res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

            for header in HEADER_ORDER: # Use the updated HEADER_ORDER
                is_float = header in temp_float_cols
                # Ensure all expected columns are present, even if calculation failed
                formatted_row[header] = format_value(res_dict.get(header, FAILED_METRIC_PLACEHOLDER), is_float)
            data_for_df.append(formatted_row)

        df = pd.DataFrame(data_for_df, columns=HEADER_ORDER) # Ensure columns match header order
        df.to_csv(output_path, index=False, encoding='utf-8')

        # Append Collective Metrics
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow([]);
            # --- PATCHED: Write the correct separator ---
            writer.writerow(["--- Collective Metrics (E) ---"])
            # --- END PATCH ---
            for key, value in collective_metrics.items():
                is_coll_float = key in temp_float_cols
                writer.writerow([key, format_value(value, is_coll_float)])
        logging.info("CSV Results saved successfully.")
    except Exception as e: logging.error(f"Error writing CSV output file '{output_path}': {e}", exc_info=True)


def save_results_txt(results, collective_metrics, output_folder):
    """Saves results to a formatted TXT file."""
    output_path = os.path.join(output_folder, f"{OUTPUT_FILENAME_BASE}.txt")
    logging.info(f"Saving TXT results to: {output_path}")
    if not results: logging.warning("No results to save."); return

    # --- Define fixed widths (Adjusted for new metrics) ---
    # This might need significant tuning based on expected value ranges
    widths = {
        'Filename': 25, 'Total Words': 12, 'Total Characters': 15, 'Unique Words': 12, 'Average Sentence Length': 15, # Basic
        'Flesch Reading Ease': 15, 'Gunning Fog': 12, 'SMOG Index': 12, 'Dale-Chall Score': 15, # Readability
        'TTR': 10, 'RTTR': 10, "Herdan's C (LogTTR)": 18, f'MATTR (W={MATTR_WINDOW_SIZE})': 15, f'MSTTR (W={MSTTR_WINDOW_SIZE})': 15,
        'VOCD': 10, 'MTLD': 10, "Yule's K": 10, "Simpson's D": 12, # Diversity
        'Lexical Density': 15, 'Distinct-2': 10, 'Repetition-2': 12, 'Distinct-3': 10, 'Repetition-3': 12, # Syntactic/Phrase
        'Sentiment (VADER Comp)': 15, 'Sentiment (VADER Pos)': 15, 'Sentiment (VADER Neu)': 15, 'Sentiment (VADER Neg)': 15,
        'Sentiment (TextBlob Pol)': 15, 'Sentiment (TextBlob Subj)': 15, # Sentiment
        'Emotion (NRCLex Dominant)': 18, 'Emotion (NRCLex Scores JSON)': 25, # Emotion (JSON might be long)
        'Topic (NMF ID)': 12, 'Topic (NMF Prob)': 15, # Topics
        'Keywords (TF-IDF Top 5)': 30, 'Keywords (YAKE! Top 5)': 30 # Keywords
    }
    # Assign default width for any missing headers
    for h in HEADER_ORDER:
        if h not in widths: widths[h] = 15 # Default width

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # --- Write Header ---
            header_line = ""; separator_line = ""
            # Simple header for TXT - sections are too complex with many columns
            for i, h in enumerate(HEADER_ORDER):
                width = widths[h]
                header_line += f"{h:<{width}}"
                separator_line += "-" * width
                if i < len(HEADER_ORDER) - 1:
                    header_line += " | "
                    separator_line += "---"
            f.write(header_line + "\n")
            f.write(separator_line + "\n")

            # --- Write Data Rows ---
            for res_dict in results:
                # Handle potential key renaming
                if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict: res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
                if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict: res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

                row_line = ""
                for i, h in enumerate(HEADER_ORDER):
                    width = widths[h]
                    is_float = h in FLOAT_COLUMNS
                    # Special handling for JSON string to prevent excessive width
                    raw_value = res_dict.get(h, FAILED_METRIC_PLACEHOLDER)
                    if h == 'Emotion (NRCLex Scores JSON)' and isinstance(raw_value, str) and len(raw_value) > width:
                         value_str = raw_value[:width-3] + "..." # Truncate JSON string
                    else:
                         value_str = format_value(raw_value, is_float)

                    # Determine alignment (right-align numbers/floats, left-align others)
                    align = ">" if is_float or h in ['Total Words', 'Total Characters', 'Unique Words', 'Topic (NMF ID)'] else "<"

                    formatted_val = f"{value_str:{align}{width}}"
                    # Truncate if still too long (e.g., long filename or keyword string)
                    if len(formatted_val) > width: formatted_val = formatted_val[:width-1] + "."

                    row_line += formatted_val
                    if i < len(HEADER_ORDER) - 1:
                        row_line += " | "
                f.write(row_line + "\n")

            # --- Write Collective Metrics ---
            f.write("\n" + "="*60 + "\n"); f.write("Collective Metrics (E)\n"); f.write("="*60 + "\n") # Use (E) for consistency
            temp_float_cols = FLOAT_COLUMNS.copy() # Recalculate floats including collective
            for key, value in collective_metrics.items():
                if isinstance(value, (float, np.floating)): temp_float_cols.append(key)

            for key, value in collective_metrics.items():
                is_coll_float = key in temp_float_cols
                value_str = format_value(value, is_coll_float)
                f.write(f"{key:<40}: {value_str}\n")

        logging.info("TXT Results saved successfully.")
    except Exception as e:
        logging.error(f"Error writing TXT output file '{output_path}': {e}", exc_info=True)

# --- UPDATED HTML Saving Function (Splits table based on updated SECONDARY_HTML_COLUMNS) ---
def save_results_html(results, collective_metrics, output_folder, analysis_duration, num_files_processed):
    """Saves results to an HTML file with two sortable tables."""
    output_path = os.path.join(output_folder, f"{OUTPUT_FILENAME_BASE}.html")
    logging.info(f"Saving initial HTML results to: {output_path}")
    if not results: logging.warning("No results to save."); return

    try:
        # Add collective metrics to the float columns list if they are numeric
        temp_float_cols = FLOAT_COLUMNS.copy()
        for key, value in collective_metrics.items():
            if isinstance(value, (float, np.floating)):
                temp_float_cols.append(key)

        # Prepare data for DataFrame, keeping numeric types where possible for sorting
        data_for_df = []
        for res_dict in results:
            row_data = {}
            # Handle potential key renaming
            if f'MATTR (W={MATTR_WINDOW_SIZE})' not in res_dict and 'MATTR' in res_dict: res_dict[f'MATTR (W={MATTR_WINDOW_SIZE})'] = res_dict.pop('MATTR')
            if f'MSTTR (W={MSTTR_WINDOW_SIZE})' not in res_dict and 'MSTTR' in res_dict: res_dict[f'MSTTR (W={MSTTR_WINDOW_SIZE})'] = res_dict.pop('MSTTR')

            for header in HEADER_ORDER: # Use updated HEADER_ORDER
                raw_value = res_dict.get(header, FAILED_METRIC_PLACEHOLDER)
                if raw_value is None:
                    row_data[header] = np.nan # Use NaN for sorting
                else:
                    # Attempt conversion for numeric columns, keep others as string
                    if header in temp_float_cols or header in ['Total Words', 'Total Characters', 'Unique Words', 'Topic (NMF ID)']:
                         try: row_data[header] = pd.to_numeric(raw_value)
                         except (ValueError, TypeError): row_data[header] = str(raw_value)
                    else: row_data[header] = str(raw_value)
            data_for_df.append(row_data)

        df = pd.DataFrame(data_for_df, columns=HEADER_ORDER) # Use updated HEADER_ORDER

        # --- Split DataFrame for two tables ---
        # Use the updated SECONDARY_HTML_COLUMNS list
        main_columns = [col for col in HEADER_ORDER if col not in SECONDARY_HTML_COLUMNS or col == 'Filename']
        secondary_columns = SECONDARY_HTML_COLUMNS # Use the updated list

        df_main = df[main_columns]
        df_secondary = df[secondary_columns]

        # --- Generate HTML for Main Table ---
        html_table_main = df_main.to_html(
            index=False,
            border=1,
            classes='dataframe sortable results_table', # Keep class for potential JS targeting
            table_id='resultsTableMain', # New ID
            justify='center',
            na_rep='FAIL',
            float_format='{:.4f}'.format
        )

        # --- Generate HTML for Secondary Table ---
        html_table_secondary = df_secondary.to_html(
            index=False,
            border=1,
            classes='dataframe sortable results_table_secondary', # New class
            table_id='resultsTableSecondary', # New ID
            justify='center',
            na_rep='FAIL',
            float_format='{:.4f}'.format # Apply same formatting
        )

        # Prepare collective metrics HTML part
        collective_html = "<h2>Collective Metrics (E)</h2>\n<ul>\n" # Use (E) for consistency
        for key, value in collective_metrics.items():
            is_coll_float = key in temp_float_cols
            value_str = format_value(value, is_coll_float)
            collective_html += f"  <li><b>{key}:</b> {value_str}</li>\n"
        collective_html += "</ul>"

        # Format duration
        formatted_duration = format_duration(analysis_duration)

        # Basic HTML structure (JavaScript/CSS will be added by visualizer script)
        # Now includes placeholders for BOTH tables and updated H2 for secondary
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Analysis Suite v1.5</title>
    <!-- CSS and JS for sorting will be added by lexicon_visualizer.py -->
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.5; }}
        h1, h2, h3 {{ color: #333; margin-top: 1.5em; }}
        p {{ margin-bottom: 0.5em; }}
        /* Style for BOTH tables */
        table.dataframe {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }}
        table.dataframe th, table.dataframe td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
        table.dataframe th {{ background-color: #f0f0f0; font-weight: bold; text-align: center; }}
        table.dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
        table.dataframe tbody tr:hover {{ background-color: #e5e5e5; }}
        table.dataframe td:first-child {{ text-align: left; }} /* Left-align filename */
        /* Specific style for secondary table if needed */
        .results_table_secondary {{ margin-top: 30px; }} /* Add space before secondary table */

        ul {{ list-style-type: none; padding: 0; margin-left: 20px; /* Indent collective metrics */ }}
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
        hr {{ border: 0; height: 1px; background-color: #ccc; margin-top: 2em; margin-bottom: 2em; }} /* Style HR */
    </style>
</head>
<body>
    <h1>Lexical Analysis Results</h1>
    <p><b>Target Folder:</b> {output_folder}</p>
    <p><b>Analysis Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><b>Total Files Analyzed:</b> {num_files_processed}</p>
    <p><b>Total Time Spent Analyzing:</b> {formatted_duration}</p>

    <h2>Individual File Metrics (Main)</h2>
    {html_table_main}

    <h2>Individual File Metrics (Sentiment, Emotion & Keywords)</h2>
    {html_table_secondary}

    {collective_html}

    <!-- Descriptions, Plots and JS will be added by lexicon_visualizer.py -->
</body>
</html>
"""
        # Write initial HTML content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info("Initial HTML Results saved successfully (with updated split tables).")
    except Exception as e:
        logging.error(f"Error writing initial HTML output file '{output_path}': {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='nltk.translate.bleu_score')
    warnings.filterwarnings("ignore", message="The sentence constituent `.+` does not exist in the dictionary")
    warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn') # Ignore sklearn future warnings for now

    if len(sys.argv) < 2:
        prompt = f"Enter path to project folder (ensure .txt files are in 'txt' subdir) (or wait 5s for current: {os.getcwd()}): "
        user_path = get_input_with_timeout(prompt, 5)
        if user_path is None: target_folder = os.getcwd(); print(f"\nTimeout! Using current directory: {target_folder}"); logging.info(f"Timeout! Using current directory: {target_folder}")
        else:
            user_path_stripped = user_path.strip()
            if not user_path_stripped: target_folder = os.getcwd(); print(f"\nNo path entered. Using current directory: {target_folder}"); logging.info(f"No path entered. Using current directory: {target_folder}")
            else: target_folder = user_path_stripped; print(f"\nUsing provided path: {target_folder}"); logging.info(f"Using provided path: {target_folder}")
    else:
        target_folder = sys.argv[1]
        print(f"Using folder path from command line argument: {target_folder}")
        logging.info(f"Using folder path from command line argument: {target_folder}")

    if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
         logging.critical(f"CRITICAL: Invalid folder path '{target_folder}'")
         sys.exit(1)

    logging.info(f"\n{'='*20} Starting New Analysis Run: {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}")
    logging.info(f"Target Folder: {target_folder}")
    logging.info(f"Source .txt files expected in: {os.path.join(target_folder, 'txt')}")

    individual_results, collective_results, analysis_duration, num_files_processed = process_folder_comprehensive(target_folder)

    if individual_results is not None:
        save_results_csv(individual_results, collective_results, target_folder)
        save_results_txt(individual_results, collective_results, target_folder)
        save_results_html(individual_results, collective_results, target_folder, analysis_duration, num_files_processed)
    else:
        logging.error("Processing failed at a high level. No results saved.")
        sys.exit(1)

    logging.info("Script finished.")