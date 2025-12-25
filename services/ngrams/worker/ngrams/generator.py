"""
N-gram generation and frequency analysis logic.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import re
import hashlib

logger = logging.getLogger(__name__)


def generate_ngrams_from_corpus(
    messages: List[Dict[str, Any]],
    n_values: List[int] = [2, 3],
    min_frequency: int = 5,
    exclude_stopwords: bool = True,
    stopwords_language: str = "german"
) -> List[Dict[str, Any]]:
    """
    Generate n-grams from a corpus of messages.

    Args:
        messages: List of message dictionaries with 'id', 'text', 'chat_id'
        n_values: List of n values to generate (1=unigram, 2=bigram, 3=trigram)
        min_frequency: Minimum frequency threshold
        exclude_stopwords: Whether to exclude common stopwords
        stopwords_language: Language for stopwords

    Returns:
        List of n-gram documents for Elasticsearch
    """
    logger.info(f"Generating n-grams for {len(messages)} messages")
    logger.info(f"N-values: {n_values}, Min frequency: {min_frequency}")

    # Load stopwords
    stopwords = _get_stopwords(stopwords_language) if exclude_stopwords else set()

    # Initialize counters for each n value
    ngram_counters = {n: Counter() for n in n_values}
    ngram_docs = {n: defaultdict(lambda: {
        "texts": [],
        "example_messages": set(),
        "chat_ids": set(),
        "document_count": 0
    }) for n in n_values}

    # Process each message
    for msg in messages:
        text = msg.get("text", "")
        if not text or len(text.strip()) < 3:
            continue

        message_id = msg["id"]
        chat_id = msg["chat_id"]

        # Tokenize and clean
        tokens = _tokenize_text(text, stopwords)

        if len(tokens) < max(n_values):
            continue

        # Generate n-grams for each n value
        for n in n_values:
            if len(tokens) < n:
                continue

            message_ngrams = _extract_ngrams(tokens, n)

            for ngram in message_ngrams:
                ngram_text = " ".join(ngram)
                ngram_counters[n][ngram_text] += 1

                # Track document frequency and examples
                if ngram_text not in ngram_docs[n] or len(ngram_docs[n][ngram_text]["example_messages"]) < 10:
                    ngram_docs[n][ngram_text]["texts"].append(ngram)
                    ngram_docs[n][ngram_text]["example_messages"].add(message_id)
                    ngram_docs[n][ngram_text]["chat_ids"].add(chat_id)

                # Increment document count only once per message
                if message_id not in ngram_docs[n][ngram_text]["example_messages"]:
                    ngram_docs[n][ngram_text]["document_count"] += 1

    # Compile final n-gram documents
    ngram_results = []
    total_docs = len(messages)

    for n in n_values:
        logger.info(f"Processing {n}-grams...")

        for ngram_text, count in ngram_counters[n].items():
            if count < min_frequency:
                continue

            doc_data = ngram_docs[n][ngram_text]
            doc_freq = len(doc_data["example_messages"])

            # Calculate TF-IDF score (simplified)
            tf_idf = _calculate_tf_idf(count, doc_freq, total_docs)

            # Generate n-gram ID
            ngram_id = _generate_ngram_id(ngram_text, n)

            ngram_doc = {
                "ngram_id": ngram_id,
                "text": ngram_text,
                "n": n,
                "frequency": count,
                "document_frequency": doc_freq,
                "tf_idf_score": tf_idf,
                "chat_ids": list(doc_data["chat_ids"]),
                "example_messages": list(doc_data["example_messages"])[:5],  # Limit to 5 examples
                "language": stopwords_language,
                "generated_at": datetime.utcnow().isoformat()
            }
            ngram_results.append(ngram_doc)

        logger.info(f"Generated {len([ng for ng in ngram_results if ng['n'] == n])} {n}-grams")

    # Sort by frequency descending
    ngram_results.sort(key=lambda x: x["frequency"], reverse=True)

    logger.info(f"Total n-grams generated: {len(ngram_results)}")
    return ngram_results


def compute_word_frequency(
    messages: List[Dict[str, Any]],
    min_frequency: int = 5,
    exclude_stopwords: bool = True,
    stopwords_language: str = "german",
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Compute word frequency distribution (unigrams).

    Args:
        messages: List of message dictionaries
        min_frequency: Minimum frequency threshold
        exclude_stopwords: Whether to exclude stopwords
        stopwords_language: Language for stopwords
        limit: Maximum number of words to return

    Returns:
        List of word frequency entries
    """
    logger.info(f"Computing word frequencies for {len(messages)} messages")

    stopwords = _get_stopwords(stopwords_language) if exclude_stopwords else set()

    word_counter = Counter()
    doc_counter = Counter()  # Document frequency

    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue

        tokens = _tokenize_text(text, stopwords)
        unique_tokens = set(tokens)

        # Update frequency counters
        word_counter.update(tokens)
        doc_counter.update(unique_tokens)

    # Create word frequency list
    total_docs = len(messages)
    word_frequencies = []

    for word, count in word_counter.most_common(limit):
        if count < min_frequency:
            continue

        doc_freq = doc_counter[word]
        tf_idf = _calculate_tf_idf(count, doc_freq, total_docs)

        word_frequencies.append({
            "word": word,
            "count": count,
            "documents": doc_freq,
            "tf_idf_score": tf_idf
        })

    logger.info(f"Computed frequencies for {len(word_frequencies)} words")
    return word_frequencies


def _tokenize_text(text: str, stopwords: set) -> List[str]:
    """
    Tokenize text into clean tokens.

    Args:
        text: Input text
        stopwords: Set of stopwords to exclude

    Returns:
        List of cleaned tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove special characters but keep umlauts and basic punctuation
    text = re.sub(r'[^\w\säöüß\-]', ' ', text)

    # Split into tokens
    tokens = text.split()

    # Filter tokens
    cleaned_tokens = []
    for token in tokens:
        # Skip very short tokens
        if len(token) < 2:
            continue

        # Skip pure numbers
        if token.isdigit():
            continue

        # Skip stopwords
        if token in stopwords:
            continue

        # Skip tokens that are just punctuation
        if all(c in '-_' for c in token):
            continue

        cleaned_tokens.append(token)

    return cleaned_tokens


def _extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        List of n-gram tuples
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams


def _calculate_tf_idf(term_freq: int, doc_freq: int, total_docs: int) -> float:
    """
    Calculate simplified TF-IDF score.

    Args:
        term_freq: Term frequency (total occurrences)
        doc_freq: Document frequency (how many docs contain term)
        total_docs: Total number of documents

    Returns:
        TF-IDF score
    """
    import math

    if doc_freq == 0 or total_docs == 0:
        return 0.0

    # TF = term frequency
    tf = term_freq

    # IDF = log(total_docs / doc_freq)
    idf = math.log(total_docs / doc_freq) if doc_freq > 0 else 0

    return tf * idf


def _generate_ngram_id(ngram_text: str, n: int) -> str:
    """Generate unique ID for n-gram."""
    combined = f"{n}:{ngram_text}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _get_stopwords(language: str = "german") -> set:
    """
    Get stopwords for specified language.

    Args:
        language: Language code

    Returns:
        Set of stopwords
    """
    # German stopwords (common words to exclude)
    german_stopwords = {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem",
        "und", "oder", "aber", "wenn", "als", "wie", "bei", "nach", "vor", "über", "unter",
        "von", "zu", "zum", "zur", "im", "am", "ist", "sind", "war", "waren", "wird", "werden",
        "hat", "haben", "hatte", "hatten", "kann", "können", "muss", "müssen", "soll", "sollen",
        "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
        "nicht", "auch", "nur", "so", "sehr", "mehr", "noch", "nun", "schon", "doch", "ja", "nein",
        "auf", "aus", "mit", "durch", "für", "gegen", "ohne", "um",
        "da", "hier", "dort", "jetzt", "dann", "immer", "nie", "oft",
        "in", "an", "auf", "bei", "mit", "nach", "seit", "von", "zu", "bis"
    }

    # English stopwords
    english_stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "as", "at", "by", "for", "from",
        "in", "of", "on", "to", "with", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should", "could",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "not", "no", "yes", "can", "may", "must", "shall"
    }

    language_map = {
        "german": german_stopwords,
        "de": german_stopwords,
        "english": english_stopwords,
        "en": english_stopwords
    }

    return language_map.get(language.lower(), german_stopwords)
