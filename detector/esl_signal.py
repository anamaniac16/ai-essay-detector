"""
esl_signal.py — Independent ESL (English as a Second Language) bias detector.

This module estimates whether an essay shows patterns typical of non-native
English writing. These patterns can overlap with AI-detection signals,
causing false positives. This module is INTENTIONALLY INDEPENDENT from the
AI classifier so the two can be compared side-by-side.

Signals used:
  1. Article/preposition irregularity rate
  2. Lexical diversity (type-token ratio variants)
  3. Sentence complexity distribution
  4. Repeated simple sentence structures
"""

import re
import math
from typing import Dict, List
from collections import Counter


# ---------------------------------------------------------------------------
# 1. Article/preposition irregularity heuristics
# ---------------------------------------------------------------------------

# Common article/preposition patterns that native speakers rarely get wrong
# but ESL writers frequently do
ARTICLE_PATTERNS = {
    # Missing article before singular countable noun after verb
    r'\b(?:is|was|has|have|had)\s+(?:very\s+)?(?:good|great|important|big|small|new|old)\s+(?:thing|place|person|idea|problem|way)\b': "missing_article",
    # Double preposition
    r'\b(?:in|on|at|to|for|with|by)\s+(?:in|on|at|to|for|with|by)\b': "double_preposition",
    # "the" before proper nouns that don't need it
    r'\bthe\s+(?:God|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b': "unnecessary_article",
}

# Prepositions that ESL writers commonly confuse
CONFUSED_PREPS = [
    (r'\bdepend of\b', "depend_on"),
    (r'\bconsist in\b', "consist_of"),
    (r'\binterested for\b', "interested_in"),
    (r'\bsuffer of\b', "suffer_from"),
    (r'\binsist to\b', "insist_on"),
    (r'\bcapable to\b', "capable_of"),
    (r'\bmarried with\b', "married_to"),
    (r'\bsimilar with\b', "similar_to"),
    (r'\bdifferent of\b', "different_from"),
]


def article_preposition_irregularity(text: str) -> float:
    """
    Score [0, 1] estimating article/preposition irregularity.
    Higher = more irregularities detected (more likely ESL).
    """
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    irregularities = 0

    # Check article patterns
    for pattern in ARTICLE_PATTERNS:
        irregularities += len(re.findall(pattern, text_lower))

    # Check confused prepositions
    for pattern, _ in CONFUSED_PREPS:
        irregularities += len(re.findall(pattern, text_lower))

    # Normalize by word count (per 100 words)
    rate = (irregularities / word_count) * 100
    return min(rate, 1.0)  # Cap at 1.0


# ---------------------------------------------------------------------------
# 2. Lexical diversity measures
# ---------------------------------------------------------------------------

def type_token_ratio(text: str) -> float:
    """Basic TTR — unique words / total words."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hapax_legomena_ratio(text: str) -> float:
    """Proportion of words that appear exactly once."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(words)


def vocabulary_sophistication(text: str) -> float:
    """
    Ratio of "sophisticated" words (>= 3 syllables) to total words.
    ESL writers often use simpler vocabulary or awkwardly complex words.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0

    def syllable_count(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and count > 1:
            count -= 1
        return max(count, 1)

    sophisticated = sum(1 for w in words if syllable_count(w) >= 3)
    return sophisticated / len(words)


# ---------------------------------------------------------------------------
# 3. Sentence complexity distribution
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def sentence_complexity_features(text: str) -> Dict[str, float]:
    """
    Analyze sentence complexity distribution.
    ESL writing often shows:
      - Lower clause density (fewer subordinate clauses)
      - More uniform sentence length
      - Higher ratio of simple sentences
    """
    sentences = _split_sentences(text)
    if not sentences:
        return {
            "avg_words_per_sentence": 0.0,
            "sentence_length_cv": 0.0,
            "simple_sentence_ratio": 0.0,
            "subordinate_clause_density": 0.0,
        }

    # Word counts
    word_counts = [len(s.split()) for s in sentences]
    avg_wps = sum(word_counts) / len(word_counts)
    std_wps = (sum((w - avg_wps) ** 2 for w in word_counts) / len(word_counts)) ** 0.5
    cv = std_wps / avg_wps if avg_wps > 0 else 0.0

    # Simple sentence ratio (no commas, no subordinating conjunctions)
    subordinators = {"because", "although", "while", "when", "if", "since",
                     "unless", "whereas", "though", "whenever", "wherever",
                     "whether", "after", "before", "until", "as"}

    simple_count = 0
    sub_clause_count = 0
    for s in sentences:
        words_lower = s.lower().split()
        has_subordinator = any(w.strip('.,;:') in subordinators for w in words_lower)
        has_comma = ',' in s
        if not has_subordinator and not has_comma:
            simple_count += 1
        if has_subordinator:
            sub_clause_count += 1

    return {
        "avg_words_per_sentence": avg_wps,
        "sentence_length_cv": cv,
        "simple_sentence_ratio": simple_count / len(sentences),
        "subordinate_clause_density": sub_clause_count / len(sentences),
    }


# ---------------------------------------------------------------------------
# 4. Repeated structure detection
# ---------------------------------------------------------------------------

def sentence_start_repetition(text: str) -> float:
    """
    Ratio of sentences starting with the same word.
    ESL writers often start many sentences with "I", "The", "It", etc.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
    counts = Counter(starts)
    most_common_count = counts.most_common(1)[0][1] if counts else 0
    return most_common_count / len(sentences)


# ---------------------------------------------------------------------------
# Combined ESL signal
# ---------------------------------------------------------------------------

ESL_FEATURE_NAMES = [
    "article_prep_irregularity",
    "type_token_ratio",
    "hapax_ratio",
    "vocabulary_sophistication",
    "avg_words_per_sentence",
    "sentence_length_cv",
    "simple_sentence_ratio",
    "subordinate_clause_density",
    "sentence_start_repetition",
]


def extract_esl_features(text: str) -> Dict[str, float]:
    """
    Extract all ESL-indicator features from an essay.
    These are INDEPENDENT from the AI-detection features.
    """
    complexity = sentence_complexity_features(text)

    return {
        "article_prep_irregularity": article_preposition_irregularity(text),
        "type_token_ratio": type_token_ratio(text),
        "hapax_ratio": hapax_legomena_ratio(text),
        "vocabulary_sophistication": vocabulary_sophistication(text),
        "avg_words_per_sentence": complexity["avg_words_per_sentence"],
        "sentence_length_cv": complexity["sentence_length_cv"],
        "simple_sentence_ratio": complexity["simple_sentence_ratio"],
        "subordinate_clause_density": complexity["subordinate_clause_density"],
        "sentence_start_repetition": sentence_start_repetition(text),
    }


def esl_likelihood_score(text: str) -> float:
    """
    Heuristic ESL likelihood score [0, 1].
    Based on combination of ESL-indicator features.

    This is NOT a trained classifier — it's a rule-based heuristic
    that flags patterns commonly associated with non-native English.
    """
    features = extract_esl_features(text)

    score = 0.0
    signals = 0

    # Article/preposition issues (strong signal)
    if features["article_prep_irregularity"] > 0.02:
        score += 0.3
        signals += 1

    # Low vocabulary sophistication
    if features["vocabulary_sophistication"] < 0.10:
        score += 0.15
        signals += 1

    # Very high simple sentence ratio
    if features["simple_sentence_ratio"] > 0.7:
        score += 0.15
        signals += 1

    # Low subordinate clause density
    if features["subordinate_clause_density"] < 0.15:
        score += 0.1
        signals += 1

    # High sentence start repetition
    if features["sentence_start_repetition"] > 0.4:
        score += 0.15
        signals += 1

    # Low sentence length variation
    if features["sentence_length_cv"] < 0.25:
        score += 0.1
        signals += 1

    # Low type-token ratio (limited vocabulary)
    if features["type_token_ratio"] < 0.4:
        score += 0.1
        signals += 1

    return min(score, 1.0)


def esl_flag_active(text: str, threshold: float = 0.35) -> bool:
    """Return True if ESL signal exceeds threshold."""
    return esl_likelihood_score(text) >= threshold


def esl_report(text: str) -> Dict:
    """
    Generate a detailed ESL analysis report.
    Returns features, score, and whether the flag is active.
    """
    features = extract_esl_features(text)
    score = esl_likelihood_score(text)
    return {
        "features": features,
        "esl_score": score,
        "flag_active": score >= 0.35,
        "message": (
            "This essay shows patterns common in non-native English writing "
            "that can resemble AI text on some metrics — treat the AI score "
            "with added caution."
            if score >= 0.35
            else "No significant ESL patterns detected."
        ),
    }
