"""
Initialize NER model (spaCy).
"""

import logging
import spacy
from spacy.cli import download

logger = logging.getLogger(__name__)


def init_ner_model(load_to_ram: bool = True, language: str = "de"):
    """
    Initialize the Named Entity Recognition model using spaCy.

    Uses de_core_news_lg for German NER with multilingual fallback.

    Args:
        load_to_ram: If True, loads model into memory. If False, only validates availability.
        language: Language code (default: "de" for German)

    Returns:
        spaCy NLP pipeline or None if load_to_ram=False
    """
    # Map language codes to spaCy models
    model_map = {
        "de": "de_core_news_lg",
        "en": "en_core_web_lg",
        "multilingual": "xx_ent_wiki_sm"
    }

    model_name = model_map.get(language, "de_core_news_lg")

    try:
        if not load_to_ram:
            # Just validate model availability without loading
            logger.info(f"Validating NER model availability: {model_name}")
            try:
                spacy.load(model_name)
                logger.info(f"NER model {model_name} is available")
            except OSError:
                logger.warning(f"Model {model_name} not found, attempting download...")
                download(model_name)
                logger.info(f"NER model {model_name} downloaded successfully")
            return None

        # Load model into memory
        logger.info(f"Loading NER model: {model_name}...")

        nlp = spacy.load(model_name)

        # Disable unnecessary pipeline components for performance
        # Keep only: tok2vec, tagger, morphologizer, parser, ner
        disabled_pipes = []
        for pipe_name in nlp.pipe_names:
            if pipe_name not in ["tok2vec", "tagger", "morphologizer", "parser", "ner", "attribute_ruler", "lemmatizer"]:
                disabled_pipes.append(pipe_name)
                nlp.disable_pipe(pipe_name)

        logger.info(f"NER model loaded successfully: {model_name}")
        logger.info(f"Active pipeline components: {nlp.pipe_names}")
        if disabled_pipes:
            logger.info(f"Disabled components for performance: {disabled_pipes}")

        return nlp

    except Exception as e:
        logger.error(f"Error initializing NER model: {e}")
        raise
