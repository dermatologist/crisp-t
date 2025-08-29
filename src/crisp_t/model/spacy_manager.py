"""spaCy model management."""

import logging
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)


class SpacyManager:
    """Manages spaCy model loading and initialization."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._model = None

    def get_model(self):
        """Get or load the spaCy model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the spaCy model with fallback strategies."""
        try:
            return spacy.load(self.model_name)
        except TypeError:
            # Try loading from local path
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / self.model_name
            return spacy.load(str(model_path))
        except OSError:
            logger.info("Downloading spaCy English model, this may take a while...")
            from spacy.cli import download

            download(self.model_name)
            return spacy.load(self.model_name)
