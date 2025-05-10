from typing import Any

from linguafuse.cloud import ConnectionManager, Scope
from transformers import AutoModelForSequenceClassification


def load_transformer(scope: Scope, model_details: Any = "bert-base-uncased") -> AutoModelForSequenceClassification:
    """
    Load the transformer model for sequence classification based on scope.
    """
    if scope == Scope.LOCAL:
        # Load from Hugging Face hub when local
        model_path=model_details
    else:
        # Fetch the weights path via ConnectionManager for cloud scopes
        conn = ConnectionManager(scope=scope, asset_details=model_details)
        conn.connect()
        model_path = conn.path
    
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
