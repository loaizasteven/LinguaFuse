from typing import Any, Optional

from linguafuse.cloud import ConnectionManager, Scope
from transformers import AutoModelForSequenceClassification


def load_transformer(scope: Scope, model_details: Any = "bert-base-uncased", num_labels: Optional[int] = None) -> AutoModelForSequenceClassification:
    """
    Load the transformer model for sequence classification based on scope.
    """
    def return_connection_source():
       connection = ConnectionManager(scope=scope, asset_details=model_details)
       connection.connect()
       return connection.path
    
    # Determine pretrained source
    source = model_details if scope == Scope.LOCAL else return_connection_source()
    # Load model with optional num_labels override
    if num_labels is not None:
        return AutoModelForSequenceClassification.from_pretrained(source, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(source)
