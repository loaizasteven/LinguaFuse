from pydantic import BaseModel, Field
from typing import Union
import os

from linguafuse.loader.dataset import ProcessedDataset
import pandas as pd

from linguafuse.cloud import (
    Scope,
    ConnectionManager,
)

class FineTuneOrchestration(BaseModel):
    """ Orchestration for E2E fine-tuning. """
    path: Union[str, os.PathLike] = Field(..., description="Path or URI to the dataset.")
    scope: str = Field(
        default=Scope.LOCAL,
        description="The scope of the orchestration, indicating the cloud service it connects to."
    ) 
    def _return_dataset(self):
        """ Returns the dataset. """
        conn = ConnectionManager(scope=self.scope, asset_details=self.path)
        conn.connect()
        data = conn.read_pandas()
        if data.empty:
            raise ValueError("The dataset is empty.")
        return ProcessedDataset(data=data)
