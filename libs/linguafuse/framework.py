from pydantic import BaseModel, Field
from typing import Union
import os

from linguafuse.loader import ProcessedDataset
import pandas as pd

class FineTuneOrchestration(BaseModel):
    """ Orchestration for E2E fine-tuning. """
    path: Union[str, os.PathLike] = Field(..., description="Path or URI to the dataset.")

    def _return_dataset(self):
        """ Returns the dataset. """
        data = pd.read_csv(self.path)
        if data.empty:
            raise ValueError("The dataset is empty.")
        return ProcessedDataset(data=data)
    