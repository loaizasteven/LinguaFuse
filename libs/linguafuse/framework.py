from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Union
import os

from transformers import PreTrainedTokenizerFast

from linguafuse.loader.dataset import ProcessedDataset
from linguafuse.loader.transformer import load_transformer

import pandas as pd

from linguafuse.cloud import (
    Scope,
    ConnectionManager,
)


class AwsDataArguments(BaseModel):
    """ AWS arguments for the connection manager. """
    bucket: str = Field(..., description="The S3 bucket name.")


class AmlDataArguments(BaseModel):
    """ AWS arguments for the connection manager. """
    name: str = Field(..., description="The name of the data asset.")
    version: str = Field(..., description="The version of the data asset.")

class LocalDataArguments(BaseModel):
    """ AWS arguments for the connection manager. """
    path: Union[str, os.PathLike] = Field(..., description="The local path to the dataset.")


class FineTuneOrchestration(BaseModel):
    """ Orchestration for E2E fine-tuning. """
    data_args: Union[AwsDataArguments, LocalDataArguments, AmlDataArguments] = Field(..., description="Path or URI to the dataset.")
    scope: Scope = Field(
        default=Scope.LOCAL,
        description="The scope of the orchestration, indicating the cloud service it connects to."
    )
    tokenizer: PreTrainedTokenizerFast = Field(
        ...,
        description="The tokenizer to be used for the dataset."
    )
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def _return_dataset(self):
        """ Returns the dataset. """
        conn = ConnectionManager(scope=self.scope, asset_details=self.data_args)
        conn.connect()
        data = conn.read_pandas()
        if data.empty:
            raise ValueError("The dataset is empty.")
        return ProcessedDataset(data=data, tokenizer=self.tokenizer)
    
    def load_model(self, model_details: Any = "bert-base-uncased"):
        """
        Load the transformer model based on the orchestration scope.
        """
        # Delegate loading to loader.transformer
        return load_transformer(self.scope, model_details)
