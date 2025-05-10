from pydantic import BaseModel, Field
from typing import Union
import os

from linguafuse.loader.dataset import ProcessedDataset
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
    path: str = Field(..., description="The local path to the dataset.")


class FineTuneOrchestration(BaseModel):
    """ Orchestration for E2E fine-tuning. """
    data_args: Union[AwsDataArguments, LocalDataArguments, AmlDataArguments] = Field(..., description="Path or URI to the dataset.")
    scope: str = Field(
        default=Scope.LOCAL,
        description="The scope of the orchestration, indicating the cloud service it connects to."
    ) 
    def _return_dataset(self):
        """ Returns the dataset. """
        conn = ConnectionManager(scope=self.scope, asset_details=self.data_args)
        conn.connect()
        data = conn.read_pandas()
        if data.empty:
            raise ValueError("The dataset is empty.")
        return ProcessedDataset(data=data)
