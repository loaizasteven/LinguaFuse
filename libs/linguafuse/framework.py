from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Union, Optional
import os

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from linguafuse.loader.dataset import ProcessedDataset
from linguafuse.loader.transformer import load_transformer
from linguafuse.trainer import TrainerArguments, TrainerRunner

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
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = Field(
        ...,
        description="The tokenizer to be used for the dataset."
    )
    # Number of labels inferred from dataset, set when loading model
    num_labels: Optional[int] = Field(
        default=None,
        description="Number of labels for the sequence classification model"
    )
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    processed_dataset: ProcessedDataset = Field(
        default=None,
        description="The processed dataset."
    )
    dataset: Optional[Any] = Field(default=None, description="The DataSet object.")

    def _create_dataset(self):
        """ Returns the dataset. """
        conn = ConnectionManager(scope=self.scope, asset_details=self.data_args)
        conn.connect()
        data = conn.read_pandas()
        if data.empty:
            raise ValueError("The dataset is empty.")
        self.processed_dataset = ProcessedDataset(data=data, tokenizer=self.tokenizer)
        self.dataset = self.processed_dataset.generate()

    def load_model(self, model_details: Any = "bert-base-uncased"):
        """
        Load the transformer model based on the orchestration scope and dataset.
        """
        self.num_labels = len(self.processed_dataset.label_mapping or {})
        # Delegate loading with num_labels
        return load_transformer(self.scope, model_details, num_labels=self.num_labels)

    def train(self, trainer_args: TrainerArguments = Field(..., description="Arguments for the trainer.")):
        """
        Train the model using the trainer arguments and dataset.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Please load the dataset before training.")
        else:
            if isinstance(self.dataset, tuple):
                trainer_args.training_data = self.dataset[0]
                trainer_args.validation_data = self.dataset[1]
            else:
                raise TypeError(f"Unable to process self.dataset with type {type(self.dataset)}. Expected a tuple of (train, test) datasets.")
            
        # Load the model
        model = self.load_model()
        
        # Create trainer runner
        trainer_runner = TrainerRunner(
            trainer_args=trainer_args,
            model=model,
            output_dir="."
        )
        
        # Invoke training
        trainer_runner.invoke()
        