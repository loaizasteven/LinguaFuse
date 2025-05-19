from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum
from pathlib import PosixPath

import pandas as pd

from linguafuse import (
    aml,
    aws
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scope(Enum):
    AML = "Azure Machine Learning"
    AWS = "Amazon Web Services"
    LOCAL = "Local"


class ConnectionManager(BaseModel):
    """
    Connection manager for different cloud services.
    """

    scope: Scope = Field(
        description="The scope of the connection manager, indicating the cloud service it connects to."
    )
    asset_details: Any = Field(
        description="Connection details for the asset, which can vary based on the scope."
    )
    path: Optional[str] = None
    class Config:
        json_schema_extra = {
            "example": {
                "scope": "AML",
                "asset_details": {
                    "name": "BERT-base-uncased",
                    "version": "1.0",
                }
            }
        }
    
    def connect(self):
        """
        Establish a connection to the specified cloud service.
        """
        connection_handler = ConnectionFactory.get_connection(self.scope, self.asset_details)
        self.path = connection_handler.connect()
    
    def read_pandas(self):
        """
        Read a pandas DataFrame from the specified cloud service.
        """
        return pd.read_csv(filepath_or_buffer=self.path)


class ConnectionFactory:
    """
    Factory class to create connection handlers for different cloud services.
    """

    @staticmethod
    def get_connection(scope: Scope, asset_details: Any):
        if scope == Scope.AML:
            return AMLConnectionHandler(asset_details)
        elif scope == Scope.AWS:
            return AWSConnectionHandler(asset_details)
        elif scope == Scope.LOCAL:
            return LocalConnectionHandler(asset_details)
        else:
            raise ValueError("Unsupported scope")


class AMLConnectionHandler:
    def __init__(self, asset_details: Any):
        self.asset_details = asset_details

    def connect(self):
        logger.info(f"Connecting to Azure Machine Learning with asset: {self.asset_details}")
        return aml.connections.get_asset_path(self.asset_details)


class AWSConnectionHandler:
    def __init__(self, asset_details):
        self.asset_details = asset_details

    def connect(self):
        logger.info(f"Connecting to Amazon Web Services with asset: {self.asset_details}")
        return aws.connections.get_asset_path(self.asset_details)


class LocalConnectionHandler:
    def __init__(self, asset_details):
        self.asset_details = asset_details

    def connect(self):
        logger.info(f"Connecting locally with asset: {self.asset_details}", type(self.asset_details))
        if isinstance(self.asset_details, PosixPath):
            return self.asset_details
        else:
            return self.asset_details.path
