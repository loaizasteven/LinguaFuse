from pydantic import BaseModel, Field
from typing import Any
from enum import Enum

from linguafuse import (
    aml,
    aws
)


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

    class Config:
        schema_extra = {
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
        return connection_handler.connect()


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
        print(f"Connecting to Azure Machine Learning with asset: {self.asset_details}")
        return aml.connection.get_weights_path(**self.asset_details)


class AWSConnectionHandler:
    def __init__(self, asset_details):
        self.asset_details = asset_details

    def connect(self):
        print(f"Connecting to Amazon Web Services with asset: {self.asset_details}")
        return aws.connection.get_weights_path(**self.asset_details)


class LocalConnectionHandler:
    def __init__(self, asset_details):
        self.asset_details = asset_details

    def connect(self):
        print(f"Connecting locally with asset: {self.asset_details}")
        return self.asset_details
