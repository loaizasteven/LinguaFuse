from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from typing import Union
import os

def get_weights_path(name:str, version:str) -> Union[str, os.PathLike]:
    """
    Get the path to the weights file for a given model version.
    
    Args:
        name (str): The name of the data asset.
        version (str): The version of the data asset.
    
    Returns:
        str: The path to the weights file.
    """
    # Initialize MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
    )

    # Get the weights file path
    asset = ml_client.data.get(name=name, version=version)
    return asset.path
