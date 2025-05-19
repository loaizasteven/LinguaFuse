from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from pydantic import BaseModel, Field, ConfigDict
from typing import Union, Dict, List, Any
import os
import pydantic

import uuid
import warnings
import logging

from openai import AzureOpenAI, AsyncAzureOpenAI
from http import HTTPStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_asset_path(config:BaseModel) -> Union[str, os.PathLike]:
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
    asset = ml_client.data.get(name=config.name, version=config.version)
    return asset.path


class OpenAIClient(BaseModel):
    api_key: str = Field(
        default=os.getenv("AZURE_OPENAI_API_KEY"),
        description="The API key for Azure OpenAI."
    )
    api_version: str = Field(
        default=os.getenv("AZURE_OPENAI_API_VERSION"),
        description="The API version for Azure OpenAI."
    )
    deployment_name: str = Field(
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        description="The deployment name for Azure OpenAI."
    )
    model_name: str = Field(
        default=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        description="The model name for Azure OpenAI."
    )
    session_id: str = Field(
        default=str(uuid.uuid4()),
        description="The session ID for Azure OpenAI."
    )
    client: Union[AzureOpenAI, AsyncAzureOpenAI] = Field(..., description="The OpenAI client.")
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries for API calls."
    )
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def model_post_init(self, context):
        if not self.client:
            logger.info("Initializing OpenAI client...")
            self.environment_valiation()
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.deployment_name,
                max_retries=self.max_retries,
            )
        return super().model_post_init(context)
    
    def environment_validation(self):
        """
        Validate the environment variables for the OpenAI client.
        
        Raises:
            ValueError: If any of the required environment variables are not set.
        """
        ENVIRONMENT_VARIABLES = (
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_API_VERSION"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            os.getenv("AZURE_OPENAI_MODEL_NAME"),
        )
        if not all(ENVIRONMENT_VARIABLES):
            raise ValueError(
                "One or more environment variables are not set. "
                "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, "
                "AZURE_OPENAI_DEPLOYMENT_NAME, and AZURE_OPENAI_MODEL_NAME."
            )

    async def ainvoke(self, messages:List, temperature:int = 1) -> Any:
        """
        Asynchronous method to invoke the OpenAI API.
        
        Args:
            messages (List): The messages to send to the OpenAI API.
            temperature (int): The temperature for the OpenAI API.
    
        """
        logger.info("Close default connection to OpenAI API.")
        self.client.close()
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.deployment_name,
            max_retries=self.max_retries,
        )
        try:
            response = await self.client.chat.completions.create(
                deployment_id=self.deployment_name,
                messages=messages,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Error invoking OpenAI API: {e}")
            return HTTPStatus.INTERNAL_SERVER_ERROR
    
    def invoke(self, messages:List, response_format: Dict, temperature:int = 1) -> Any:
        """
        Synchronous method to invoke the OpenAI API.
        
        Args:
            messages (List): The messages to send to the OpenAI API.
            response_format (Dict): The format of the response from the OpenAI API.
            temperature (int): The temperature for the OpenAI API.
    
        """
        if response_format:
            warnings.warn(
                "Structure output is not supported in this version. See https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure for more details.",
            )
        try:
            response = self.client.chat.completions.create(
                deployment_id=self.deployment_name,
                messages=messages,
                temperature=temprature
            )
            return response
        except Exception as e:
            logger.error(f"Error invoking OpenAI API: {e}")
            return HTTPStatus.INTERNAL_SERVER_ERROR
        

if __name__ == "__main__":
    import asyncio

    async def coroutine_llm_call(prompt):
        """
        Asynchronous function to call the OpenAI API.

        If the user wants to run this script in a jupyter notebook, it will raise an exception due to the
        existing event loop preventing async.run() from being called.

        try:
        ```python
        import nest_asyncio
        nest_asyncio.apply()
        ```
        """
        client = OpenAIClient()
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        response = await client.ainvoke(messages=messages)
        return response
    
    asyncio.run(coroutine_llm_call(prompt="Hello, how are you?"))
