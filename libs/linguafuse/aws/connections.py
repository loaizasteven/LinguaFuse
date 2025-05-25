from typing import Union
import os
from pathlib import Path

import subprocess
import json

file_dir = Path(__file__).resolve(strict=True).parent

# TODO: This function is a placeholder and should be implemented in the future for AWS S3 bucket connectivity.
def get_s3_bucket_name(terraform_dir: str = file_dir.parents[2] / "infra", key="s3_storage_name") -> str:
    """Get the S3 bucket name output from Terraform using subprocess."""
    result = subprocess.run([
        "terraform", "output", f"-json"
    ], cwd=terraform_dir, capture_output=True, text=True, check=True)
    outputs = json.loads(result.stdout)
    return outputs[f'{key}']['value']

def get_asset_path(*args, **kwargs) -> Union[str, os.PathLike, NotImplementedError]:
    raise NotImplementedError("This function is not implemented yet.")
