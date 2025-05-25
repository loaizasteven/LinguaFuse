# In this example, we will create an AWS S3 bucket using Terraform

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "us-west-2"
}

resource "random_pet" "storage" {
  keepers = {
    # Generate a new pet name each time we switch to a new s3 id
    s3_id = var.s3_instance_name
  }
}
