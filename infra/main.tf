# In this example, we will create an AWS S3 bucket using Terraform

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.83"
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
  prefix = "s3-linguafuse"
}

module "s3_bucket" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket = random_pet.storage.id
  tags   = {
    Name        = "s3-${random_pet.storage.id}"
    Environment = var.s3_instance_name
    Project     = "LinguaFuse"
  }
  acl    = "private"

  control_object_ownership = true
  object_ownership         = "ObjectWriter"

  versioning = {
    enabled = true
  }
}
