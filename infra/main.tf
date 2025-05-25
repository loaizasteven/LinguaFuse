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

resource "aws_s3_bucket" "linguafuse_storage"{
  bucket = random_pet.storage.id
  tags   = {
    Name        = "s3-${random_pet.storage.id}"
    Environment = var.s3_instance_name
    Project     = "LinguaFuse"
  }

}

// Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "linguafuse_storage" {
  bucket = aws_s3_bucket.linguafuse_storage.id

  versioning_configuration {
    status = "Enabled"
  }
}

// Enable object ownership for the S3 bucket
resource "aws_s3_bucket_ownership_controls" "linguafuse_storage" {
  bucket = aws_s3_bucket.linguafuse_storage.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

// Set the ownership controls for the S3 bucket
resource "aws_s3_bucket_acl" "linguafuse_storage" {
  depends_on = [aws_s3_bucket_ownership_controls.linguafuse_storage]

  bucket = aws_s3_bucket.linguafuse_storage.id
  // The canned_acl is used to set the access control list for the bucket
  // private means that only the bucket owner has access to the objects in the bucket
  acl    = "private"
}
