# Output value definitions

// This file contains the output values for the Terraform configuration
// for the AWS S3 bucket.
// The output values are used to display information about the resources
output "s3_storage_name" {
  description = "Name of the S3 bucket used to store project files."

  value = aws_s3_bucket.linguafuse_storage.id
}