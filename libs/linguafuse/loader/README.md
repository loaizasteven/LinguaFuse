"""
This directory contains modules and classes for handling datasets used in the project.

dataset.py
----------
This module provides classes and methods for loading, parsing, and managing datasets from CSV files.

CSV Format Requirements:
- The CSV files must be UTF-8 encoded.
- The first row should contain the following column headers: `text`, `encoded_label`, and `label`.
- Each subsequent row represents a single data record.
- `text`: The input text data (string).
- `encoded_label`: The encoded representation of the label (integer or string, depending on implementation).
- `label`: The human-readable label (string).
- All three columns are required for each record. Missing or malformed data may result in errors or be handled according to the implementation.

Classes and Methods:
- Each class in `dataset.py` is responsible for representing a dataset or a data record.
- Methods are provided for loading data from CSV files, validating the data, and converting it into appropriate data structures for further processing.

Please refer to the docstrings of individual classes and methods in `dataset.py` for detailed information about the expected CSV schema and usage examples.
"""