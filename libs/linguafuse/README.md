# LinguaFuse Modules
![](../../docs/static/Repo%20Design.png)

## Overview

LinguaFuse is a comprehensive library that provides machine learning modules designed for natural language processing workflows. These modules facilitate seamless connections to various services, efficient data handling, and integration with cloud platforms.

## Features

- **Connections Management**: Simplifies the process of connecting to various data sources and cloud services
- **Dataset Loading & Processing**: Robust utilities for loading, preprocessing, and managing datasets
- **Cloud Integration**: Seamless integration with Azure Machine Learning and AWS services
- **Scalability**: Optimized performance for handling large-scale datasets and complex workflows
- **Extensibility**: Modular design that supports adding new data sources and ML frameworks

## Module Structure

- `aml/connections.py`: Azure Machine Learning service connection management
- `aws/connections.py`: AWS service connection management and configuration
- `loader/dataset.py`: Dataset loading, preprocessing, and transformation utilities
- `loader/transformer.py`: Model transformation and artifact management utilities
- `trainer/trainer.py`: Model training utilities with configurable training loops and evaluation metrics
- `core/framework.py`: Core framework components that provide abstractions for different ML libraries

### loader/transformer.py

The `transformer.py` module handles model transformations and manages model artifacts in cloud storage. It implements a standardized approach for referencing and retrieving model snapshots.

#### Model Artifact Storage

Model artifacts in LinguaFuse follow a specific storage pattern:

1. Each model is stored in a base directory containing:
    - A `refs/` directory that maintains references to specific model versions
    - A `snapshots/` directory that contains the actual model artifacts

2. The `refs/main` file contains a reference ID (typically a commit hash or version identifier) that points to the current production model version.

3. Model snapshots are stored in the `snapshots/{reference}` directory, where `{reference}` is the identifier from the `refs/main` file.

#### Using return_model_path

The `return_model_path` function resolves the current production model path by reading the reference from `refs/main` and constructing the path to the specific snapshot. This approach enables versioned model management and easy rollbacks by simply updating the reference in the main file.

## Getting Started

Refer to the documentation in each module for specific usage examples and configuration options. Each module provides detailed API references and implementation guides.

## Contributing

Contributions to LinguaFuse are welcome! Please refer to our contribution guidelines for more information.