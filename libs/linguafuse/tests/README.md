# Tests Directory

This directory contains test-related files and data for the LinguaFuse project.

## Structure

- `example_data/`: Contains example datasets used for testing purposes.
  - `sample_data.csv`: A sample dataset with encoded labels, text, and their corresponding sentiment labels (e.g., Positive, Negative, Neutral).

## Running Tests

To run the tests, ensure you have the required dependencies installed (see the main [README.md](../README.md) for installation instructions). Then, execute the following command from the root directory:

```bash
pytest tests/
```

This will discover and run all test cases within the `tests/` directory.

## Adding New Tests

1. Create a new Python file in this directory or its subdirectories.
2. Use the `pytest` framework to write your test cases.
3. Ensure your test files are named with the `test_` prefix (e.g., `test_example.py`).

For more information on writing tests, refer to the [pytest documentation](https://docs.pytest.org/).
