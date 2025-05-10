import unittest

from linguafuse.cloud import ConnectionManager, Scope
from linguafuse.framework import (
    FineTuneOrchestration, 
    AwsDataArguments,
)
from linguafuse.loader.dataset import ClassificationDataset
from transformers import PreTrainedTokenizerFast

from pathlib import Path
test_path = Path(__file__).parent

SAMPLE_DATA_PATH = test_path / "example_data.csv"

class TestConnectionManager(unittest.TestCase):
    def test_local_connection(self):
        manager = ConnectionManager(scope=Scope.LOCAL, asset_details=SAMPLE_DATA_PATH)
        manager.connect()
        self.assertEqual(manager.path, SAMPLE_DATA_PATH)

class TestFineTuneOrchestration(unittest.TestCase):
    def test_return_dataset(self):
        data_args = AwsDataArguments(bucket="sample-bucket")
        orchestration = FineTuneOrchestration(data_args=data_args, scope=Scope.AWS)
        with self.assertRaises(NotImplementedError):
            orchestration._return_dataset()

class TestClassificationDataset(unittest.TestCase):
    def test_dataset_length(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
        dataset = ClassificationDataset(
            text=["sample text 1", "sample text 2"],
            labels=[0, 1],
            tokenizer=tokenizer
        )
        self.assertEqual(len(dataset), 2)

    def test_dataset_item(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
        dataset = ClassificationDataset(
            text=["sample text"],
            labels=[0],
            tokenizer=tokenizer
        )
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)

if __name__ == "__main__":
    unittest.main()
