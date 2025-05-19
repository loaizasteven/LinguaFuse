import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Dict, List, Union, Tuple, Optional, Any
from pydantic import BaseModel, ConfigDict, field_validator
from linguafuse.errors import validate_columns, validate_encodings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class ClassificationDataset(Dataset):
    def __init__(self, text: List[str], labels: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_len: int = 512):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt',
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # Assuming labels are already in the correct format 
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }


class ProcessedDataset(BaseModel):
    data: pd.DataFrame
    encoding_validator = field_validator('data', mode='before')(validate_encodings)
    column_validator = field_validator('data', mode='before')(validate_columns)
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    max_len: int = 512
    text: Optional[Any] = None
    encodings: Optional[Any] = None
    labels: Optional[Any] = None
    label_mapping: Optional[Dict[int, str]] = None
    inverse_label_mapping: Optional[Dict[str, int]] = None
    training_dataset: Optional[Any] = None
    validation_dataset: Optional[Any] = None

    def model_post_init(self, context):
        self.text = self.data['text'].to_numpy()
        self.encodings = self.data['encoded_label'].to_numpy()
        self.labels = self.data['encoded_label'].to_numpy()
        
        # Create label mappings
        self.label_mapping = {int(encoded_label): str(label) for encoded_label, label in zip(self.encodings, self.labels)}    
        self.inverse_label_mapping = {value:key for key, value in self.label_mapping.items()}

    def create_data_loader(self, dataset: ClassificationDataset, batch_size: int = 32, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    def _stratified_sampling(self, data: Tuple[np.ndarray, np.ndarray], split: float = 0.2, min_sample: int = 1) -> Tuple:
        "Stratified sampling to ensure each class is represented in the train and test sets conditioned on the minimum sample size"
        
        print("Hint: Expecting 'data' to be a tuple of (text, labels)")

        # Get the unique labels and their counts
        unique_labels, counts = np.unique(data[1], return_counts=True)

        # Filter labels based on the minimum sample size
        filtered_labels = unique_labels[counts > min_sample]
        idx = np.isin(data[1], filtered_labels)
        represented_data = (data[0][idx], data[1][idx])

        # sample
        data_train, data_test, labels_train, labels_test = train_test_split(represented_data[0], represented_data[1],
                         test_size=split, stratify=represented_data[1], random_state=42)

        # append the remaining labels to the train set
        data_train = np.concatenate((data_train, data[0][~idx]))
        labels_train = np.concatenate((labels_train, data[1][~idx]))

        return data_train, data_test, labels_train, labels_test

    def generate(self, split: Optional[int] = 0.2) -> Tuple:
        """Generate datasets."""
        # Perform stratified sampling
        if split:
            data = (self.text, self.labels)
            data_train, data_test, labels_train, labels_test = self._stratified_sampling(data, split=split)

            # Create the datasets
            train_dataset = ClassificationDataset(
                text=data_train,
                labels=labels_train,
                tokenizer=self.tokenizer,
                max_len=self.max_len
            )

            test_dataset = ClassificationDataset(
                text=data_test,
                labels=labels_test,
                tokenizer=self.tokenizer,
                max_len=self.max_len
            )

            # Create the training/validation dataset
            self.training_dataset = self.create_data_loader(dataset=train_dataset)
            self.validation_dataset = self.create_data_loader(dataset=test_dataset, shuffle=False)
            
            return self.training_dataset , self.validation_dataset

    def get_training_steps(self, epochs: int) -> int:
        """Get the number of steps for training."""
        if self.training_dataset is None:
            raise ValueError("Dataset is not loaded. Please load the dataset before training.")
        else:
            return len(self.training_dataset) * epochs
        