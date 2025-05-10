import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Dict, List, Union
from pydantic import BaseModel, ConfigDict, field_validator
from linguafuse.errors import validate_columns
import pandas as pd


class ClassificationDataset(Dataset):
    def __init__(self, text: List[str], labels: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_len: int = 512)-> Dict:
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        # Tokenize the text
        # Assuming you have a tokenizer that converts text to input_ids
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
    _validate_data = field_validator('data', mode='before')(validate_columns)
    # tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    max_len: int = 512
    
    def model_post_init(self, context):
        self.text = self.data['text'].to_numpy()
        self.encodings = self.data['encoded_label'].to_numpy()
        self.labels = self.data['encoded_label'].to_numpy()
        
        # Create label mappings
        self.label_mapping = {int(encoded_label): str(label) for encoded_label, label in zip(self.encodings, self.labels)}    
        self.invese_label_mapping = {value:key for key, value in self.label_mapping.items()}
