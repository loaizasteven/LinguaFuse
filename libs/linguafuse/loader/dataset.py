import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Dict, List, Union


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
