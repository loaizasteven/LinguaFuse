from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import logging

from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainerArguments(BaseModel):
    """
    Trainer class to manage training configurations.
    """
    name: str = Field(..., description="Name of the trainer")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate for optimizer")
    epochs: int = Field(10, description="Number of epochs to train")
    save_model: bool = Field(True, description="Flag to save the model after training")
    evaluation_strategy: str = Field("epoch", description="Evaluation strategy to use during training")
    training_data: Optional[DataLoader] = Field(..., description="Training data loader")
    validation_data: Optional[DataLoader] = Field(..., description="Validation data loader")
    optimizer: Optimizer = Field("adam", description="Optimizer to use for training")
    scheduler: LRScheduler = Field("linear", description="Learning rate scheduler to use")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

class TrainerRunner(BaseModel):
    """
    Trainer runner class to manage the training process.
    """
    trainer_args: TrainerArguments = Field(..., description="Arguments for the trainer")
    model: PreTrainedModel = Field(..., description="Model to be trained")
    output_dir: str = Field(..., description="Directory to save the trained model")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def invoke(self):
        """
        Invoke the training process.
        """        
        if self.trainer_args.evaluation_strategy is not None and self.trainer_args.validation_data is None:
            raise ValueError(f"You have set an evaluation strategy == {self.trainer_args.evaluation_strategy} but have not provided validation data.")
        
        for epoch in range(self.trainer_args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.trainer_args.epochs}")
            train_loss = self._train_step_iterator()
            logger.info(f"Epoch {epoch + 1} completed with Training loss: {train_loss}")

    def _train_step_iterator(self) -> float:
        """
        Step iterator for the training process, Trians a single epoch and returns the loss.
        """
        # Placeholder for step iteration logic
        logger.info("Starting step iteration...")

        # Instantiate the model and parameters
        self.model.train()
        total_loss = 0.0
        batch_idx = 0
        progress_bar = tqdm(iterable=self.trainer_args.training_data, desc="Training", position=0)
        
        for batch in progress_bar:
            batch_idx += 1
            input_ids = batch["input_ids"].long()
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Simulate training step
            try:
                # Reset gradients at each batch
                self.trainer_args.optimizer.zero_grad()
                
                # Forward pass and compute loss
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass and optimization step
                # Stabilize training by ensuring gradients are not too large (limit to 1.0) [exploding gradients]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.trainer_args.optimizer.step()
                self.trainer_args.scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({"loss": f"loss.item():.4f"})

                # Model checkpointing
                pass
            except Exception as e:
                logger.error(f"Error during training step: {e}")
                continue
        return total_loss / len(self.trainer_args.training_data)

if __name__ == "__main__":
    # Example usage
    trainer_args = TrainerArguments(name="MyTrainer", batch_size=16, learning_rate=0.01, epochs=5)
    trainer_runner = TrainerRunner(trainer_args=trainer_args, model="MyModel", dataset="MyDataset", output_dir="./output")
    trainer_runner.invoke()