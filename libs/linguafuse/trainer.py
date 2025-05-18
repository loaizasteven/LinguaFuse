from pydantic import BaseModel, Field
import logging

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


class TrainerRunner(BaseModel):
    """
    Trainer runner class to manage the training process.
    """
    trainer_args: TrainerArguments = Field(..., description="Arguments for the trainer")
    model: str = Field(..., description="Model to be trained")
    dataset: str = Field(..., description="Dataset to be used for training")
    output_dir: str = Field(..., description="Directory to save the trained model")

    def invoke(self):
        """
        Invoke the training process.
        """
        # Placeholder for the training logic
        logger.info(f"Training {self.model} on {self.dataset} with batch size {self.trainer_args.batch_size}")
        
        
if __name__ == "__main__":
    # Example usage
    trainer_args = TrainerArguments(name="MyTrainer", batch_size=16, learning_rate=0.01, epochs=5)
    trainer_runner = TrainerRunner(trainer_args=trainer_args, model="MyModel", dataset="MyDataset", output_dir="./output")
    trainer_runner.invoke()