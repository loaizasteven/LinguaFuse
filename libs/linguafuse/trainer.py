from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
import logging

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import AdamW
import torch

from transformers import (
    PreTrainedModel,
    TrainerCallback
)
from transformers.trainer_callback import EarlyStoppingCallback

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainerControl(BaseModel):
    should_training_stop: bool = Field(False, description="Flag to control training stop")


class TrainerState(BaseModel):
    best_metric: Optional[float] = Field(None, description="Best metric achieved during training")


class TrainerArguments(BaseModel):
    """
    Trainer class to manage training configurations.
    """
    name: str = Field(..., description="Name of the trainer")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate for optimizer")
    epochs: int = Field(10, description="Number of epochs to train")
    save_model: bool = Field(True, description="Flag to save the model after training")
    save_strategy: str = Field("steps", description="Strategy to save the model")
    eval_strategy: str = Field("epoch", description="Evaluation strategy to use during training")
    eval_steps: int = Field(500, description="Number of steps between evaluations")
    metric_for_best_model: str = Field("eval_loss", description="Metric to use for best model selection")
    load_best_model_at_end: bool = Field(True, description="Flag to load the best model at the end of training")
    training_data: Optional[DataLoader] = Field(..., description="Training data loader")
    validation_data: Optional[DataLoader] = Field(..., description="Validation data loader")
    optimizer: Optimizer = Field(default_factory=lambda: AdamW(params=[], lr=0.001), description="Optimizer to use for training")
    scheduler: LRScheduler = Field("linear", description="Learning rate scheduler to use")
    callbacks: Optional[List[TrainerCallback]] = Field(default_factory=lambda: [EarlyStoppingCallback(early_stopping_patience=3)], description="List of callbacks to use during training")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

class CallbackHandler(BaseModel):
    callbacks: List[TrainerCallback] = Field(default_factory=list, description="List of callbacks to handle during training")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def on_train_begin(self, args: TrainerArguments, state: TrainerState, control: TrainerControl):
        """
        Called at the beginning of training.
        """
        logger.info("Training started.")
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)
    
    def on_evaluate(self, args: TrainerArguments, state: TrainerState, control: TrainerControl, metrics):
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)
    
    def call_event(self, event, args, state, control, **kwargs):
        """
        Call the appropriate callback event.
        """
        for callback in self.callbacks:
            result = getattr(callback, event)(args, state, control, **kwargs)
            # A Callback can skip the return of the control object if it does not change it
            if result is not None:
                control = result
        return control

def save_best_model(args: TrainerArguments, state: TrainerState, control: TrainerControl, metrics:dict, save_path:str, strategy:str, model:PreTrainedModel) -> bool:
    """
    Save the best model based on the specified metric.
    """
    if args.save_model and args.save_strategy == strategy and (state.best_metric is None or metrics[args.metric_for_best_model] < state.best_metric):
        state.best_metric = metrics[args.metric_for_best_model]
        logger.info(f"Saving best model to {save_path}")
        # Placeholder for actual model saving logic
        full_save_path = f"{save_path}/best_model_PLACEHOLDER"
        torch.save(model.state_dict(), f"{full_save_path}.pth")
        torch.save(model, f"{full_save_path}.bin")
        logger.info(f"New best model saved at {full_save_path} with metric {state.best_metric} = {state.best_metric:.4f}")
        
        return True
    else:
        logger.info("No improvement in metric, not saving the model.")
        return False

def load_best_model(args: TrainerArguments, state: TrainerState, control: TrainerControl, load_path:str, model:PreTrainedModel):
    """
    Load the best model from the specified path.
    """
    # Placeholder for actual model loading logic
    if args.load_best_model_at_end or control.should_training_stop:
        logger.info(f"Loading best model from {load_path}")
        model.load_state_dict(torch.load(f"{load_path}/best_model_PLACEHOLDER.pth"))
        logger.info("Best model loaded successfully.")


class EvaluationStrategy(BaseModel):
    """
    Evaluation strategy class to manage evaluation configurations.
    """
    args: TrainerArguments = Field(..., description="Arguments for the trainer")
    state: TrainerState = Field(..., description="State of the trainer")
    control: TrainerControl = Field(..., description="Control object for the trainer")
    model: PreTrainedModel = Field(..., description="Model to be trained")
    metrics: dict = Field(..., description="Metrics to be used for evaluation")
    callback_handler: CallbackHandler = Field(..., description="Callback handler for the trainer")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def model_post_init(self, **kwargs):
        self.metrics = {"epoch": [], "train_loss": [], "eval_loss": []} if self.metrics is None else self.metrics
    
    def generate_metrics(self, metrics: dict):
        """
        Generate metrics for the current round.
        """
        self.metrics["epoch"] = (torch.max(torch.tensor(metrics["epoch"])) + 1).item() if metrics["epoch"] else 0
        self.metrics["train_loss"].append(metrics.get("train_loss", 0))
        self.metrics["eval_loss"].append(metrics.get("eval_loss", 0))

    def evaluation_handler(self, loss:float):
        callback_metrics = {self.args.metric_for_best_model: loss}
        self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=callback_metrics)

    def evaluate(self, stage:str, round: int) -> dict:
        """
        Predict method to evaluate the model.
        """
        if self.args.eval_strategy == stage and round % self.args.eval_steps == 0:
            self.model.eval()
            total_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(self.args.validation_data, desc="Evaluating", position=0):
                    input_ids = batch["input_ids"].long()
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    # Placeholder for actual prediction logic
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else:
            logger.debug(f"Skipping evaluation at step {round} as per strategy.")

        # Generate metrics
        total_loss /= len(self.args.validation_data)
        logger.info(f"Evaluation loss: {total_loss:.4f}")
        self.generate_metrics({"train_loss": total_loss, "eval_loss": total_loss})
        self.evaluation_handler(loss=total_loss)

        return self.metrics
    

class TrainerRunner(BaseModel):
    """
    Trainer runner class to manage the training process.
    """
    trainer_args: TrainerArguments = Field(..., description="Arguments for the trainer")
    model: PreTrainedModel = Field(..., description="Model to be trained")
    output_dir: str = Field(..., description="Directory to save the trained model")
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    state: Optional[TrainerState] = Field(default=TrainerState(), description="State of the trainer")
    control: Optional[TrainerControl] = Field(default=TrainerControl(), description="Control object for the trainer")
    callbacks: Optional[List[TrainerCallback]] = Field(default=None, description="List of callbacks to use during training")
    eval_strategy: Optional[EvaluationStrategy] = Field(default=None, description="Evaluation strategy to use during training")
    
    def invoke(self):
        """
        Invoke the training process.
        """        
        if self.trainer_args.eval_strategy is not None and self.trainer_args.validation_data is None:
            raise ValueError(f"You have set an evaluation strategy == {self.trainer_args.eval_strategy} but have not provided validation data.")
        
        # Initialize CallbackHandler
        callback_handler = CallbackHandler(callbacks=self.trainer_args.callbacks)
        callback_handler.on_train_begin(self.trainer_args, TrainerState(), TrainerControl())

        # Initialize EvaluationStrategy
        self.eval_strategy = EvaluationStrategy(args=self.trainer_args, state=self.state, control=self.control, model=self.model, callback_handler=callback_handler)

        for epoch in range(self.trainer_args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.trainer_args.epochs}")
            train_loss = self._train_step_iterator()
            logger.info(f"Epoch {epoch + 1} completed with Training loss: {train_loss}")
            
            if self.control.should_training_stop:
                logger.info(f"Training stopped by callback. Ending training at epoch {epoch}.")
                break
            # Checkpointing and evaluation
            metrics = self.eval_strategy.evaluate(stage='epoch', round=epoch)
            save_best_model(self.trainer_args, self.state, self.control, metrics, self.output_dir, "epoch", self.model)
            load_best_model(self.trainer_args, self.state, self.control, self.output_dir, self.model)

    def _train_step_iterator(self) -> float:
        """
        Step iterator for the training process, Trains a single epoch and returns the loss.
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
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Model checkpointing
                metrics = self.eval_strategy.evaluate(stage='steps', round=batch_idx)
                load_best_model(self.trainer_args, self.state, self.control, self.output_dir, self.model)
                save_best_model(self.trainer_args, self.state, self.control, metrics, self.output_dir, "steps", self.model)
            except Exception as e:
                logger.error(f"Error during training step: {e}")
                continue
        return total_loss / len(self.trainer_args.training_data)

if __name__ == "__main__":
    # Example usage
    trainer_args = TrainerArguments(name="MyTrainer", batch_size=16, learning_rate=0.01, epochs=5)
    trainer_runner = TrainerRunner(trainer_args=trainer_args, model="MyModel", dataset="MyDataset", output_dir="./output")
    trainer_runner.invoke()