import time
import torch
from utils import get_metrics
from data_loader import unprep, preprocess

class Trainer:
    """Trainer class for SPA-GAN."""

    def __init__(self, model, optimizer, criterion, dataloader, tokenizer):
        """Initialize the trainer.
        Args:
            model: The model to train
            optimizer: The optimizer to use for training
            dataloader: The dataloader to use for training
            tokenizer: The tokenizer to use for training
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.model.to(self.device)
        
    def encode(self, text):
        return self.tokenizer.batch_encode_plus(text, padding=True, return_tensors="pt")["input_ids"].to(self.device)


    def train_step(self):
        """Train the model for one step.
            """
        self.model.train()
        train_loss = 0.0
        for input_batch, target_batch in self.dataloader:
            # Zero the gradients
            self.optimizer.zero_grad()

            # Encode the input text and decode the target text
            input_ids = self.encode(input_batch)
            target_ids = self.encode(target_batch)

            outputs = self.model(input_ids=input_ids, decoder_input_ids=target_ids[:, :-1]).logits
            loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), target_ids[:, 1:].reshape(-1))

            # Backpropagation the gradients and update the model weights
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.dataloader)

    def evaluate(self, val_loader):
        """
        Evaluate the model on the validation set.
        Args:
            val_loader: The dataloader to use for evaluation
        Returns:
            val_loss: The loss on the validation set
            total_char_acc: The character accuracy on the validation set
            total_precision: The precision on the validation set
            total_recall: The recall on the validation set
            total_f1_score: The F1-score on the validation set
         """
        self.model.eval()
        val_loss, total_char_acc, total_precision, total_recall, total_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                # Encode the input text and decode the target text
                input_ids = self.encode(input_batch)
                target_ids = self.encode(target_batch)
                # Run the model and compute the loss
                outputs = self.model(input_ids=input_ids, decoder_input_ids=target_ids[:, :-1]).logits
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), target_ids[:, 1:].reshape(-1))

                # Decode the predicted text
                predicted_ids = outputs.argmax(dim=-1)
                predicted = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                target = self.tokenizer.batch_decode(target_ids[:, 1:], skip_special_tokens=True)

                # Evaluate the performance of the model on the current batch
                char_acc, precision, recall, f1_score = get_metrics(predicted, target)
                total_char_acc += char_acc
                total_precision += precision
                total_recall += recall
                total_f1_score += f1_score
                val_loss += loss.item()

        num_batches = len(val_loader)
        avg_char_acc = total_char_acc / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        avg_f1_score = total_f1_score / num_batches
        avg_val_loss = val_loss / num_batches

        return avg_val_loss, avg_char_acc, avg_precision, avg_recall, avg_f1_score

    def train(self, val_loader, num_epochs=10, wandb_log=False):
        """Train the model.
        Args:
            val_loader: The dataloader to use for evaluation
            num_epochs: Number of epochs to train for
            wandb_log: Whether to log the training and validation loss to WandB
        """

        # To CUDA
        self.model.to(self.device)

        # Start training
        for epoch in range(num_epochs):
            start = time.time()
            # Train the model for one epoch
            train_loss = self.train_step()
            total = time.time() - start

            # Evaluate the model on the validation set
            val_loss, char_acc, precision, recall, f1_score = self.evaluate(val_loader)
            if wandb_log:
                import wandb
                wandb.log({
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Char Accuracy": char_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score,
                })
            # Log the training and validation loss to WandB
            print(
                f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
                f"char_acc = {char_acc:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, "
                f"f1_score = {f1_score:.4f}, time taken = {total}")
        # Save the trained model
        # torch.save(self.model.state_dict(), "model.pt")

    def corrected(self, input_text):
        input_ids = self.tokenizer.encode(preprocess(input_text), return_tensors="pt").to(self.device)
        output_ids = self.model.generate(input_ids, max_length=47)
        return unprep(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))


