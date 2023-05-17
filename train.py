import time
import torch
from data_loader import unprep, preprocess
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class SpellingChecker:
    """SpellingChecker class"""

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

                predicted_ids = outputs.argmax(dim=-1)

                # Evaluate the performance of the model on the current batch
                char_acc, precision, recall, f1_score = self.metrics(predicted_ids.detach().cpu(),
                                                                     target_ids[:, 1:].detach().cpu())
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
            val_loss, char_acc, precision, recall, f1_scores = self.evaluate(val_loader)
            if wandb_log:
                import wandb
                wandb.log({
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Char Accuracy": char_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_scores,
                })
            # Log the training and validation loss to WandB
            print(
                f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
                f"char_acc = {char_acc:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, "
                f"f1_score = {f1_scores:.4f}, time taken = {total}")

    def corrected(self, input_text):
        input_ids = self.tokenizer.encode(preprocess(input_text), return_tensors="pt").to(self.device)
        output_ids = self.model.generate(input_ids, max_length=47)
        return unprep(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

    @torch.no_grad()
    def metrics(self, labels, target_labels):
        accuracy = sum(accuracy_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        precision = sum(precision_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        recall = sum(recall_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        f1_scores = sum(f1_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        return accuracy, precision, recall, f1_scores

# class for binay error rate
class BERT2CER:
    def __init__(self, model, optimizer, criterion, dataloader, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.classifier = torch.nn.Linear(768, 1)
        self.model.to(self.device)
        self.classifier.to(self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()

    def encode(self, text):
        return self.tokenizer.batch_encode_plus(text, padding=True, return_tensors="pt")["input_ids"].to(self.device)

    def train_step(self):
        self.model.train()
        train_loss = 0.0
        for input_batch, target_batch in self.dataloader:
            self.optimizer.zero_grad()

            # train for Bert
            input_ids = self.encode(input_batch)
            attention_mask = self.tokenizer.batch_encode_plus(input_batch, padding=True, return_tensors="pt")[
                'attention_mask'].to(self.device)
            target_ids = self.encode(target_batch)

            outputs = self.model(input_ids, attention_mask)
            hidden_states = outputs.last_hidden_state
            logits = self.classifier(hidden_states)
            logits = self.sigmoid(logits)

            # Squeeze the logits to remove the extra dimension
            logits = logits.squeeze(-1)

            # Calculate the labels for the batch
            labels = []
            for raw_tokens, corrected_tokens in zip(input_ids, target_ids):
                batch_labels = [0] * len(raw_tokens)
                for token in corrected_tokens:
                    if token in raw_tokens:
                        index = torch.where(raw_tokens == token)[0][0]
                        batch_labels[index] = 1

                labels.append(batch_labels)

            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.dataloader)

    def evaluate(self, val_loader):
        self.model.eval()

        val_loss, total_char_acc, total_precision, total_recall, total_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0
        for input_batch, target_batch in val_loader:
            input_ids = self.encode(input_batch)
            attention_mask = self.tokenizer.batch_encode_plus(input_batch, padding=True, return_tensors="pt")[
                'attention_mask'].to(self.device)
            target_ids = self.encode(target_batch)

            outputs = self.model(input_ids, attention_mask)
            hidden_states = outputs.last_hidden_state
            logits = self.classifier(hidden_states)
            logits = self.sigmoid(logits)

            # Squeeze the logits to remove the extra dimension
            logits = logits.squeeze(-1)

            # Calculate the labels for the batch
            labels = []
            for raw_tokens, corrected_tokens in zip(input_ids, target_ids):
                batch_labels = [0] * len(raw_tokens)
                for token in corrected_tokens:
                    if token in raw_tokens:
                        index = torch.where(raw_tokens == token)[0][0]
                        batch_labels[index] = 1

                labels.append(batch_labels)

            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
            loss = self.criterion(logits, labels)

            char_acc, precision, recall, f1_scores = self.metrics(logits.detach().cpu(), labels.detach().cpu())
            total_char_acc += char_acc
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_scores
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
            val_loss, char_acc, precision, recall, f1_scores = self.evaluate(val_loader)

            if wandb_log:
                import wandb
                wandb.log({
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Char Accuracy": char_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_scores,
                })
            # Log the training and validation loss to WandB
            print(
                f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
                f"char_acc = {char_acc:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, "
                f"f1_score = {f1_scores:.4f}, time taken = {total}")

    @torch.no_grad()
    def metrics(self, labels, target_labels):
        labels = [[int(x >= 0.5) for x in label] for label in labels]
        accuracy = sum(accuracy_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        precision = sum(precision_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        recall = sum(recall_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        f1_scores = sum(f1_score(label, target_label) for label, target_label in zip(labels, target_labels)) / len(labels)
        return accuracy, precision, recall, f1_scores
