import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class CsiNeuralNet(nn.Module):
    """
    ### Neural network classifier for CSI.

    The neural network has 2 layers with ReLU activation functions with 2048 neurons each.
    The output layer uses the softmax activation function.

    #### Attributes:
    - `device (torch.device)`: The device on which to run the model.
    - `relu (nn.ReLU)`: The ReLU activation function.
    - `softmax (nn.Softmax)`: The softmax activation function.
    - `fc1 (nn.Linear)`: The first fully connected layer.
    - `fc2 (nn.Linear)`: The second fully connected layer.
    - `output (nn.Linear)`: The output layer.
    - `loss_fn (nn.CrossEntropyLoss)`: The loss function.
    - `optimizer (torch.optim.Adam)`: The optimizer.
    - `state_loaded (bool)`: Whether the model state has been loaded.

    #### Methods:
    - `forward(x: torch.Tensor) -> torch.Tensor`: The forward pass of the neural network.
    - `get_position_probabilities(x: torch.Tensor) -> list[float]`: Gets the probabilities of each position.
    - `train_model(train_loader: DataLoader[torch.Tuple[torch.Tensor]], num_epochs: int = 1000, log_interval: int = 10, profile: bool = False) -> dict[str, torch.Tensor]`: Trains the model on a game states dataset.
    - `test_model(test_loader: DataLoader[torch.Tuple[torch.Tensor]], model_file: str = None) -> list[int]`: Tests the model on a game states dataset.
    """

    def __init__(self, input_dim: int, device: torch.device, state_file: str = None):
        super().__init__()
        self.device = device
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, 18)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.state_loaded = False

        if state_file:
            self.load_state_dict(torch.load(state_file))
            self.state_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x if self.training else self.softmax(x)

    def get_position_probabilities(self, x: torch.Tensor) -> list[float]:
        """
        Gets the probabilities of each position.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            list[float]: The probabilities of each position.
        """
        self.eval()
        x = self(x)
        return x[0].tolist()

    def train_model(
        self,
        train_loader: DataLoader[torch.Tuple[torch.Tensor]],
        test_loader: DataLoader[torch.Tuple[torch.Tensor]],
        save_path: str,
        num_epochs: int = 1000,
        log_interval: int = 10,
    ) -> dict[str, torch.Tensor]:
        """
        Trains the model on a game states dataset.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            test_loader (DataLoader): The DataLoader for the test data.
            save_path (str): The path to save the model.
            num_epochs (int): The number of epochs to train the model.
            log_interval (int): The epoch interval at which to log the training progress.

        Returns:
            dict[str, torch.Tensor]: The state dictionary of the trained model.
        """
        start_time = time.time()
        self.to(self.device)
        print(f"Training on device: {self.device}")

        self.state_loaded = False

        loss_patience = 10
        accuracy_patience = 30
        min_delta = 1e-6

        epoch, accuracy, avg_loss = self._train(
            train_loader,
            test_loader,
            num_epochs,
            loss_patience,
            accuracy_patience,
            min_delta,
            log_interval,
            save_path,
        )

        self.state_loaded = True
        end_time = time.time() - start_time
        print(
            f"Training complete: Epochs: {epoch+1}, Accuracy: {accuracy*100:.2f}%, Loss: {avg_loss}, In {end_time:.2f}s"
        )
        return self.state_dict()

    def test_model(
        self,
        test_loader: DataLoader[torch.Tuple[torch.Tensor]],
        model_file: str = None,
    ) -> tuple[list[int], float]:
        """
        Tests the model on a game states dataset.

        Args:
            datafile (str): The path to the data file.
            model_file (str): The model file to be loaded.

        Returns:
            tuple[list[int], float]: The predictions and the accuracy of the model.
        """
        start_time = time.time()
        self.to(self.device)
        print(f"Testing on device: {self.device}")

        if not self.state_loaded:
            self.load_state_dict(torch.load(model_file))
            self.state_loaded = True

        accuracy, predictions = self._test(test_loader)

        end_time = time.time() - start_time
        print(f"Accuracy: {100 * accuracy:.2f}%, In {end_time:.2f}s")
        return predictions, accuracy

    def _train(
        self,
        train_loader: DataLoader[torch.Tuple[torch.Tensor]],
        test_loader: DataLoader[torch.Tuple[torch.Tensor]],
        num_epochs: int,
        loss_patience: int,
        accuracy_patience: int,
        min_delta: float,
        log_interval: int,
        save_filename: str,
    ) -> tuple[int, float]:
        """
        Trains the model.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            test_loader (DataLoader): The DataLoader for the test data.
            num_epochs (int): The number of epochs to train the model.
            loss_patience (int): The patience value for early stopping based on loss.
            accuracy_patience (int): The patience value for early stopping based on accuracy.
            min_delta (float): The minimum delta value for early stopping.
            save_filename (str): The filename to save the model.
            log_interval (int): The epoch interval at which to log the training progress.

        Returns:
            tuple[int, float, float]: The number of epochs, accuracy and the average loss.
        """
        start_time = time.time()
        scaler = GradScaler()
        best_loss = float("inf")
        loss_patience_counter = 0
        accuracy_patience_counter = 0
        best_accuracy = 0
        last_saved_epoch = 0
        save_interval = log_interval

        for epoch in range(num_epochs):
            self.train()
            batch_loss = 0
            batch_count = 0
            correct_predictions = 0
            total_predictions = 0

            for states_batch, targets_batch in train_loader:
                states_batch = states_batch.to(self.device, non_blocking=True)
                targets_batch = targets_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                with autocast(self.device.type):
                    outputs = self(states_batch)
                    loss = self.loss_fn(outputs, targets_batch)

                if torch.isnan(loss):
                    print(f"Epoch {epoch+1}: NaN loss encountered. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                batch_loss += loss.item()
                batch_count += 1
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets_batch).sum().item()
                total_predictions += targets_batch.size(0)

            if batch_count == 0:
                raise ValueError(
                    f"Epoch {epoch+1}: No batches processed. Exiting training."
                )

            accuracy = correct_predictions / total_predictions
            avg_loss = batch_loss / batch_count

            test_accuracy, _ = self._test(test_loader)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = self.state_dict()
                best_model_epoch = epoch + 1
                accuracy_patience_counter = 0
            else:
                accuracy_patience_counter += 1

            if (
                best_model_epoch != last_saved_epoch
                and (epoch + 1) % save_interval == 0
            ):
                torch.save(best_model, save_filename)
                print(
                    f"Model saved for epoch {best_model_epoch} with accuracy {best_accuracy*100:.2f}%"
                )
                last_saved_epoch = best_model_epoch

            if best_loss - min_delta < avg_loss <= best_loss:
                loss_patience_counter += 1
            else:
                loss_patience_counter = 0
                best_loss = avg_loss

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                end_time = time.time() - start_time
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy*100:.2f}%, Loss: {avg_loss:.8f}, In {end_time:.2f}s"
                )
                start_time = time.time()

            if accuracy_patience_counter > accuracy_patience:
                print(f"Early stopping at epoch {epoch+1} due to accuracy")
                break

            if loss_patience_counter > loss_patience:
                print(f"Early stopping at epoch {epoch+1} due to loss")
                break

        return epoch, accuracy, avg_loss

    def _test(
        self, test_loader: DataLoader[torch.Tuple[torch.Tensor]]
    ) -> tuple[float, list[int]]:
        """
        Tests the model.

        Args:
            test_loader (DataLoader): The DataLoader for the test data.

        Returns:
            tuple[float, list[int]]: The accuracy and the predictions.
        """
        self.eval()
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for states_batch, targets_batch in test_loader:
                states_batch = states_batch.to(self.device, non_blocking=True)
                targets_batch = targets_batch.to(self.device, non_blocking=True)

                with autocast(self.device.type):
                    outputs = self(states_batch)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets_batch).sum().item()
                total += targets_batch.size(0)
                predictions.extend(predicted.tolist())

        accuracy = correct / total
        return accuracy, predictions
