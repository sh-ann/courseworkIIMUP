from src.utils import TrainUtils
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.generators import (
    LSTMGenerator,
    RNNGenerator,
    GRUGenerator,
    TransformerGenerator,
    LinearModel,
)
import os
from src.preprocessing import load_midi_files
from sklearn.model_selection import train_test_split
import mido
import numpy as np
from tqdm import tqdm

PURE_FILE_DIR = "/Users/annasalamova/Desktop/good ones/Children"


def train(model, criterion, dataloader, val_loader, epochs, lr, device):
    """
    Train the given model for a specified number of epochs using the provided data loader and criterion.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): Loss function used to compute training loss.
        dataloader (DataLoader): DataLoader providing training batches (inputs and targets).
        val_loader (DataLoader): DataLoader providing validation batches (currently unused in this function).
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        nn.Module: The trained model after completing all epochs.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1)):
        logging.info(f"Epoch {epoch}/{epochs}")
        model.train()
        total_loss = 0
        val_loss = 0
        val_correct = 0
        val_total = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in tqdm(dataloader, ):
            optimizer.zero_grad()
            outputs = model(X_batch)
            if model.__class__.__name__ == LSTMGenerator.__name__:
                loss = criterion(outputs, y_batch.float())
            else:
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if len(outputs.shape) > 1:
                _, preds = torch.max(outputs, 1)
            else:
                preds = outputs
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        for X_batch, y_batch in tqdm(val_loader):
            with torch.no_grad():
                outputs = model(X_batch)
            if model.__class__.__name__ == LSTMGenerator.__name__:
                loss = criterion(outputs, y_batch.float())
            else:
                loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            if len(outputs.shape) > 1:
                _, preds = torch.max(outputs, 1)
            else:
                preds = outputs
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

        logging.info(f"Train loss: {total_loss / len(dataloader):.8f}")
        logging.info(f"Train acc: {(train_correct / train_total):.8f}")
        logging.info(f"Val loss: {val_loss / len(val_loader):.8f}")
        logging.info(f"Val acc: {(val_correct / val_total):.8f}")
    return model


def train_model(model, criterion, train_loader, val_loader):
    """
    Set up the training environment, train the model for a single epoch, save the trained model, and return training history.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): Loss function used to compute training loss.
        train_loader (DataLoader): DataLoader providing training batches.
        val_loader (DataLoader): DataLoader providing validation batches.

    Returns:
        tuple:
            - nn.Module: The trained model after saving.
            - dict: A dictionary containing empty lists for 'train_loss' and 'val_loss' (placeholders for future tracking).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    lr = 1e-3

    model = train(model, criterion, train_loader, val_loader, epochs, lr, device)

    os.makedirs("./models", exist_ok=True)
    torch.save(model, f"./models/final_model_{model.__class__.__name__}.pt")

    return model, {"train_loss": [], "val_loss": []}


def prepare_training_validation_data(midi_data, label, sequence_length=32):
    """
    Convert a list of MIDI files into fixed-length sequences of note and time features, paired with a uniform label.

    Args:
        midi_data (list of MidiFile): A list of loaded MIDI file objects (from mido.MidiFile).
        label (int): The target label to assign to all generated sequences (e.g., 1 for “pure” class).
        sequence_length (int, optional): Number of consecutive MIDI events to include in each sequence. Default is 32.

    Returns:
        tuple:
            - np.ndarray: Array of shape (num_sequences, sequence_length, 2) containing the features [note, time] for each event.
            - np.ndarray: Array of shape (num_sequences,) containing the integer label for each sequence.
    """
    sequences, labels = [], []

    for midi_file_data in midi_data:
        midi_events = [event for track in midi_file_data.tracks for event in track]

        for start_index in range(len(midi_events) - sequence_length):
            sequence = midi_events[start_index : start_index + sequence_length]
            sequence_features = [
                [event.note if hasattr(event, "note") else 0, event.time]
                for event in sequence
            ]
            sequences.append(sequence_features)
            labels.append(label)

    return np.array(sequences), np.array(labels)


def train_function():
    """
    Load MIDI files from a specified directory, preprocess them into training and validation sets,
    vectorize the feature sequences and labels, define and train multiple model architectures sequentially,
    and return the last trained model along with its training history.

    This function performs the following steps:
        1. Loads MIDI files from a given directory path.
        2. Splits the file list into training and validation subsets.
        3. Converts each MIDI file into fixed-length sequences of note/time features.
        4. Uses TrainUtils to vectorize sequences and labels into PyTorch tensors.
        5. Constructs DataLoader objects for training and validation.
        6. Trains the following models in sequence: LSTMGenerator, RNNGenerator, GRUGenerator, TransformerGenerator, and LinearModel.
        7. Saves each trained model to disk under "./models".
        8. Returns the final model trained (LinearModel) and its history dictionary.

    Returns:
        tuple:
            - nn.Module: The last model trained (LinearModel) after saving to disk.
            - dict: A dictionary containing lists for 'train_loss' and 'val_loss' (history placeholders).
    """
    pure_file_directory = PURE_FILE_DIR
    pure_midi_files = load_midi_files(pure_file_directory)

    train_files, val_files = train_test_split(
        pure_midi_files, test_size=0.2, random_state=42
    )

    train_pure = [mido.MidiFile(file) for file in train_files]
    val_pure = [mido.MidiFile(file) for file in val_files]

    train_sequences, train_labels = prepare_training_validation_data(train_pure, 1)
    val_sequences, val_labels = prepare_training_validation_data(val_pure, 1)

    input_shape = train_sequences.shape[1:]
    num_classes = np.max(train_labels) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tu = TrainUtils()

    if np.min(train_labels) == 1:
        train_labels = train_labels - 1
        val_labels = val_labels - 1

    X_train, y_train = tu.vectorize(train_sequences, train_labels, device)
    X_val, y_val = tu.vectorize(val_sequences, val_labels, device)

    input_dim = X_train.size(-1)
    output_dim = max(int(torch.max(y_train).item()) + 1, num_classes)

    batch_size = 64

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # LSTM
    # logging.info("Training LSTM model")
    # model = LSTMGenerator(input_dim, 128, 2, output_dim)
    # criterion = nn.BCELoss()
    # model, history = train_model(model, criterion, train_loader, val_loader)

    # RNN
    logging.info("Training RNN model")
    model = RNNGenerator(input_dim, 128, 1, output_dim)
    criterion = nn.CrossEntropyLoss()
    model, history = train_model(model, criterion, train_loader, val_loader)

    # GRU
    logging.info("Training GRU model")
    model = GRUGenerator(input_dim, 128, 1, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    model, history = train_model(model, criterion, train_loader, val_loader)

    # Transformer
    logging.info("Training Transformer model")
    model = TransformerGenerator(
        input_dim=input_dim,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        output_dim=num_classes,
        d_model=64,
    )
    criterion = nn.CrossEntropyLoss()
    model, history = train_model(model, criterion, train_loader, val_loader)

    # Linear
    logging.info("Training Linear model")
    model = LinearModel(input_dim=64, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    model, history = train_model(model, criterion, train_loader, val_loader)

    return model, history


def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
