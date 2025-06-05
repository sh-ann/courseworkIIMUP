import logging
import mido
from sklearn.model_selection import train_test_split
import torch
import numpy as np


class MidiPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse(self, path: str) -> mido.MidiFile:
        self.logger.info(f"Parsing MIDI file: {path}")
        return mido.MidiFile(path)

    def to_music21(self, midi: mido.MidiFile):
        self.logger.info("Converting to music21 stream")
        return converter.parse(midi)


class TrainUtils:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def split_data(self, data, test_size=0.2, random_state=42):
        self.logger.info(f"Splitting data: test_size={test_size}")
        return train_test_split(data, test_size=test_size, random_state=random_state)

    def vectorize(self, sequences, labels, device):
        self.logger.info("Vectorizing sequences and labels into tensors")
        X = torch.tensor(np.stack(sequences), dtype=torch.float32, device=device)
        y = torch.tensor(np.stack(labels), dtype=torch.long, device=device)
        return X, y
