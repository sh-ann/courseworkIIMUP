from src.training import train_function
import logging
import warnings

warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    train_function()
