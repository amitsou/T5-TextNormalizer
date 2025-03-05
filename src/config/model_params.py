""" Model hyperparameters and configuration. """


class ModelParams:
    """
    A class used to represent the parameters for a machine learning model.

    Attributes:
    MODEL_NAME (str):
        The name of the model to be used.
    MAX_LENGTH (int):
        The maximum length of the input sequences (reduce for GPU memory).
    BATCH_SIZE (int):
        The number of samples per batch (to fit a 6GB GPU RTX3060 set it to 4).
    NUM_EPOCHS (int):
        The number of epochs for training.
    LEARNING_RATE (float):
        The learning rate for the optimizer.
    WEIGHT_DECAY (float):
        The weight decay (L2 penalty) for the optimizer.
    TEST_SIZE (float):
        The size of the test set (0.1 for 10%).
    VALIDATION_SIZE (float):
        The size of the validation set (0.1 for 10%).
    """

    MODEL_NAME = "t5-small"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_WORKERS = 8

    # #Dataset splits (80/10/10)
    TEST_SIZE = 0.1  # #10% for testing
    VALIDATION_SIZE = 0.1  # #10% for validation
