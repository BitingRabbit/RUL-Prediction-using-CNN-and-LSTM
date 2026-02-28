FILENAME: str = "datasets/N-CMAPSS_DS01-005.h5"
CHECKPOINT_PATH: str = "trained_models/network_checkpoint4.1.pth"
AUTOSAVE_EVERY: int = 2000  # batches
EPOCHS: int = 20
LEARNING_RATE: float = 1e-5
BATCH_SIZE: int = 64
WIN_LEN: int = 10 # window size
LOAD_MODEL: bool = False # load an existing, already trained model