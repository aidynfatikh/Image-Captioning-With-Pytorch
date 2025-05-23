# DATASET
IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
CAPTIONS_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

IMAGES_ZIP_PATH = "raw/Flickr8k_Dataset.zip"
CAPTIONS_ZIP_PATH = "raw/Flickr8k_text.zip"
VOCAB_PATH = "dataset/vocab.pkl"
CHECKPOINT_PATH = "checkpoints/chk.pt"

# MODEL CONFIGURATION
BATCH_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
N_EMBD = 256

LR = 1E-3
MAX_STEPS = 10000
WARMUP_STEPS = 200
MAX_LR = 1E-3
MIN_LR = 2E-4
SAVE_EVERY = 30