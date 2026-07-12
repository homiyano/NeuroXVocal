import os

# Paths
BASE_DIR = 'path/to/processed_data/'
AD_TEXT_DIR = 'path/to/processed_data/ad/'
CN_TEXT_DIR = 'path/to/processed_data/cn/'
AD_CSV = 'path/to/processed_data/ad/audio_features_ad.csv'
CN_CSV = 'path/to/processed_data/cn/audio_features_cn.csv'
AD_EMBEDDING_CSV = 'path/to/processed_data/ad/audio_embeddings_ad.csv'
CN_EMBEDDING_CSV = 'path/to/processed_data/cn/audio_embeddings_cn.csv'

# Model configuration
TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES = 47
NUM_EMBEDDING_FEATURES = 768 
AUDIO_CHANNELS = 1
CUDA = True

# Training parameters
BATCH_SIZE = 8               # author-indicated range [4, 8]; Zhang et al. (2021) for few-sample fine-tuning
EPOCHS = 200                 # stated in paper (Ntampakis et al., 2025)
LEARNING_RATE = 1e-5         # author-indicated range [1e-3, 1e-4, 1e-5]; He et al. (2021) DeBERTa fine-tuning
WEIGHT_DECAY = 1e-4          # hardcoded in train.py; within author-indicated range [1e-2, 1e-3, 1e-4]
NUM_FOLDS = 5                # stated in paper (Ntampakis et al., 2025)
SAVE_BEST_MODEL = True

# Early stopping criteria
EARLY_STOPPING_PATIENCE = 10  # not disclosed by authors; upper bound of common range (ICACS 2024)

# Saving paths
SAVE_MODEL_PATH = 'path/to/results/folder/model' #Create a folder "results" for saving the model
LOG_PATH = 'path/to/results/folder/training.log' #Create a folder "results" for saving logs


