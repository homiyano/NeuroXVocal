import os

AD_TEXT_DIR        = r"/mnt/ssd2/Ali/processed/ad_text_processed"
CN_TEXT_DIR        = r"/mnt/ssd2/Ali/processed/cn_text_processed"
AD_CSV             = r"/mnt/ssd2/Ali/processed/ad/audio_features_ad_processed.csv"
CN_CSV             = r"/mnt/ssd2/Ali/processed/cn/audio_features_cn_processed.csv"
AD_EMBEDDING_CSV   = r"/mnt/ssd2/Ali/processed/ad/audio_embeddings_ad_processed.csv"
CN_EMBEDDING_CSV   = r"/mnt/ssd2/Ali/processed/cn/audio_embeddings_cn_processed.csv"

TEXT_EMBEDDING_MODEL   = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES      = 47
NUM_EMBEDDING_FEATURES = 768
AUDIO_CHANNELS         = 1
CUDA                   = True

BATCH_SIZE              = 8
EPOCHS                  = 200
LEARNING_RATE           = 1e-5
WEIGHT_DECAY            = 1e-4
NUM_FOLDS               = 5
SAVE_BEST_MODEL         = True
EARLY_STOPPING_PATIENCE = 10

SAVE_MODEL_PATH = r"/mnt/ssd2/Ali/results/model"
LOG_PATH        = r"/mnt/ssd2/Ali/results/training.log"
