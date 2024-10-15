from ml_collections import ConfigDict


param_config = ConfigDict(
    {
        "BATCH_SIZE": 64,
        "EPOCHS": 500,
        "LEARNING_RATE": 3e-3,
        "INITIAL_LR": 1e-2,
        "PEAK_LR": 4e-2,
        "END_LR": 5e-3,
        "WARMUP_EPOCHS": 15,
        "SEED": 42,
        "GDN_CLIPPING": True,
        "NORMALIZE_PROB": False,
        "NORMALIZE_ENERGY": True,
        "ZERO_MEAN": True,
        "USE_BIAS": False,
        "CS_KERNEL_SIZE": 21,
        "GDNGAUSSIAN_KERNEL_SIZE": 11,
        "GABOR_KERNEL_SIZE": 31,
        #     "N_SCALES": 4,
        #     "N_ORIENTATIONS": 16,
        "USE_GAMMA": True,
        "INIT_JH": True,
        "INIT_GABOR": True,
        "A_GABOR": True,
        "A_GDNSPATIOFREQORIENT": True,
        ## Freezing config
        "TRAIN_GDNGAMMA": False,
        "TRAIN_JH": False,
        "TRAIN_GDNCOLOR": False,
        "TRAIN_CS": False,
        "TRAIN_GDNGAUSSIAN": False,
        "TRAIN_GABOR": False,
        "TRAIN_ONLY_LAST_GDN": True,
    }
)

original_config = ConfigDict(
    {
        "BATCH_SIZE": 64,
        "EPOCHS": 500,
        "LEARNING_RATE": 3e-3,
        "INITIAL_LR": 1e-2,
        "PEAK_LR": 4e-2,
        "END_LR": 5e-3,
        "WARMUP_EPOCHS": 15,
        "SEED": 42,
        "GDN_CLIPPING": True,
        "NORMALIZE_PROB": False,
        "NORMALIZE_ENERGY": True,
        "ZERO_MEAN": True,
        "USE_BIAS": False,
        "CS_KERNEL_SIZE": 5,
        "GDNGAUSSIAN_KERNEL_SIZE": 1,
        "GABOR_KERNEL_SIZE": 5,
        #     "N_SCALES": 4,
        #     "N_ORIENTATIONS": 16,
        "USE_GAMMA": False,
        "INIT_JH": False,
        "INIT_GABOR": False,
        "A_GABOR": True,
        "A_GDNSPATIOFREQORIENT": True,
        ## Freezing config
        "TRAIN_GDNGAMMA": False,
        "TRAIN_JH": False,
        "TRAIN_GDNCOLOR": False,
        "TRAIN_CS": False,
        "TRAIN_GDNGAUSSIAN": False,
        "TRAIN_GABOR": False,
        "TRAIN_ONLY_LAST_GDN": True,
    }
)