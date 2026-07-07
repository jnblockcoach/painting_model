import os
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
    except ImportError:
        DEVICE = torch.device("cpu")

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")

IMG_SIZE = 32
CHANNELS = 3
Z_DIM = 100
G_FEATURES = 128
D_FEATURES = 32

BATCH_SIZE = 128
EPOCHS = 100
G_LR = 2e-4
D_LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999

N_CRITIC = 1
SAMPLE_INTERVAL = 1
CKPT_INTERVAL = 10
