import os

import torch

# print("torch:", torch.__version__)
# if torch.__version__ != '0.1.12_2':
#     print("Cuda:", torch.backends.cudnn.cuda)
#     print("CuDNN:", torch.backends.cudnn.version())

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')

CKPT_DIR = os.path.join(BASE_DIR, 'save')

