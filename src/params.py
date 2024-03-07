import torch

DEVICE = torch.device('cuda')
EPOCHS = 300
STYLE_PATH = '../data/style-image.jpg'
STYLE_WEIGHT = 1000
CONTENT_PATH = '../data/content-image.jpg'
CONTENT_WEIGHT = 1
OUTPUT_PATH = '../output/result.png'