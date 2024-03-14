import torch

DEVICE = torch.device('cuda')
SIZE = 512
EPOCHS = 10
LEARNING_RATE = .0001
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 1
STYLE_PATH = '../data/starry-night.jpg'
CONTENT_PATH = '../data/escher.jpg'
OUTPUT_PATH = '../output/escher_starry-night.png'
