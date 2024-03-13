import torch

DEVICE = torch.device('cuda')
SIZE = 512
EPOCHS = 300
LEARNING_RATE = .0001
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 1
STYLE_PATH = '../data/escher.jpg'
CONTENT_PATH = '../data/starry-night.jpg'
OUTPUT_PATH = '../output/testing1.png'

