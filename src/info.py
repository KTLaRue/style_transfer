import torch

DEVICE = torch.device('cpu')
SIZE = 512
EPOCHS = 300
LEARNING_RATE = .0001
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 1
STYLE_PATH = '../data/starry-night.jpg'
CONTENT_PATH = '../data/escher.jpg'
OUTPUT_PATH = 'output/result_E300_SW100000.png'
