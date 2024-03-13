from PIL import Image
import torch
import torchvision.transforms as T
from info import DEVICE, SIZE

# load transform to get square image
load_transform = T.Compose([
				 T.Resize(SIZE),
				 T.CenterCrop(SIZE),
				 T.ToTensor()
				])

# get tensor to PIL image
saver = T.ToPILImage()


# move image to tensor
def load_image(path):
	image = load_transform(Image.open(path)).unsqueeze(0)
	return image.to(DEVICE, torch.float)

# save image to desired path
def save_image(tensor, path):
	image = saver(tensor.cpu().clone().squeeze(0))
	image.save(path)