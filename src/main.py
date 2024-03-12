import model
import utils
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from info import DEVICE, EPOCHS, STYLE_PATH, CONTENT_PATH, OUTPUT_PATH, STYLE_WEIGHT, CONTENT_WEIGHT, LEARNING_RATE

style_image = utils.load_image(STYLE_PATH)
content_image = utils.load_image(CONTENT_PATH)
print("loaded images") # this is not even running

# pretrained VGG
cnn = models.vgg19(models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
#changed to match mean and std of imagenet and pytorch website
norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
norm_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

# duplicate our content image for target
target_image = content_image.clone()

# modify model
model, style_losses, content_losses = model.style_model(cnn, DEVICE, norm_mean, norm_std, style_image, content_image)

# Optimization algorithm can we use adam? does it matter?
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Run style transfer
for i in range(EPOCHS):
	# Keep target values between 0 and 1
	target_image.data.clamp_(0, 1)

	#reset gradients
	optimizer.zero_grad()
	
	#run image through model stack
	model(target_image)
	
	#recalculate style/content score
	style_score = 0
	content_score = 0

	#check if these are being updated each epoch
	for s1 in style_losses:
		style_score += s1.loss
	for c1 in content_losses:
		content_score += c1.loss

	style_score *= STYLE_WEIGHT
	content_score *= CONTENT_WEIGHT

	loss = style_score + content_score
	loss.backward()

	if i % 10 == 0:
		print(f"Run: {i}, Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
	# print("combined loss", style_score + content_score)

#clamp any extra values that are too big or small
target_image.data.clamp_(0, 1)

utils.save_image(target_image, OUTPUT_PATH)