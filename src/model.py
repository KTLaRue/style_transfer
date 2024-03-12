import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Content Loss Layer - helps manage data for image content
class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

# Style Loss Layer - helps manage data for image content
class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

# this defines the gram matrix as described in paper:
# inner product between vectorized feature map i and j per layer (think this equation 4)
def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a* b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)

# Normalization Layer to transform input images - based on tensorflow normalization practices
class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = mean.clone().detach()
		self.std = std.clone().detach()

	def forward(self, image):
		return (image - self.mean) / self.std

# modify  model with content/style loss layers
def style_model(cnn, device, normalization_mean, normalization_std, style_image, content_image):
	# track added loss layers
	style_losses = []
	content_losses = []
	
	# Copy network for recration
	cnn = copy.deepcopy(cnn)

	# add normalization to our sequence block first
	model = nn.Sequential(Normalization(normalization_mean, normalization_std).to(device))

	# Keep track of conv layers for naming
	i = 0
	# Loop through vgg layers - this has to be the most ridicouldous method name and im still not sure it is correct
	for layer in cnn.children():
		# Add layer to our sequence block
		model.append(layer)
     
		#identitfy if conv layer
		if isinstance(layer, nn.Conv2d):
			i += 1
			# Insert style loss layer after first 5 conv layers
			if i <= 5:
				target_feature = model(style_image).detach() #not really sure about the detach - is it needed?
				model.append(StyleLoss(target_feature))
				style_losses.append(StyleLoss(target_feature))

				# Insert content loss layer after conv 4
				if i == 4:
					target = model(content_image).detach()
					model.append(ContentLoss(target))
					content_losses.append(ContentLoss(target))

	return model, style_losses, content_losses