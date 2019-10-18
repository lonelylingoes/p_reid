import torch

class ColorAugmentation(object):

	def __init__(self, p=0.5):
		self.p = p
		self.eig_vec = torch.Tensor([
			[0.4009, 0.7192, -0.5675],
			[-0.8140, -0.0045, -0.5808],
			[0.4203, -0.6948, -0.5836],
		])
		self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])


	def _check_input(self, tensor):
		assert tensor.dim() == 3 and tensor.size(0) == 3


	def __call__(self, tensor):
		if random.uniform(0, 1) > self.p:
			return tensor

		alpha = torch.normal(mean = torch.zeros_like(self.eig_val)) * 0.1
		quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
		tensor = tensor + quatity.view(3, 1, 1)
		return tensor