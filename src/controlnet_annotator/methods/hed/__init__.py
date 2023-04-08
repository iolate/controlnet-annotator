import numpy as np
import cv2
import torch
from einops import rearrange
import os

class Network(torch.nn.Module):
	def __init__(self, model_path):
		super().__init__()

		self.netVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.netCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

		self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(model_path).items()})
	# end

	def forward(self, tenInput):
		tenInput = tenInput * 255.0
		tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)

		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)

		tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
	# end
# end

# cpu = torch.device('cpu')

class Model:
	def __init__(self):
		self.network = None
	
	def load(self):
		if self.network is not None: return
		
		modeldir = os.path.dirname(os.path.realpath(__file__))
		# https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth
		modelpath = os.path.join(modeldir, 'network-bsds500.pth')
		self.network = Network(modelpath)
		self.network.eval()
	
	def apply(self, input_image):
		assert input_image.ndim == 3
		
		if self.network is None: self.load()
		
		input_image = input_image[:, :, ::-1].copy()
		with torch.no_grad():
			image_hed = torch.from_numpy(input_image).float() # .to(devices.get_device_for("controlnet"))
			image_hed = image_hed / 255.0
			image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
			edge = self.network(image_hed)[0]
			edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
			return edge[0]
	
	# fake_scribble
	def nms(self, x, t, s):
		x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

		f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
		f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
		f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
		f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

		y = np.zeros_like(x)

		for f in [f1, f2, f3, f4]:
			np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

		z = np.zeros_like(y, dtype=np.uint8)
		z[y > t] = 255
		return z