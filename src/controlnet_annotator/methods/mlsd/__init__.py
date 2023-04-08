import cv2
import numpy as np
import torch
import os

from .mbv2_mlsd_large import MobileV2_MLSD_Large
from .util import pred_lines

class Model:
	def __init__(self):
		self.model = None
	
	def load(self):
		if self.model is not None: return
		
		modeldir = os.path.dirname(os.path.realpath(__file__))
		# https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth
		modelpath = os.path.join(modeldir, 'mlsd_large_512_fp32.pth')
		
		self.model = MobileV2_MLSD_Large()
		self.model.load_state_dict(torch.load(modelpath), strict=True)
		self.model.eval()
	
	def apply(self, input_image, thr_v, thr_d):
		assert input_image.ndim == 3
		
		if self.model is None: self.load()
		
		img = input_image
		img_output = np.zeros_like(img)
		try:
			with torch.no_grad():
				lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
				for line in lines:
					x_start, y_start, x_end, y_end = [int(val) for val in line]
					cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
		except Exception as e:
			pass
		return img_output[:, :, 0]
