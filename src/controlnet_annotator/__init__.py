__all__ = ['canny', 'hed', 'fake_scribble', 'scribble', 'mlsd']

# sd-webui-controlnet/scripts/processor

from .util import HWC3, resize_image

import cv2
import numpy as np

# ========================================================================================

models = {}

def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
	img = resize_image(HWC3(img), res)
	return cv2.Canny(img, thr_a, thr_b)

def hed(img, res=512, **kwargs):
	img = resize_image(HWC3(img), res)
	if 'hed' not in models:
		from .methods.hed import Model as HedModel
		models['hed'] = HedModel()
	return models['hed'].apply(img)

def fake_scribble(img, res=512, **kwargs):
	result = hed(img, res)
	
	result = models['hed'].nms(result, 127, 3.0)
	result = cv2.GaussianBlur(result, (0, 0), 3.0)
	result[result > 10] = 255
	result[result < 255] = 0
	return result

def scribble(img, res=512, **kwargs):
	img = resize_image(HWC3(img), res)
	result = np.zeros_like(img, dtype=np.uint8)
	result[np.min(img, axis=2) < 127] = 255
	return result

def mlsd(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
	img = resize_image(HWC3(img), res)
	if 'mlsd' not in models:
		from .methods.mlsd import Model as MlsdModel
		models['mlsd'] = MlsdModel()
	return models['mlsd'].apply(img, thr_a, thr_b)
