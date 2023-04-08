# ControlNet Annotator

Annotator Modules (Preprocessor) for [ControlNet](https://github.com/lllyasviel/ControlNet)

Working on **CPU**


## Supported methods
- Canny
- HED
- Scribble
- Fake Scribble
- MLSD


## Download Models
```
# HED
wget -O methods/hed/network-bsds500.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth

# MLSD
wget -O methods/mlsd/mlsd_large_512_fp32.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth
```


## PyTorch on CPU

### Linux
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Or

```
cat > ./venv/pip.conf <<EOT
[global]
extra-index-url = 
    https://download.pytorch.org/whl/cpu
EOT

pip install torch
```

### Mac
```
pip install torch
```


## Example
```
import cv2
from controlnet_annotator import canny, hed, scribble, fake_scribble, mlsd

fp = '/path/to/img.jpg'
img = cv2.imread(fp)

resp = canny(img)
cv2.imshow('image', resp); cv2.waitKey(0)
cv2.destroyAllWindows(); cv2.waitKey(1)
```


## References
- [ControlNet](https://github.com/lllyasviel/ControlNet)
  - License: Apache-2.0 license
 
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
  - License: MIT

- [M-LSD](https://github.com/navervision/mlsd)
  - Copyright 2021-present NAVER Corp.
  - License: Apache License v2.0
