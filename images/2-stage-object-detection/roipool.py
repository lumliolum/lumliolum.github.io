import torch
import torchvision

import numpy as np
feature_map = np.array([
    [0.70, 0.41, 0.38, 1.23, 0.24],
    [0.14, 0.45, 0.31, 0.73, 3.22],
    [0.11, 0.41, 0.79, 0.69, 0.44],
    [1.47, 0.25, 0.09, 0.32, 2.98],
    [0.48, 0.87, 0.77, 0.26, 0.11],
])


feature_map = torch.tensor(feature_map, requires_grad=False).float()

# (batch, channel, h, w) -> (1, 1, 5, 5)
feature_map = feature_map.unsqueeze(0).unsqueeze(0)

# boxes -> (1, 5)
boxes = np.array([
    [0, 0, 1, 5],
])
boxes = torch.tensor(boxes, requires_grad=False).float()

# roi pooling layer of 3x3
pool = torchvision.ops.roi_pool(input=feature_map, boxes=[boxes], output_size=3, spatial_scale=1.0)
print(pool)
