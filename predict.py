# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import yaml
import torch
import numpy as np
import skimage
from torchvision import transforms
from PIL import Image
from skimage import io
from skimage.util import img_as_float32

from cog import BasePredictor, Input, Path
from utils import get_dataset, get_model, eval, load_model

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        with open("./configs/v2/neuralops_FiveK_Dark_With_Color_Bilateral_Renderer_ZBDATA.yaml", 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as error:
                print(error)

        self.model = get_model(config)
        state, _ = load_model(config['eval_model_path'])
        self.model.load_state_dict(state)


    def load_preprocess(self, path):
        """
        load image and convert to torch.Tensor 1xCxHxW
        """
        image = img_as_float32(io.imread(path))
        image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(dim=0)
        return image
    
    def postprocess(self, tensor):
        """
        1xCxHxW torch.Tensor to np.uint8 [0,255] HxWxC image
        """
        if len(tensor.shape) == 4:
                tensor = tensor[0]
                tensor = (tensor.cpu().detach().numpy()).transpose(1, 2, 0)
        else:
            raise ValueError("Tensors to be saved as image must be 4-dim (NCHW).")
        tensor = skimage.exposure.rescale_intensity(tensor, out_range=(0.0, 255.0)).astype(np.uint8)
        
    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        image = self.load_preprocess(image)
        output = self.model(image)
        output = self.postprocess(output)

        return output
        
