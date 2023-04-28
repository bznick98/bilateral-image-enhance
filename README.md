# bilateral-image-enhance

### Install
```bash
git clone https://github.com/bznick98/bilateral-image-enhance.git
[optional] create a virtual/conda environment
pip install -r requirement.txt
```

### Currently Supported **Models**
* **HDRNet**
    * Real-time image enhancement using Bilateral Grid.
* **NeuralOps**
    * Sequential image enhancement using learned neural operators. Real-time on 1M pixel images.
* **Zero DCE++**
    * Zero-Reference training (no target image needed), lightweight low-light enhancement networks, learns the pixel-wise coefficient for enhance curves.

### Currently Supported **Loss**
* L2 Loss: `torch.nn.MSELoss`
* L1 Loss: `torch.nn.SmoothL1Loss`
* TV Loss: `loss.TVLoss`
* Cosine Loss: `loss.CosineLoss`
* Zero Reference Loss: `loss.ZeroReferenceLoss`
    * Color Consistency Loss: `loss.ColorConsistencyLoss`
    * Spatial Consistency Loss: `loss.SpatialConsistencyLoss`
    * Exposure Control Loss: `ExposureControlLoss`
    * Illumination Smoothness Loss (TV Loss): `IlluminationSmoothnessLoss`
* Or just specify your own loss in `train.py`

### Currently Supported **Dataset**
* **LOL**: low-light image dataset
    * trainset: 485 image pairs
    * testset: 15 image pairs
    * size: 600x400
* **VELOL**: low-light image dataset
    * trainset: 400 image pairs
    * testset: 100 image pairs
    * size: 600x400
* **MIT-Adobe-5K-Lite**: from NeuralOps paper, image enhancement dataset
    * trainset: 4500 image pairs
    * testset: 500 image pairs (1 pair corrupted, usable 499 pairs)
    * size: 332x500 or 500x333 (could be landscape/portrait)
* **SICE**: multi-exposure (both +/-) light enhancement dataset
    * trainset: 360 image sets (each set could contain ~7 images for each exposure level).
    * testset: 229 image sets (each set could contain ~7 images for each exposure level).
    * size: ~4000x3000, variable size

### Training
Training & testing are managed by config files. It saves model weights to `config['checkpoint_save_path']` (default to `result/checkpoints/`) every `config['checkpoint_interval']` epochs.
```bash 
python train.py --config configs/hdrnet_FiveK_Lite.yaml
```

### Testing
test.py evaluates a pretrained model on a specific dataset, reports average PSNR, SSIM, MAE, LPIPS. It also saves visualization output to `result/visualization` every `config['visualization_interval']` epochs.
```bash 
python test.py --config configs/neuralops_FiveK_Lite.yaml
```