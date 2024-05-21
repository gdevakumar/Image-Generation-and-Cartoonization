# Image-Generation-and-Cartoonization

This project has 2 stages - Finetuning a pretrained Text-to-image model [StableDiffusion V-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) using [DreamBoothing](https://dreambooth.github.io/) technique and [Cartoon GAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf).

## Files
- ```download_datasets.py``` file downloads the datasets for Cartoon GAN and Dreambooth networks.
- ```download_checkpoints.py``` file downloads the model checkpoints.

## Dataset
- Cartoon GAN: https://drive.google.com/file/d/1esNF4ZDtdQ0-UpIqNq4OHSkD-8Yp0JFP/view?usp=sharing

## To run this project
1. Clone the repository
```
git clone https://github.com/gdevakumar/Image-Generation-and-Cartoonization.git
cd Image-Generation-and-Cartoonization
```

2. Create a virtual environment for best practice (*optional*)
- On **Windows** machines
```
python -m venv env
env\Scripts\activate
```

- On **Linux/Mac** machines
```
python -m venv env
source env/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Download datasets
```
python download_datasets.py
```


## Inference
1. Download the best checkpoints 
```
python download_checkpoints.py
```

2. Run the inference script
```
python main.py --test
```

