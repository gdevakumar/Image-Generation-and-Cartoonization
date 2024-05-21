import argparse
from cartoongan import *

parser = argparse.ArgumentParser(description="Train different models or run inference on the model")

parser.add_argument('--train', choices=['cartoongan', 'dreambooth'], help='Train a model')
parser.add_argument('--test', choices=['cartoongan', 'dreambooth'], help='Test a model')

args = parser.parse_args()

if args.train:
    if args.train == 'cartoongan':
        print('Training Cartoon GAN model')
        model = CartoonGAN()
        losses, validation_losses = model.train(220, 'checkpoints')
        plot_loss_curves(losses, validation_losses)
    elif args.train == 'dreambooth':
        
        print('Finetuning Dreambooth model')

if args.test:
    if args.test == 'cartoongan':
        print('Running Inference on Cartoon GAN')
        cartoongan_inference()
    elif args.test == 'dreambooth':
        print('Running Inference on Dreambooth')
        dreambooth_inference()
