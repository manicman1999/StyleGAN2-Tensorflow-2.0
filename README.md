# StyleGAN2 Tensorflow 2.0

Unofficial implementation of StyleGAN 2 using TensorFlow 2.0.

Original paper: Analyzing and Improving the Image Quality of StyleGAN

Arxiv: https://arxiv.org/abs/1912.04958


This implementation includes all improvements from StyleGAN to StyleGAN2, including:

Modulated/Demodulated Convolution, Skip block Generator, ResNet Discriminator, No Growth,

Lazy Regularization, Path Length Regularization, and can include larger networks (by adjusting the cha variable).



## Image Samples
Trained on Landscapes for 3.48 million images (290k steps, batch size 12, channel coefficient 24):
*To clarify, 3.48 million images were shown to the Discriminator, but the dataset consists of only ~14k images.
Thus, of those 3.48 million images, most are repeats of already seen images.*

![Teaser image](./landscapes3.png)


Mixing Styles:

![Teaser image](./styles3.png)



## Before Running
Please ensure you have created the following folders:
1. /Models/
2. /Results/
3. /data/

Additionally, please ensure that your folder with images is in /data/ and changed at the top of stylegan.py.

For **pretrained models**, download the pretrained models from [here](https://drive.google.com/drive/folders/1jE0lIJxgaHS7EE6mzm_ftl0ZcaY6olqI?usp=sharing)

Then, adjust the main code at the bottom to use model.load(*model_num*), where *model_num* = the number from the downloaded model.
