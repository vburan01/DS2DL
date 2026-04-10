# Deep Spatially-Regularized Superpixel-based Diffusion Learning: 

This code is the implementation of **Deep Superpixel-based and Spatially regularized Diffusion Learning** proposed in "Deep Spatially-regularized and Superpixel-Based Diffusion Learning for Unsupervised Hyperspectral Image Clustering". DS2DL is an unsupervised clustering method used on HSI datasets.

DS2DL uses Matlab Toolboxes such as [Entropy Rate Superpixel](https://github.com/mingyuliutw/EntropyRateSuperpixel), and [Diffusion Learning](https://github.com/sampolk/DiffusionLearning).
- Contact: vutichart.buranasiri@tufts.edu

  # How to run DS2DL:

The algorithm is split up into two parts. the "run_latent.py" file is used to generate the compressed representation of the HSI from the Unsupervised Masked Autoencoder (UMAE) in Python, and saves it as a .mat file in the "outputs" folder. Then, the "DS2DL.m" file takes in a specfied .mat file from the "outputs" folder based on user prompts, and runs the modified S2DL algorithm using the compressed latent. The steps to run this code are outlined below as follows.    

  1. ensure that the dependencies in "requirements.txt" are installed in your python environment.

  2. run "run_latent.py" with desired hyperparameters. An example command line is as follows:

      python run_latent.py --num_train_pixels 600 --mask_ratio 0.65 --dataset Botswana

  After the code as finished running, this will produce a file "Botswana_FPS600_MR0.65.mat" in the outputs folder. 

  3. open and run the DS2DL file and input the prompts based on the previous hyperparameters chosen. An example is given as follows in the MATLAB terminal:

     DS2DL.m
     600
     0.65

  The clustering results will be displayed in the MATLAB terminal. 
  
# Reproducibility 

details on hyperparamters that reproduce figures from the paper are contained in the "best_hyperparameters.m" file. 


  
