# Deep Spatially-Regularized Superpixel-based Diffusion Learning: 

This code is the implementation of **Deep Superpixel-based and Spatially regularized Diffusion Learning** proposed in "Deep Spatially-regularized and Superpixel-Based Diffusion Learning for Unsupervised Hyperspectral Image Clustering". DS2DL is an unsupervised clustering method used on HSI datasets.

DS2DL uses Matlab Toolboxes such as [Entropy Rate Superpixel](https://github.com/mingyuliutw/EntropyRateSuperpixel), and [Diffusion Learning](https://github.com/sampolk/DiffusionLearning).
- Contact: vutichart.buranasiri@tufts.edu

  # How to run DS2DL:

The algorithm is split up into two parts. the "run_latent.py" file is used to generate the compressed representation of the HSI from the Unsupervised Masked Autoencoder (UMAE) in Python, and saves it as a .mat file in the "outputs" folder. Then, the "DS2DL.m" file takes in a specfied .mat file from the "outputs" folder based on user prompts, and runs the modified S2DL algorithm using the compressed latent. The steps to run this code are outlined below as follows.  

 # Running Code to test for OA, AA, Kappa:

  1. ensure that the dependencies in "requirements.txt" are installed in your python environment.

  2. run "run_latent.py" with desired hyperparameters. An example command line is as follows:

      `python run_latent.py --num_train_pixels 600 --mask_ratio 0.65 --dataset Botswana`

  After the code as finished running, this will produce a file "Botswana_FPS600_MR0.65.mat" in the outputs folder. 

  3. open and run the DS2DL file and input the prompts based on the previous hyperparameters chosen. An example is given as follows in the MATLAB terminal:

     `DS2DL.m 600 0.65`

  The clustering results will be displayed in the MATLAB terminal. 

  # Running Code to test for Purity and NMI:

  4. Run steps 1-3 from above and collect resulting output files from step 3 into a folder.
  5. run "Find_Purity.m" (instructions in function contract in file). This saves and displays global best Purity and NMI results from all .mat files in the folder from step 4.
  
# Reproducibility 

details on hyperparamters that reproduce figures from the paper are contained in the "best_hyperparameters.m" file. 

# Citations

Please kindly cite these papers if you find any of this code useful. 


K. Cui, R. Li, S. L. Polk, Y. Lin, H. Zhang, J. M. Murphy, R. J. Plemmons, and R. H. Chan, “Superpixel-based and spatially regularized diffusion learning for unsupervised hyperspectral image clustering,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–18, 2024

  
      @ARTICLE{Cui2024S2DL,
        title={Superpixel-Based and Spatially Regularized Diffusion Learning for Unsupervised Hyperspectral Image Clustering},
        author={Cui, Kangning and Li, Ruoning and Polk, Sam L. and Lin, Yinyi and Zhang, Hongsheng and Murphy, James M. and Plemmons, Robert J. and Chan, Raymond H.},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        volume={62},
        pages={1-18},
        year={2024},
        doi={10.1109/TGRS.2024.XXXXX}
      }
    

D. Ibañez, R. Fernandez-Beltran, F. Pla and N. Yokoya, "Masked Auto-Encoding Spectral–Spatial Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5542614, doi: 10.1109/TGRS.2022.3217892.

    @ARTICLE{9931741,
      author={Ibañez, Damian and Fernandez-Beltran, Ruben and Pla, Filiberto and Yokoya, Naoto},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Masked Auto-Encoding Spectral–Spatial Transformer for Hyperspectral Image Classification}, 
      year={2022},
      volume={60},
      number={},
      pages={1-14},
      doi={10.1109/TGRS.2022.3217892}
      }

  
