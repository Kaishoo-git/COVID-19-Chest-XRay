This is a project on using classification models on a visual dataset.
### Repository Link
[Covid19-Chest-XRay](https://github.com/Kaishoo-git/COVID-19-Chest-XRay)

### Results 
[Google Slides](https://docs.google.com/presentation/d/1_slouAIcIl-NpzyisdY78LP2o4_Mr8cLlrTmBbe4nUQ/edit?usp=sharing)

```
covid19_chest_xray/      
├── config/   
│       └── config.yaml         
├── models/   
│   ├── performance/    stores the validation losses during training
│   └── weights/        stores weights for trained_models
├── data/                   
│   ├── images/
│   ├── csv/
│   └── preprocessed/   stores training and test data
├── scripts/            stores scripts to train, evaluate as well as generate heatmaps                
├── modules/            stores functions
├── visualisations/  
│   ├── heatmaps/       stores gradcam and gradcam++ heatmaps
│   ├── plots/          stores roc_auc curves
│   └── tables/         stores tables for metrics of models   
├── tests/                 
└── README.md
```

### How to run
Open shell in root directory
```
Add root dir in shell `$env:PYTHONPATH = "."`
!pip install -r requirements.txt
!python {script}.py     # For purpose needed
```
You can control the global variables in config/config.yaml

### References
1. [Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks. 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), 839–847.](https://doi.org/10.1109/WACV.2018.00097)

2. [Cohen, J. P., Viviano, J. D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., Lungren, M. P., Chaudhari, A., Brooks, R., Hashir, M., & Bertrand, H. (2021). TorchXRayVision: A library of chest X-ray datasets and models (arXiv:2111.00595). arXiv.](https://doi.org/10.48550/arXiv.2111.00595)

3. [Han, K., Wang, Y., Zhang, C., Li, C., & Xu, C. (2018). AutoEncoder Inspired Unsupervised Feature Selection (arXiv:1710.08310). arXiv.](https://doi.org/10.48550/arXiv.1710.08310)

4. [Huang, G., Liu, Z., Maaten, L. van der, & Weinberger, K. Q. (2018). Densely Connected Convolutional Networks (arXiv:1608.06993). arXiv.](https://doi.org/10.48550/arXiv.1608.06993)

5. [Kaya, Y., & Gürsoy, E. (2023). A MobileNet-based CNN model with a novel fine-tuning mechanism for COVID-19 infection detection. Soft Computing, 27(9), 5521–5535.](https://doi.org/10.1007/s00500-022-07798-y)

6. [Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. International Journal of Computer Vision, 128(2), 336–359.](https://doi.org/10.1007/s11263-019-01228-7)

7. [Virk, J. S., & Bathula, D. R. (2020). Domain Specific, Semi-Supervised Transfer Learning for Medical Imaging (arXiv:2005.11746). arXiv.](https://doi.org/10.48550/arXiv.2005.11746)
