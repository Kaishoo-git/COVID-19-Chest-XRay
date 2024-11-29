This is a project on using classification models on a visual dataset.

Add root dir in shell `$env:PYTHONPATH = "."`
```
covid19_chest_xray/
├── main.py                 
├── config/               
├── models/   
│   ├── performance/
│   ├── stats/
│   └── weights/
├── data/                   
│   ├── images/
│   ├── csv/
│   └── preprocessed/
├── scripts/               
├── modules/     
├── visualisations/  
│   ├── heatmaps/
│   ├── plots/
│   └── tables/    
├── tests/                 
├── logs/                   # Optional
└── README.md
```
models/performance - precision, recall and f1 score for each K-Fold
models/weights - model weights to load

scripts/ - scripts to run for specific purposes
modules/ - modules containing functions to call

visualisations/heatmaps - gradcam, gradcam++ heatmaps
visualisations/tables - tables to compare metrics
visualisations/plots - roc-auc curve

```
from modules.exploration import explore_data, show_image    # For EDA, not needed
from modules.preprocess import get_data, stratified_split, process_all    # For preprocessing (Already done, can just call from data/preprocessed)
from modules.datasets import Covid19DataSet    # (Already done, can just call from data/preprocessed)
from modules.models import get_model    # Create a fresh model
from modules.training import train_model, get_metrics    # For training and testing model
from modules.metrics import plot_roc_auc    # , create_table, plot_loss_and_metric
```