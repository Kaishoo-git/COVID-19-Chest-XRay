This is a project on using classification models on a visual dataset.

Add root dir in shell `$env:PYTHONPATH = "."`

covid19_chest_xray/
├── main.py                 
├── config/                 
│   └── config.yaml
├── models/   
│   ├── linearnet.pth
│   ├── convnet.pth
│   ├── resnet.pth
│   ├── densenet.pth
│   └── models_stats.json
├── data/                   
│   ├── images/
│   ├── csv/
│   │   └── metadata.csv
│   └── preprocessed/
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
├── scripts/               
│   ├── __init__.py
│   ├── preprocess_script.py
│   ├── training_model_script.py
│   ├── evaluate_model_script.py
│   └── explain_model_script.py
├── modules/                
│   ├── preprocess.py      
│   ├── dataset.py          
│   ├── models.py           
│   ├── evaluation.py       
│   ├── explain.py         
│   └── simulator.py        
├── tests/                 
│   └── test_preprocess.py
├── logs/                   # Optional
│   ├── training.log
│   └── evaluation.log
└── README.md