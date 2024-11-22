This is a project on using classification models on a visual dataset.

covid19_chest_xray/
├── main.py                 
├── config/                 
│   └── config.yaml
├── models/   
│   └── linearnet.pth
│   └── convnet.pth
│   └── resnet.pth
│   └── densenet.pth
├── data/                   
│   └── images/
│   └── csv/
│       └── metadata.csv
├── scripts/               
│   └── preprocess_script.py
│   └── training_model_script.py
│   └── evaluate_model_script.py
│   └── explain_model_script.py
├── modules/                
│   └── preprocess.py      
│   └── dataset.py          
│   └── models.py           
│   └── evaluation.py       
│   └── explain.py         
│   └── simulator.py        
├── tests/                 
│   └── test_preprocess.py
└── README.md               