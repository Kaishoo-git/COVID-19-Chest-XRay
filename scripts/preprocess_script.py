import torchxrayvision as xrv
import yaml
import pickle

from modules.preprocess import stratified_split

def preprocess_workflow():
    print("Loading datasets")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RANDOM_STATE = config['misc']['random_seed']
    IMG_PATH, CSV_PATH =  config['path']['dataset']['images'], config['path']['dataset']['csv']
    DATALOADER_PATH = config['path']['dataset']['preprocessed']

    data = xrv.datasets.COVID19_Dataset(imgpath = IMG_PATH, csvpath = CSV_PATH)
    train_rs, val, test = stratified_split(data, True, RANDOM_STATE)
    train_us, _, _ = stratified_split(data, False, RANDOM_STATE)

    with open(f"{DATALOADER_PATH}train_resampled.pkl", "wb") as f:
        pickle.dump(train_rs, f)
    with open(f"{DATALOADER_PATH}train_unsampled.pkl", "wb") as f:
        pickle.dump(train_us, f)
    with open(f"{DATALOADER_PATH}val.pkl", "wb") as f:
        pickle.dump(val, f)
    with open(f"{DATALOADER_PATH}test.pkl", "wb") as f:
        pickle.dump(test, f)
    print("Preprocessed datasets saved in data/preprocessed/")

if __name__ == "__main__":
    preprocess_workflow()
