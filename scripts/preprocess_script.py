import torchxrayvision as xrv
import yaml
import pickle

from modules.preprocess import stratified_split, process_all

def preprocess_workflow():
    print("Loading datasets")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RANDOM_STATE = config['misc']['random_seed']
    IMG_PATH, CSV_PATH =  config['path']['dataset']['images'], config['path']['dataset']['csv']
    DATASET_PATH = config['path']['dataset']['preprocessed']

    data = xrv.datasets.COVID19_Dataset(imgpath = IMG_PATH, csvpath = CSV_PATH)
    train, test = stratified_split(data, RANDOM_STATE)
    print("Downloading")
    with open(f"{DATASET_PATH}train.pkl", "wb") as f:
        pickle.dump(train, f)
    # with open(f"{DATASET_PATH}val.pkl", "wb") as f:
    #     pickle.dump(val, f)
    with open(f"{DATASET_PATH}test.pkl", "wb") as f:
        pickle.dump(test, f)

    # dataset = process_all(data)
    # with open(f"{DATASET_PATH}dataset.pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    print("Preprocessed datasets saved in data/preprocessed/")

if __name__ == "__main__":
    preprocess_workflow()
