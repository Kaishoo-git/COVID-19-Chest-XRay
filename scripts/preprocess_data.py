import torchxrayvision as xrv
import yaml
from torch.utils.data import DataLoader

from modules.preprocess import stratified_split
from modules.dataset import Covid19DataSet

def preprocess_data_workflow(resample = False):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RANDOM_STATE = config['misc']['random_seed']
    BATCH_SIZE, NUM_WORKERS = config['training']['batch_size'], config['training']['num_workers']
    IMG_PATH, CSV_PATH =  config['path']['dataset']['images'], config['path']['dataset']['csv']

    data = xrv.datasets.COVID19_Dataset(imgpath = IMG_PATH, csvpath = CSV_PATH)
    train, val, test = stratified_split(data, resample, RANDOM_STATE)

    train = DataLoader(dataset = Covid19DataSet(train), batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    val = DataLoader(dataset = Covid19DataSet(val), batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    test = DataLoader(dataset = Covid19DataSet(val), batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

    return train, val, test

if __name__ == "__main__":
    resample_choice = input("Do you want to resample the dataset? (yes/no): ").strip().lower() == "yes"
    train, val, test = preprocess_data_workflow(resample = resample_choice)
    print("Data preprocessing completed.")
