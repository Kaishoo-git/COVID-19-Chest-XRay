from torch.utils.data import DataLoader
from dataLoader import *
from models import *
from evaluation import *
import time

BATCH_SIZE = 16
def main():

    data_since = time.time()

    # Data Loader
    print("Loading Data...")
    train_dataset = Covid19DataSet('train', transform = 'vanilla')
    print(f"Vanilla Train of length {len(train_dataset)} Loaded")
    augmented_dataset = Covid19DataSet('train', transform = 'augment')
    print(f"Augmented Train of length {len(augmented_dataset)} Loaded")
    validation_dataset = Covid19DataSet('val', transform = 'vanilla')
    print(f"Validation of length {len(validation_dataset)} Loaded")
    test_dataset = Covid19DataSet('test', transform = 'vanilla')
    data_time = time.time() - data_since
    data_mins, data_sec = data_time//60, data_time % 50
    print (f"Data Loaded in {data_mins:.0f}mins {data_sec:.0f}s")

    print("Creating DataLoaders")
    loader_since = time.time()
    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    loader_time = time.time() - loader_since
    loader_mins, loader_sec = loader_time//60, loader_time % 50
    print (f"Data Loaded in {loader_mins:.0f}mins {loader_sec:.0f}s")

    print("Creating Models")
    # Model instantiation
    model_11 = ConvNet()
    model_12 = ConvNet()
    model_21 = ConvNetGlobPooling()
    model_22 = ConvNetGlobPooling()
    print("Models Created")

    # Model Training 
    # print("Training vanilla model with vanilla dataset")
    # train_model(model_11, train_loader, validation_loader)
    # print("Training vanilla model with augmented dataset")
    # train_model(model_12, train_loader, test_loader)
    print("Training global pooling model with vanilla dataset")
    train_model(model_21, train_loader, validation_loader)
    print("Training global pooling model with augmented dataset")
    train_model(model_22, train_loader, validation_loader)

    # Model evaluation
    metric_21_prec = evaluate_model(model_21, test_loader, 'prec') * 100
    metric_21_rec = evaluate_model(model_21, test_loader, 'rec') * 100
    print(f"model 2.1: Precision = {metric_21_prec:.0f}% | Recall = {metric_21_rec:.0f}%")

    metric_22_prec = evaluate_model(model_21, test_loader, 'prec') * 100
    metric_22_rec = evaluate_model(model_22, test_loader, 'rec') * 100
    print(f"model 2.2: Precision = {metric_22_prec:.0f}% | Recall = {metric_22_rec:.0f}%")
    
    # Model visualisation with GradCam

if __name__ == "__main__":
    main()