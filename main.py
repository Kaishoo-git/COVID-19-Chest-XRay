from torch.utils.data import DataLoader
from dataLoader import *
from models import *
from evaluation import *
import time
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 16
def main():
    writer = SummaryWriter("runs/COVID19")

    data_since = time.time()

    # Data Loader
    print("Loading Data...")
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
    train_loader = DataLoader(dataset = augmented_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    loader_time = time.time() - loader_since
    loader_mins, loader_sec = loader_time//60, loader_time % 50
    print (f"Data Loaded in {loader_mins:.0f}mins {loader_sec:.0f}s")

    gen = iter(train_loader)
    images, labels = next(gen)
    # output = globpool_model(features)
    print(images)
    # print("Creating Models")
    # model_22 = ConvNetGlobPooling()
    # print("Models Created")

    # # Model Training 
    # print("Training global pooling model with augmented dataset")
    # train_model(model_22, train_loader, validation_loader)

    # # Model evaluation

    # metric_22_prec = evaluate_model(model_22, test_loader, 'prec') * 100
    # metric_22_rec = evaluate_model(model_22, test_loader, 'rec') * 100
    # print(f"model 2.2: Precision = {metric_22_prec:.0f}% | Recall = {metric_22_rec:.0f}%")
    
    # Model visualisation with GradCam

if __name__ == "__main__":
    main()