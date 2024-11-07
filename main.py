from torch.utils.data import DataLoader
from dataLoader import *
from models import *
from evaluation import *
from explain import *
import sys

from torch.utils.tensorboard import SummaryWriter
import torchvision

def main():
    BATCH_SIZE = 8
    writer1 = SummaryWriter("logs/COVID19_train")
    writer2 = SummaryWriter("logs/COVID19_test")

    data_since = time.time()

    # Data Loader
    print("Loading Data")
    augmented_dataset = Covid19DataSet('train', transform = 'augment')
    validation_dataset = Covid19DataSet('val', transform = 'vanilla')
    test_dataset = Covid19DataSet('test', transform = 'vanilla')
    data_time = time.time() - data_since
    data_mins, data_sec = data_time//60, data_time % 50
    print(f"Train size: {len(augmented_dataset)} | Validation size: {len(validation_dataset)} | Test size {len(test_dataset)}")
    print (f"Time taken: {data_mins:.0f}mins {data_sec:.2f}s")

    print("Creating DataLoaders")
    loader_since = time.time()
    train_loader = DataLoader(dataset = augmented_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    loader_time = time.time() - loader_since
    loader_mins, loader_sec = loader_time//60, loader_time % 50
    print(f"Train size: {len(train_loader)} | Validation size: {len(validation_loader)} | Test size {len(test_loader)}")
    print (f"Time taken: {data_mins:.0f}mins {data_sec:.2f}s")

    ### TensorBoard add sample Images. We will use BATCH_SIZE of 1 for clearer visualisations
    vanilla_dataset = Covid19DataSet('train', transform = 'vanilla')
    vanilla_loader = DataLoader(dataset = vanilla_dataset, batch_size = 1, shuffle = False, num_workers = 4)

    d = xrv.datasets.COVID19_Dataset(imgpath = "data/images/", csvpath = "data/csv/metadata.csv")
    img_numpy = d[0]['img'][0]

    gen = iter(vanilla_loader)
    sample_img, sample_labels = next(gen)
    img_grid = torchvision.utils.make_grid(sample_img)
    writer1.add_image('xray_images', img_grid)
    ###

    # Model 
    time_build = time.time()
    print("Building resnet18 model (modified)")
    model = models.resnet18(weights = 'DEFAULT')
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim = 1, keepdim = True))
    for params in model.parameters():
        params.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    run_time = time.time() - time_build
    print(f"Time taken: {run_time:.2f}s")

    # TensorBoard add Graph for model architecture
    writer1.add_graph(model, sample_img)
    writer1.close()

    # Model Training, creates a tensorboard scalar 
    resnet_trained = train_model(model, train_loader, validation_loader, writer1, writer2)

    # Model evaluation, creates a tensorboard AUC curve
    recall, prec, t, f, n = get_metrics(resnet_trained, test_loader, writer1)
    print(f"Total Entries: {n} | Positive Preds: {t} | Negative Preds: {f}")
    print(f"Fine-tuned: Precision = {prec:.0f}% | Recall = {recall:.0f}%")
    
    # Model visualisation with GradCam
    sample_pred = infer(resnet_trained, sample_img)

    # Create heatmap using cam
    heatmap = get_cam(resnet_trained, sample_pred, sample_img)

    # Show image with heatmap overlay
    overlay_cam(heatmap, img_numpy)


if __name__ == "__main__":
    main()