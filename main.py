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
    print (f"Time taken: {loader_mins:.0f}mins {loader_sec:.2f}s")

    ### We will use BATCH_SIZE of 1 for clearer visualisations, this would be used for subsequent parts
    vanilla_dataset = Covid19DataSet('train', transform = 'vanilla')
    vanilla_loader = DataLoader(dataset = vanilla_dataset, batch_size = 1, shuffle = False, num_workers = 4)

    d = xrv.datasets.COVID19_Dataset(imgpath = "data/images/", csvpath = "data/csv/metadata.csv")
    img_numpy = d[0]['img'][0]
    sample_label = d[0]['lab'][3]

    gen = iter(vanilla_loader)
    sample_img, sample_labels = next(gen)
    img_grid = torchvision.utils.make_grid(sample_img)

    # TENSORBOARD: Add sample Images.
    writer1.add_image('xray_images', img_grid)

    # Model 
    print("")
    time_build = time.time()
    print("Building resnet18 model (modified)")
    resnet_v = MyResNet18()
    print("Building densenet model (modified)")
    densenet_v = MyDenseNet()
    run_time = time.time() - time_build
    print(f"Time taken: {run_time:.2f}s")

    # TENSORBOARD: Add Graph for model architecture
    writer1.add_graph(resnet_v, sample_img)
    writer1.add_graph(densenet_v, sample_img)

    # Model Training, creates a tensorboard scalar 
    resnet = train_model(resnet_v, train_loader, validation_loader, writer1, writer2)
    densenet = train_model(densenet_v, train_loader, validation_loader, writer1, writer2)

    print("")
    # TENSORBOARD: Model evaluation, AUC curve
    print("ResNet Model metrics:")
    recall_rn, prec_rn, t_rn, f_rn, n_rn = get_metrics(resnet, test_loader, writer1)
    
    print("DenseNet Model metrics:")
    recall_dn, prec_dn, t_dn, f_dn, n_dn = get_metrics(resnet, test_loader, writer1)

    print("")
    # Enable gradients for gradcam and gradcam++
    for param in resnet.parameters():
        param.requires_grad = True
    for param in densenet.parameters():
        param.requires_grad = True

    # Model visualisation with GradCam
    # print(f'Patient is COVID {'Positive' if sample_label == 1 else 'Negative'}')

    print("Generating GradCam")
    output_rn = infer(resnet, sample_img)
    prob_rn_tensor = torch.sigmoid(output_rn)
    prob_rn = prob_rn_tensor.item()
    label_rn = 'Positive' if prob_rn >= 0.5 else 'Negative'

    # Generate heatmaps using Grad-CAM and Grad-CAM++
    heatmap_gc_rn = get_gradcam(resnet, output_rn, sample_img)
    heatmap_gcpp_rn = get_gradcam_pp(resnet, output_rn, sample_img)

    vis_comparison(img_numpy, heatmap_gc_rn, heatmap_gcpp_rn, f'Actual: {'Positive' if sample_label == 1 else 'Negative'}| ResNet Predicted: {label_rn}, {(prob_rn*100):.0f}%')

    print("Generating GradCam++")
    # Model visualisation with GradCam
    output_dn = infer(densenet, sample_img)
    prob_dn_tensor = torch.sigmoid(output_dn)
    prob_dn = prob_dn_tensor.item()
    label_dn = 'Positive' if prob_dn >= 0.5 else 'Negative'

    # Generate heatmaps using Grad-CAM and Grad-CAM++
    heatmap_gc_dn = get_gradcam(densenet, output_dn, sample_img)
    heatmap_gcpp_dn = get_gradcam_pp(densenet, output_dn, sample_img)

    vis_comparison(img_numpy, heatmap_gc_dn, heatmap_gcpp_dn, f'Actual: {'Positive' if sample_label == 1 else 'Negative'} | DenseNet Predicted: {label_dn} with {(prob_dn*100):.0f}%')


if __name__ == "__main__":
    main()