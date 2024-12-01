import yaml
import pickle

from modules.datasets import Covid19DataSet
from modules.heatmaps import find_img, vis_comparison
from metrics_script import get_model

from torch.utils.data import DataLoader

def explainable_workflow():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    BATCH_SIZE = ['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    DATASET_PATH = config['path']['dataset']['preprocessed']
    HEATMAP_PATH = config['path']['visualisations']['heatmaps']
    with open(f"{DATASET_PATH}test.pkl", "rb") as f:
        test_data = pickle.load(f)
    test_dataset = Covid19DataSet(test_data['features'], test_data['labels'], transform='vanilla')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model_classes = [('convnet', 'convnet', None), ('resnet', 'resnet', 'default'), ('densenet_default', 'densenet', 'default')]
    model = get_model(model_classes[0] ,config)
    pimg = find_img(model, test_loader, True)
    nimg = find_img(model, test_loader, False)
    # Get positive and negative image

    save_path = f'{HEATMAP_PATH}{model_classes[0][0]}.png'
    vis = vis_comparison(model, pimg, nimg) 
    vis.savefig(save_path, dpi = 300)
    print(f"Heatmap saved to {save_path}")
    

if __name__ == "__main__":    
    explainable_workflow()