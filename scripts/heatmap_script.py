import yaml

from evaluate_script import get_models
from modules.metrics import plot_loss_and_metric, plot_roc_auc, create_table

def explainable_workflow(resample):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    models = get_models(resample, config)

    # Get positive and negative image
    # vis_comparison(model, img_pos, img_neg) # save image
    pass

if __name__ == "__main__":    
    resample_choice = input("Which model results would you like? (unsampled/resampled): ").strip().lower() == "resampled"
    explainable_workflow(resample=resample_choice)