import torchxrayvision as xrv
from preprocess import *
from dataset import *
from models import *
from evaluation import *
from explain import *

def main():
    print("")
    d = xrv.datasets.COVID19_Dataset(imgpath = "data/images/", csvpath = "data/csv/metadata.csv")
    pos_idx, neg_idx = find_positive(d), find_negative(d, 4)
    sample_img_pos, sample_img_neg = d[pos_idx]['img'][0], d[neg_idx]['img'][0]
    sample_input_pos, sample_input_neg = tensorize_image(sample_img_pos).unsqueeze(0), tensorize_image(sample_img_neg).unsqueeze(0)
    BATCH_SIZE = 16

    unsampled_d = preprocess(d, resample = False)
    resampled_d = preprocess(d, resample = True)

    show_image(unsampled_d, pos_idx, processed = True)
    show_image(unsampled_d, neg_idx, processed = True)
    show_image(resampled_d, neg_idx+1, processed = True)

    unsampled_train_dataset = Covid19DataSet('train', unsampled_d)
    unsampled_validation_dataset = Covid19DataSet('val', unsampled_d)
    unsampled_test_dataset = Covid19DataSet('test', unsampled_d)
    unsampled_train_loader = DataLoader(dataset = unsampled_train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    unsampled_validation_loader = DataLoader(dataset = unsampled_validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    unsampled_test_loader = DataLoader(dataset = unsampled_test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

    linearnet_v_us = LinearNet()
    linearnet_us, linearnet_stats_us = train_model(linearnet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    ln_t_loss_us, ln_t_acc_us, ln_v_loss_us, ln_v_acc_us, ln_time_us = linearnet_stats_us
    ln_prec_us, ln_rec_us, ln_f1_us, *ln_rest_us = get_metrics(linearnet_us, unsampled_test_loader)

    convnet_v_us = ConvNet()
    convnet_us, convnet_stats_us = train_model(convnet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    cn_t_loss_us, cn_t_acc_us, cn_v_loss_us, cn_v_acc_us, cn_time_us = convnet_stats_us
    cn_prec_us, cn_rec_us, cn_f1_us, *cn_rest_us = get_metrics(convnet_us, unsampled_test_loader)

    resnet_v_us = MyResNet18()
    resnet_us, resnet_stats_us = train_model(resnet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    rn_t_loss_us, rn_t_acc_us, rn_v_loss_us, rn_v_acc_us, rn_time_us = resnet_stats_us
    rn_prec_us, rn_rec_us, rn_f1_us, *rn_rest_us = get_metrics(resnet_us, unsampled_test_loader)

    densenet_v_us = MyDenseNet()
    densenet_us, densenet_stats_us = train_model(densenet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    dn_t_loss_us, dn_t_acc_us, dn_v_loss_us, dn_v_acc_us, dn_time_us = densenet_stats_us
    dn_prec_us, dn_rec_us, dn_f1_us, *dn_rest_us = get_metrics(densenet_us, unsampled_test_loader)

    headers = ['Model', 'Precision', 'Recall', 'F1 Score']
    unsampled_metrics = [
        ['linearnet', ln_prec_us, ln_rec_us, ln_f1_us],
        ['convnet', cn_prec_us, cn_rec_us, cn_f1_us],
        ['resnet', rn_prec_us, rn_rec_us, rn_f1_us],
        ['densnet', dn_prec_us, dn_rec_us, dn_f1_us]
    ]
    save_table_as_image(headers, unsampled_metrics, 'unsampled_metrics.png')

    plot_loss(10, ln_t_loss_us, ln_v_loss_us, 'linearnet, unsampled')
    plot_loss(10, cn_t_loss_us, cn_v_loss_us, 'convnet, unsampled')
    plot_loss(10, rn_t_loss_us, rn_v_loss_us, 'resnet, unsampled')
    plot_loss(10, dn_t_loss_us, dn_v_loss_us, 'densenet, unsampled')

    for param in convnet_us.parameters():
        param.requires_grad = True
    for param in resnet_us.parameters():
        param.requires_grad = True
    for param in densenet_us.parameters():
        param.requires_grad = True
    visualize_heatmaps(convnet_us, sample_img_pos, sample_input_pos)
    visualize_heatmaps(resnet_us, sample_img_pos, sample_input_pos)
    visualize_heatmaps(densenet_us, sample_img_pos, sample_input_pos)

    visualize_heatmaps(convnet_us, sample_img_neg, sample_input_neg)
    visualize_heatmaps(resnet_us, sample_img_neg, sample_input_neg)
    visualize_heatmaps(densenet_us, sample_img_neg, sample_input_neg)

if __name__ == "__main__":
    main()

