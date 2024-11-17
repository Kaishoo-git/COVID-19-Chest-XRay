import torchxrayvision as xrv
from preprocess import *
from dataset import *
from models import *
from evaluation import *
from explain import *

def main():
    d = xrv.datasets.COVID19_Dataset(imgpath = "data/images/", csvpath = "data/csv/metadata.csv")
    pos_idx, neg_idx = find_positive(d), find_negative(d)
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
    ln_t_loss_us, ln_t_prec_us, ln_t_rec_us, ln_t_f1_us, ln_v_loss_us, ln_v_prec_us, ln_v_rec_us, ln_v_f1_us, ln_time_us = linearnet_stats_us
    ln_test_prec_us, ln_test_rec_us, ln_test_f1_us, *ln_test_rest_us = get_metrics(linearnet_us, unsampled_test_loader)
    ln_train_prec_us, ln_train_rec_us, ln_train_f1_us, *ln_train_rest_us = get_metrics(linearnet_us, unsampled_train_loader)

    convnet_v_us = ConvNet()
    convnet_us, convnet_stats_us = train_model(convnet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    cn_t_loss_us, cn_t_prec_us, cn_t_rec_us, cn_t_f1_us, cn_v_loss_us, cn_v_prec_us, cn_v_rec_us, cn_v_f1_us, cn_time_us = convnet_stats_us
    cn_test_prec_us, cn_test_rec_us, cn_test_f1_us, *cn_test_rest_us = get_metrics(convnet_us, unsampled_test_loader)
    cn_train_prec_us, cn_train_rec_us, cn_train_f1_us, *cn_train_rest_us = get_metrics(convnet_us, unsampled_train_loader)

    resnet_v_us = MyResNet18()
    resnet_us, resnet_stats_us = train_model(resnet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    rn_t_loss_us, rn_t_prec_us, rn_t_rec_us, rn_t_f1_us, rn_v_loss_us, rn_v_prec_us, rn_v_rec_us, rn_v_f1_us, rn_time_us = resnet_stats_us
    rn_test_prec_us, rn_test_rec_us, rn_test_f1_us, *rn_test_rest_us = get_metrics(resnet_us, unsampled_test_loader)
    rn_train_prec_us, rn_train_rec_us, rn_train_f1_us, *rn_train_rest_us = get_metrics(resnet_us, unsampled_train_loader)

    densenet_v_us = MyDenseNet()
    densenet_us, densenet_stats_us = train_model(densenet_v_us, unsampled_train_loader, unsampled_validation_loader, epochs = 10)
    dn_t_loss_us, dn_t_prec_us, dn_t_rec_us, dn_t_f1_us, dn_v_loss_us, dn_v_prec_us, dn_v_rec_us, dn_v_f1_us, dn_time_us = densenet_stats_us
    dn_test_prec_us, dn_test_rec_us, dn_test_f1_us, *dn_test_rest_us = get_metrics(densenet_us, unsampled_test_loader)
    dn_train_prec_us, dn_train_rec_us, dn_train_f1_us, *dn_train_rest_us = get_metrics(densenet_us, unsampled_train_loader)

    headers = ['Model', 'Precision', 'Recall', 'F1 Score']
    unsampled_metrics = [
        ['linearnet (train)', ln_train_prec_us, ln_train_rec_us, ln_train_f1_us],
        ['linearnet (test)', ln_test_prec_us, ln_test_rec_us, ln_test_f1_us],
        ['convnet (train)', cn_train_prec_us, cn_train_rec_us, cn_train_f1_us],
        ['convnet (test)', cn_test_prec_us, cn_test_rec_us, cn_test_f1_us],
        ['resnet (train)', rn_train_prec_us, rn_train_rec_us, rn_train_f1_us],
        ['resnet (test)', rn_test_prec_us, rn_test_rec_us, rn_test_f1_us],
        ['densenet (train)', dn_train_prec_us, dn_train_rec_us, dn_train_f1_us],
        ['densenet (test)', dn_test_prec_us, dn_test_rec_us, dn_test_f1_us]
    ]
    save_table_as_image(headers, unsampled_metrics, 'unsampled_metrics.png')

    plot_loss(10, ln_t_loss_us, ln_v_loss_us, 'linearnet, unsampled')
    plot_loss(10, cn_t_loss_us, cn_v_loss_us, 'convnet, unsampled')
    plot_loss(10, rn_t_loss_us, rn_v_loss_us, 'resnet, unsampled')
    plot_loss(10, dn_t_loss_us, dn_v_loss_us, 'densenet, unsampled')

    plot_metric(10, ln_t_f1_us, ln_v_f1_us, 'f1 score with each epoch (linearnet, unsampled)')
    plot_metric(10, cn_t_f1_us, cn_v_f1_us, 'f1 score with each epoch (convnet, unsampled)')
    plot_metric(10, rn_t_f1_us, rn_v_f1_us, 'f1 score with each epoch (resnet, unsampled)')
    plot_metric(10, dn_t_f1_us, dn_v_f1_us, 'f1 score with each epoch (densenet, unsampled)')

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

    # Resampling is done here

    resampled_train_dataset = Covid19DataSet('train', resampled_d)
    resampled_validation_dataset = Covid19DataSet('val', resampled_d)
    resampled_test_dataset = Covid19DataSet('test', resampled_d)
    resampled_train_loader = DataLoader(dataset = resampled_train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    resampled_validation_loader = DataLoader(dataset = resampled_validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    resampled_test_loader = DataLoader(dataset = resampled_test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

    linearnet_v_rs = LinearNet()
    linearnet_rs, linearnet_stats_rs = train_model(linearnet_v_rs, resampled_train_loader, resampled_validation_loader, epochs = 10)
    ln_t_loss_rs, ln_t_prec_rs, ln_t_rec_rs, ln_t_f1_rs, ln_v_loss_rs, ln_v_prec_rs, ln_v_rec_rs, ln_v_f1_rs, ln_time_rs = linearnet_stats_rs
    ln_test_prec_rs, ln_test_rec_rs, ln_test_f1_rs, *ln_test_rest_rs = get_metrics(linearnet_rs, resampled_test_loader)
    ln_train_prec_rs, ln_train_rec_rs, ln_train_f1_rs, *ln_train_rest_rs = get_metrics(linearnet_rs, resampled_train_loader)


    convnet_v_rs = ConvNet()
    convnet_rs, convnet_stats_rs = train_model(convnet_v_rs, resampled_train_loader, resampled_validation_loader, epochs = 10)
    cn_t_loss_rs, cn_t_prec_rs, cn_t_rec_rs, cn_t_f1_rs, cn_v_loss_rs, cn_v_prec_rs, cn_v_rec_rs, cn_v_f1_rs, cn_time_rs = convnet_stats_rs
    cn_test_prec_rs, cn_test_rec_rs, cn_test_f1_rs, *cn_test_rest_rs = get_metrics(convnet_rs, resampled_test_loader)
    cn_train_prec_rs, cn_train_rec_rs, cn_train_f1_rs, *cn_train_rest_rs = get_metrics(convnet_rs, resampled_train_loader)

    resnet_v_rs = MyResNet18()
    resnet_rs, resnet_stats_rs = train_model(resnet_v_rs, resampled_train_loader, resampled_validation_loader, epochs = 10)
    rn_t_loss_rs, rn_t_prec_rs, rn_t_rec_rs, rn_t_f1_rs, rn_v_loss_rs, rn_v_prec_rs, rn_v_rec_rs, rn_v_f1_rs, rn_time_rs = resnet_stats_rs
    rn_test_prec_rs, rn_test_rec_rs, rn_test_f1_rs, *rn_test_rest_rs = get_metrics(resnet_rs, resampled_test_loader)
    rn_train_prec_rs, rn_train_rec_rs, rn_train_f1_rs, *rn_train_rest_rs = get_metrics(resnet_rs, resampled_train_loader)

    densenet_v_rs = MyDenseNet()
    densenet_rs, densenet_stats_rs = train_model(densenet_v_rs, resampled_train_loader, resampled_validation_loader, epochs = 10)
    dn_t_loss_rs, dn_t_prec_rs, dn_t_rec_rs, dn_t_f1_rs, dn_v_loss_rs, dn_v_prec_rs, dn_v_rec_rs, dn_v_f1_rs, dn_time_rs = densenet_stats_rs
    dn_test_prec_rs, dn_test_rec_rs, dn_test_f1_rs, *dn_test_rest_rs = get_metrics(densenet_rs, resampled_test_loader)
    dn_train_prec_rs, dn_train_rec_rs, dn_train_f1_rs, *dn_train_rest_rs = get_metrics(densenet_rs, resampled_train_loader)

    headers = ['Model', 'Precision', 'Recall', 'F1 Score']
    resampled_metrics = [
        ['linearnet (train)', ln_train_prec_rs, ln_train_rec_rs, ln_train_f1_rs],
        ['linearnet (test)', ln_test_prec_rs, ln_test_rec_rs, ln_test_f1_rs],
        ['convnet (train)', cn_train_prec_rs, cn_train_rec_rs, cn_train_f1_rs],
        ['convnet (test)', cn_test_prec_rs, cn_test_rec_rs, cn_test_f1_rs],
        ['resnet (train)', rn_train_prec_rs, rn_train_rec_rs, rn_train_f1_rs],
        ['resnet (test)', rn_test_prec_rs, rn_test_rec_rs, rn_test_f1_rs],
        ['densenet (train)', dn_train_prec_rs, dn_train_rec_rs, dn_train_f1_rs],
        ['densenet (test)', dn_test_prec_rs, dn_test_rec_rs, dn_test_f1_rs]
    ]
    save_table_as_image(headers, resampled_metrics, 'resampled_metrics.png')

    plot_loss(10, ln_t_loss_rs, ln_v_loss_rs, 'linearnet, resampled')
    plot_loss(10, cn_t_loss_rs, cn_v_loss_rs, 'convnet, resampled')
    plot_loss(10, rn_t_loss_rs, rn_v_loss_rs, 'resnet, resampled')
    plot_loss(10, dn_t_loss_rs, dn_v_loss_rs, 'densenet, resampled')

    plot_metric(10, ln_t_f1_rs, ln_v_f1_rs, 'f1 score with each epoch (linearnet, resampled)')
    plot_metric(10, cn_t_f1_rs, cn_v_f1_rs, 'f1 score with each epoch (convnet, resampled)')
    plot_metric(10, rn_t_f1_rs, rn_v_f1_rs, 'f1 score with each epoch (resnet, resampled)')
    plot_metric(10, dn_t_f1_rs, dn_v_f1_rs, 'f1 score with each epoch (densenet, resampled)')

    for param in convnet_rs.parameters():
        param.requires_grad = True
    for param in resnet_rs.parameters():
        param.requires_grad = True
    for param in densenet_rs.parameters():
        param.requires_grad = True
    visualize_heatmaps(convnet_rs, sample_img_pos, sample_input_pos)
    visualize_heatmaps(resnet_rs, sample_img_pos, sample_input_pos)
    visualize_heatmaps(densenet_rs, sample_img_pos, sample_input_pos)

    visualize_heatmaps(convnet_rs, sample_img_neg, sample_input_neg)
    visualize_heatmaps(resnet_rs, sample_img_neg, sample_input_neg)
    visualize_heatmaps(densenet_rs, sample_img_neg, sample_input_neg)

if __name__ == "__main__":
    main()

