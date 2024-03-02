#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import SimpleITK as sitk
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.fewshot import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
import cv2
import datetime
import matplotlib.pyplot as plt
from utils import *
from config import ex
import pandas as pd


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        # _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg(alpha=_config['alpha'])
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])

    # Loop over classes.
    class_dice = {}
    class_iou = {}

    df_empty = pd.DataFrame(columns=['label_name', 'sample_id', 'n_part_id', 'patch_token_id', 'query_image_s_', 'heatmap'])

    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], _config['test_label'])):
            continue

        _log.info(f'Test Class: {label_name}')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])
        test_dataset.label = label_val

        # Test.
        with torch.no_grad():
            model.eval()


            # Unpack support data.
            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]  # n_shot x H x W

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):

                # Unpack query data.
                query_image = [sample['image'][i].float().cuda() for i in
                               range(sample['image'].shape[0])]  # [C x 3 x H x W]
                ########
                # print('sample[image].shape', sample['image'].shape)
                # print('sample[image].shape', sample['image'].shape)
                # q_label = [sample['label'][i].float().cuda() for i in
                #                range(sample['image'].shape[0])]
                ###
                query_label = sample['label'].long()  # C x H x W
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # Compute output.
                # Match support slice and query sub-chunck.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]
                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for n_part_id, sub_chunck in enumerate(range(_config['n_part'])):
                    support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 3 x H x W
                    #
                    # query_l = q_label[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C'x H x W
                    # print('query_l.shape', query_l.shape)
                    query_pred_s = []
                    for patch_token_id, i in enumerate(range(query_image_s.shape[0])):
                        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
                        # q_mask = query_l[[i]]
                        # print('q_mask.shape', q_mask.shape)
                        # channel_mean_q_mask = torch.mean(q_mask, dim=1, keepdim=True)
                        # channel_mean_s_mask = F.interpolate(q_mask, size=img_size, mode='bilinear', align_corners=False)
                        # channel_mean_q_mask = q_mask.squeeze(0).cpu().numpy()  # 四维压缩为二维
                        # channel_mean_q_mask = (((channel_mean_q_mask - np.min(channel_mean_q_mask)) / (
                        #         np.max(channel_mean_q_mask) - np.min(channel_mean_q_mask))) * 255).astype(np.uint8)
                        # savedir = './vicPic/'
                        # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
                        # channel_mean_q_mask = cv2.applyColorMap(channel_mean_q_mask, cv2.COLORMAP_JET)
                        # cv2.imwrite(savedir + str('q_mask') + str(mkfile_time) + '.png', channel_mean_q_mask)
                        #################################################################
                        # query_image_s_ = query_image_s[[i]]
                        # query_image_s_save = query_image_s_.cpu().numpy()
                        ##################################################


                        # print('query_image_s_.shape', query_image_s_.shape)



                        # channel_query_image_s = torch.mean(query_image_s_, dim=1, keepdim=True)
                        # # channel_mean_s_mask = F.interpolate(query_image_s, size=img_size, mode='bilinear', align_corners=False)
                        # channel_query_image_s = channel_query_image_s.cpu().numpy()  # 四维压缩为二维
                        # channel_query_image_s = (((channel_query_image_s - np.min(channel_query_image_s)) / (
                        #         np.max(channel_query_image_s) - np.min(channel_query_image_s))) * 255).astype(np.uint8)
                        # savedir = './vicPic/'
                        # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
                        # channel_query_image_s = cv2.applyColorMap(channel_query_image_s, cv2.COLORMAP_JET)
                        # cv2.imwrite(savedir + str('query_image_s') + str(mkfile_time) + '.png', channel_query_image_s)

                        # support_image_s_ = support_image_s[i]
                        # # channel_mean_q_mask = torch.mean(support_image_s_, dim=1, keepdim=True)
                        # channel_support_image_s = torch.mean(support_image_s_, dim=1, keepdim=True)
                        # # channel_mean_s_mask = F.interpolate(query_image_s, size=img_size, mode='bilinear', align_corners=False)
                        # channel_support_image_s = channel_support_image_s.cpu().numpy()  # 四维压缩为二维
                        # channel_support_image_s = (((channel_support_image_s - np.min(channel_support_image_s)) / (
                        #         np.max(channel_support_image_s) - np.min(channel_support_image_s))) * 255).astype(np.uint8)
                        # savedir = './vicPic/'
                        # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
                        # channel_support_image_s = cv2.applyColorMap(channel_support_image_s, cv2.COLORMAP_JET)
                        # cv2.imwrite(savedir + str('support_image_s') + str(mkfile_time) + '.png', channel_support_image_s)

                        # _pred_s, _ , pred = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
                        #                    train=False, n_iters=_config['n_iters'])  # C x 2 x H x W
                        _pred_s, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
                                                 train=False, n_iters=_config['n_iters'])  # C x 2 x H x W
                        # pred_heatmap_array = pred.cpu().numpy()

                        # df_empty.loc[len(df_empty.index)] = [label_name, sample["id"][0], n_part_id, patch_token_id, query_image_s_save, pred_heatmap_array]

                        query_pred_s.append(_pred_s)
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s



                # Record scores.
                scores.record(query_pred, query_label)

                # Log.
                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')
                image_path = os.path.join(f'{sample["id"][0]}')
                image = sitk.ReadImage(image_path)
                size = image.GetSize()
                # print('size',size)
                origin = image.GetOrigin()
                spacing = image.GetSpacing()
                Direction = image.GetDirection()

                # Save predictions.
                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)

                # print(query_pred.shape)

                itk_pred.SetOrigin(origin)
                itk_pred.SetSpacing(spacing)
                itk_pred.SetDirection(Direction)
                sitk.WriteImage(itk_pred, file_name, True)
                # file_name_1 = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                #                          f'image_{query_id}_{label_name}.nii.gz')
                # itk_pred_1 = sitk.GetImageFromArray(sample['image'][0].permute(0, 2, 3, 1).float().cpu())
                # itk_pred_1.SetOrigin(origin)
                # itk_pred_1.SetSpacing(spacing)
                # itk_pred_1.SetDirection(Direction)
                # sitk.WriteImage(itk_pred_1, file_name_1, True)
                _log.info(f'{query_id} has been saved. ')
                # 可视化原型
                # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                # ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # ax[0].set_title('Original Image')
                # ax[0].axis('off')
                # ax[1].imshow(prototype, cmap='jet', alpha=0.5)  # 使用颜色映射（jet）和透明度（alpha）来表示原型
                # ax[1].set_title('Segmentation Prototype')
                # ax[1].axis('off')
                # plt.tight_layout()
                # plt.show()

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')

    _log.info(f'End of validation.')

    # df_empty.to_json('./heatmap/heatmap1.json')  # seaborn_heatmap

    return 1


