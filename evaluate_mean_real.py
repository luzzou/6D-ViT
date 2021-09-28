import os
import time
import argparse
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP_mean_real

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='/home/lz/code_local/6D-2sViT_tools/data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='/home/lz/code_local/6D-2sViT_tools/ckpt_6DViT_best/PIF_POF/batchsize_36_real_0908_sota+/model_75.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=256, help='cropped image size')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('/home/lz/code_local/6D-2sViT_tools/assets/mean_points_emb.npy')

assert opt.data in ['val', 'real_test']

if opt.data == 'real_test':
    result_dir = '/home/lz/Desktop/6D-ViT/results/6D-ViT_results/real_test'
    file_path = '/home/lz/code_local/6D-2sViT_tools/data/Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)



def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)  # 2754
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)  # 25
    iou_50_idx = iou_thres_list.index(0.5)  # 50
    iou_75_idx = iou_thres_list.index(0.75)  # 75
    degree_05_idx = degree_thres_list.index(5)  # 5
    degree_10_idx = degree_thres_list.index(10)  # 10
    shift_02_idx = shift_thres_list.index(2)  # 4
    shift_05_idx = shift_thres_list.index(5)  # 10
    shift_10_idx = shift_thres_list.index(10)  # 20
    messages = []
    messages.append('mAP:')
    messages.append('bottle:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[1, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[1, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[1, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[1, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[1, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[1, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[1, degree_10_idx, shift_10_idx]))

    messages.append('bowl:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[2, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[2, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[2, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[2, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[2, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[2, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[2, degree_10_idx, shift_10_idx]))

    messages.append('camera:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[3, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[3, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[3, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[3, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[3, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[3, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[3, degree_10_idx, shift_10_idx]))

    messages.append('can:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[4, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[4, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[4, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[4, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[4, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[4, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[4, degree_10_idx, shift_10_idx]))

    messages.append('laptop:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[5, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[5, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[5, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[5, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[5, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[5, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[5, degree_10_idx, shift_10_idx]))

    messages.append('mug:')
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[6, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[6, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[6, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[6, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[6, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[6, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[6, degree_10_idx, shift_10_idx]))

    messages.append('mean:')
    messages.append('3D IoU at 25: {:.4f}'.format(
        iou_aps[-1, iou_25_idx]))
    messages.append('3D IoU at 50: {:.4f}'.format(iou_aps[-1, iou_50_idx]))
    messages.append('3D IoU at 75: {:.4f}'.format(iou_aps[-1, iou_75_idx]))
    messages.append('5 degree, 2cm: {:.4f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx]))
    messages.append('5 degree, 5cm: {:.4f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx]))
    messages.append('10 degree, 2cm: {:.4f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx]))
    messages.append('10 degree, 5cm: {:.4f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx]))
    messages.append('10 degree, 10cm: {:.4f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx]))

    messages.append('mean:')
    messages.append('3D IoU at 25: {:.1f}'.format(
        iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))

    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


    # load spd results
    pkl_spd_path = os.path.join('/home/lz/Desktop/6D-ViT/results/spd_results/real_test', 'mAP_Acc.pkl')
    with open(pkl_spd_path, 'rb') as f:
        spd_results = cPickle.load(f)
    spd_iou_aps = spd_results['iou_aps'][-1, :]
    spd_pose_aps = spd_results['pose_aps'][-1, :, :]

    # load NOCS results
    pkl_path = os.path.join('/home/lz/Desktop/6D-ViT/results/nocs_results', opt.data, 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)
    nocs_iou_aps = nocs_results['iou_aps'][-1, :]
    nocs_pose_aps = nocs_results['pose_aps'][-1, :, :]

    # load Neural-Object-Fitting results
    pkl_path = os.path.join('/home/lz/Desktop/6D-ViT/results/nof_results', opt.data, 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nof_results = cPickle.load(f)
    nof_iou_aps = nof_results['iou_aps'][-1, :]
    nof_pose_aps = nof_results['pose_aps'][-1, :]

    iou_aps = iou_aps[-1, :]
    pose_aps = pose_aps[-1, :]  # 1,62,22

    iou_aps = np.concatenate((iou_aps[None, :], spd_iou_aps[None, :], nocs_iou_aps[None, :], nof_iou_aps[None, :]), axis=0)
    pose_aps = np.concatenate((pose_aps[None, :, :], spd_pose_aps[None, :, :], nocs_pose_aps[None, :, :], nof_pose_aps[None, :, :]), axis=0)

    # plot
    plot_mAP_mean_real(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子

    setup_seed(20)


    print('Evaluating ...')
    evaluate()
