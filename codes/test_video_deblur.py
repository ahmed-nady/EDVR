

#Usage :
# python 
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

def prepare_frame(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img
def get_Torch_tensor_from_prep_frame_lst(prep_frame_lst):
    """Read a sequence of images from a given folder path
    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    # stack to Torch tensor
    imgs = np.stack(prep_frame_lst, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

# Playing video from file:

cap = cv2.VideoCapture(r'/content/EDVR/datasets/videos/camera2_20101027T210200+02.mjpeg')
#camera2_20101027T210200+02.mjpeg')



 #################
# configurations
#################
device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_mode = 'blur_comp'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
# Vid4: SR
# REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
#        blur (deblur-clean), blur_comp (deblur-compression).
stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
flip_test = False
############################################################################
#### model
if data_mode == 'Vid4':
    if stage == 1:
        model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    else:
        raise ValueError('Vid4 does not support stage 2.')
elif data_mode == 'sharp_bicubic':
    if stage == 1:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SR_L.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'
elif data_mode == 'blur_bicubic':
    if stage == 1:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth'
elif data_mode == 'blur':
    if stage == 1:
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth'
elif data_mode == 'blur_comp':
    if stage == 1:
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth'
else:
    raise NotImplementedError

if data_mode == 'Vid4':
    N_in = 7  # use N_in images to restore one HR image
else:
    N_in = 5

predeblur, HR_in = False, False
back_RBs = 40
if data_mode == 'blur_bicubic':
    predeblur = True
if data_mode == 'blur' or data_mode == 'blur_comp':
    predeblur, HR_in = True, True
if stage == 2:
    HR_in = True
    back_RBs = 20
model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

#### dataset
if data_mode == 'Vid4':
    test_dataset_folder = '../datasets/Vid4/BIx4'
    GT_dataset_folder = '../datasets/Vid4/GT'
else:
    if stage == 1:
        test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
    else:
        test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
        print('You should modify the test_dataset_folder path for stage 2')
    GT_dataset_folder = '../datasets/REDS4/GT'

#### evaluation
crop_border = 0
border_frame = N_in // 2  # border frames when evaluate
# temporal padding mode
if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
    padding = 'new_info'
else:
    padding = 'replicate'
save_imgs = True

save_folder = '../results/{}'.format(data_mode)
save_subfolder = osp.join(save_folder, 'images')
util.mkdirs(save_folder)
util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')

#### log info
logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
logger.info('Padding mode: {}'.format(padding))
logger.info('Model path: {}'.format(model_path))
logger.info('Save images: {}'.format(save_imgs))
logger.info('Flip test: {}'.format(flip_test))
#### set up the models
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

currentFrame = 0
count= 0
prep_frame_lst =[]
if save_imgs:
            util.mkdirs(save_subfolder)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    if not ret:
        print('Done processing')
        break
    prep_frame =prepare_frame(frame)
    currentFrame += 1
    if not (currentFrame%15==0):
        prep_frame_lst.append(prep_frame)
    else:
        prep_frame_lst.append(prep_frame)
         # stack to Torch tensor
        imgs_LQ = get_Torch_tensor_from_prep_frame_lst(prep_frame_lst)

         # process each image
        for img_idx in range(1,16):
            count +=1
            img_name =  "frame_%05i" % count
            select_idx = data_util.index_generation(img_idx, 15, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

        prep_frame_lst =[]

    # To stop duplicate images
    

# When everything done, release the capture
cap.release()
