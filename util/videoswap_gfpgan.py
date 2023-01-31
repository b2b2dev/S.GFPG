import os 
import torch
from torchvision.transforms.functional import normalize
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from basicsr.utils import tensor2img

gfpgan_transform_upsample = Compose([
    Resize([int(512), int(512)]),
    # ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])  

def gfpgan_downsampler(crop_size): 
  return Compose([
    Resize([int(crop_size), int(crop_size)]),
    # ToTensor(),
    # Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.chdir('/content/GFPGAN')
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')

if arch == 'clean':
  gfpgan = GFPGANv1Clean(
      out_size=512,
      num_style_feat=512,
      channel_multiplier=channel_multiplier,
      decoder_load_path=None,
      fix_decoder=False,
      num_mlp=8,
      input_is_latent=True,
      different_w=True,
      narrow=1,
      sft_half=True)

loadnet = torch.load(model_path)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
gfpgan.load_state_dict(loadnet[keyname], strict=True)
gfpgan.eval()
gfpgan = gfpgan.to(device)


os.chdir('/content/SimSwap')

'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet



def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    def _totensor(array):
        tensor = torch.from_numpy(array)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)

    gfpgan_transform_downsample = gfpgan_downsampler(crop_size)

    def gfpgan_enhance(img_tensor): 
      # gfp_t = normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
      # gfp_t = gfp_t.unsqueeze(0).to(device)
      img_tensor = gfpgan_transform_upsample(img_tensor.unsqueeze(0))
      output = gfpgan(img_tensor,return_rgb=True, weight=0.5)[0]
      # restored_face = tensor2img(output.squeeze(0), rgb2bgr=False, min_max=(-1, 1))
      down_img = tensor2img(gfpgan_transform_downsample(output).squeeze(0), rgb2bgr=True, min_max=(-1, 1))[:,:,[2,1,0]]
      return _totensor(down_img)


    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]
                swap_result_list = []
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    # print(swap_result.shape)
                    # input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    # img = cv2.resize(img, (512, 512))
                    # restore faces and background if necessary
                    
                    # these steps I think are identical to before
                    # cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    # normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    # cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                    # output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                    # # convert to image
                    # restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                    swap_result = gfpgan_enhance(swap_result)
                    # print(swap_result.shape)

                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.png'.format(frame_index)), frame)
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                    

                reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                    os.path.join(temp_results_dir, 'frame_{:0>7d}.png'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.png'.format(frame_index)), frame)
        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.png')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')
