import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap_gfpgan import video_swap
import os

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

opt = TestOptions()
opt.initialize()
opt.parser.add_argument('-f') ## dummy arg to avoid bug
opt = opt.parse()
opt.pic_a_path = '/path/to/desired/face.png' ## or replace it with image from your own google drive
opt.video_path = '/path/to/target/video.mp4' ## or replace it with video from your own google drive
opt.output_path = '/path/to/output/video.mp4'
opt.temp_path = './tmp'
opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
opt.isTrain = False
opt.use_mask = True  ## new feature up-to-date
opt.no_simswaplogo = True

start_epoch, epoch_iter = 1, 0
crop_sizes = [224,512]
opt.crop_size = crop_sizes[0]
crop_size = opt.crop_size

torch.nn.Module.dump_patches = True
if crop_size == 512:
    opt.which_epoch = 550000
    opt.name = '512'
    mode = 'ffhq'
else:
    mode = 'None'

model = create_model(opt)
model.eval()

app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
with torch.no_grad():
    pic_a = opt.pic_a_path
    # img_a = Image.open(pic_a).convert('RGB')
    img_a_whole = cv2.imread(pic_a)
    img_a_align_crop, _ = app.get(img_a_whole,crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    # pic_b = opt.pic_b_path
    # img_b_whole = cv2.imread(pic_b)

 
 
 
    # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
    # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
    # img_b = transformer(img_b_align_crop_pil)
    # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

    # convert numpy to tensor
    img_id = img_id.cuda()
    # img_att = img_att.cuda()

    #create latent id
    img_id_downsample = F.interpolate(img_id, size=(112,112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
        no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)
