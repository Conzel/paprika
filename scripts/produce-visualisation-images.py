import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inceptionv1_model = inceptionv1(pretrained=True)
_ = inceptionv1_model.to(device).eval()

filter_numbers = [256, 480, 508, 512, 512, 528, 832, 832, 1024]
image_size = 390
l_num = 0
for l in ['mixed3a','mixed3b','mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']:
  for i in range(filter_numbers[l_num]):
    _ = render.render_vis(inceptionv1_model, f'{l}:{i}', show_inline=True, save_image=True, image_name = f'../visualisations/{image_size}/{l}/{i}.jpg',param_f = lambda: param.image(image_size))
  l_num +=1
