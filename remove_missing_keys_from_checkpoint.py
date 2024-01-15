import torch
import os

dictionary=torch.load("/mayo_atlas/home/m288756/stable-diffusion/models/ldm/inpainting_big/last.ckpt",map_location='cpu')

new_dict={}
keys=dictionary['state_dict'].keys()

for i,params in enumerate(keys):
    # if 'global_step' in params:
        # print(params)
    if 'ddim_sigmas' not in params and 'ddim_alphas' not in params and 'ddim_alphas_prev' not in params and 'ddim_sqrt_one_minus_alphas' not in params: #and 'cond_stage_model' not in params:
        new_dict[params]=dictionary['state_dict'][params]
        # print(i)
        # print(f"============================={params}")
    else:
        print(params)
        # print(dictionary['state_dict'][params].shape)
        # print(i)

# #If you want to change the conditional keys in the pretrained model to a personal conditional model with random weights.
# '''
# model=torch.nn.Linear(2048,640)

# for param in model.parameters():
# param.data = torch.randn(param.data.size())

# s_d=model.state_dict()
# keys=s_d.keys()
# for i,params in enumerate(keys):
# new_dict['cond_stage_model.model.'+params]=s_d[params]
# '''

dictionary['state_dict']=new_dict

torch.save(dictionary,"/mayo_atlas/home/m288756/stable-diffusion/models/ldm/inpainting_big/new_model.ckpt")