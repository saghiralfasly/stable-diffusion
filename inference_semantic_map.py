import torch
import numpy as np

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
transform_PIL = T.ToPILImage()

# from ldm.data.flickr import FlickrSegEval
from ldm.data.Histopathology import HistoValPrompt
# from ldm.data.Histopathology import HistoVal

def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(50):

        x = next(iter(dataloader))

        seg = x['segmentation']

        with torch.no_grad():
            seg = rearrange(seg, 'b h w c -> b c h w')
            condition = model.to_rgb(seg)

            seg = seg.to('cuda').float()
            seg = model.get_learned_conditioning(seg)

            samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                        ddim_steps=200, eta=1.)

            samples = model.decode_first_stage(samples)

        save_image(condition, f'outputs/semantic/out_cond__32_{i}.png')
        save_image(samples, f'outputs//semantic/out_sample__32_{i}.png')



def ldm_cond_sample_prompt(config_path, ckpt_path, dataset, batch_size, outputDir):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(1):

        x = next(iter(dataloader))

        seg = x['segmentation']
        seg2 = x['segmentation']
        img = x['image']


        # with torch.no_grad():        
        #     # img = rearrange(img, 'b h w c -> b c h w')
        #     # img = img.to('cuda').float()
            
        #     print(img.shape)
        #     print(type(img))
        #     # print(img)
        #     print(seg.shape)
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
        #     seg = rearrange(seg, 'b h w c -> b c h w')
        #     condition = model.to_rgb(seg[0, 0, :, :].unsqueeze(0).unsqueeze(0))
        #     # condition = model.to_rgb(seg[0, 4, :, :].unsqueeze(0).unsqueeze(0))
        #     tumor_prompt = seg[0, 1:4, :, :]
        #     Non_tumor_prompt = seg[0, 5:, :, :]
            
        #     # print(tumor_prompt.numpy().tolist())

        #     seg = seg.to('cuda').float()
        #     seg = model.get_learned_conditioning(seg)

        #     samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
        #                                 ddim_steps=200, eta=1.)

        #     samples = model.decode_first_stage(samples)
            
        #     # denormalize the output
        #     # predicted_image_clamped = torch.clamp(samples[0],
        #     #                             min=0.0, max=1.0)
        #     # output_PIL=transform_PIL(samples[0])


        #     # Save images in two rows using plt
        #     plt.figure(figsize=(12, 4))

        #     # First row: Original Image, Generated Mask, Masked Image
        #     # print(img.shape)
        #     # print(type(img))
        #     # print((img[0].cpu()).numpy().transpose(2,1,0))
        #     plt.subplot(1, 5, 1)
        #     plt.imshow(img[0])
        #     plt.title('Original Image')
        #     plt.axis('off')

        #     # print(condition.shape)
        #     # print(type(condition))
            
        #     # print(seg2.transpose(1,2,0)[0].shape)
        #     plt.subplot(1, 5, 2)
        #     # plt.imshow(seg2.transpose(1,2,0)[0])
        #     plt.imshow(condition.squeeze().numpy().transpose(2,1,0))
        #     plt.title('Segmentation Map')
        #     plt.axis('off')

        #     plt.subplot(1, 5, 3)
        #     plt.imshow(tumor_prompt.numpy().transpose(1,2,0))
        #     plt.title('Normal Prompt')
        #     plt.axis('off')

        #     plt.subplot(1, 5, 4)
        #     plt.imshow(Non_tumor_prompt.numpy().transpose(1,2,0))
        #     plt.title('Tumor Prompt')
        #     plt.axis('off')

        #     plt.subplot(1, 5, 5)
        #     # print(samples[0].cpu().numpy().shape)
        #     # print("&&&&&&&&&&&&&&&&&&&")
        #     plt.imshow(samples[0].cpu().numpy().transpose(1,2,0))
        #     plt.title('Generated Image')
        #     plt.axis('off')
            
        #     plt.show()

        #     # plt.savefig(f"{outputDir}/sample_{i}_.png")
        #     plt.savefig(f"{outputDir}/sample_xxx_.png")
        #     plt.close()
        #     print(x['file_path_'])
    
    
    
    
    
        # # save_image(condition, f'outputs/semantic/out_cond__18_{i}.png')
        # save_image(samples, f'outputs/semantic/prompt/out_sample__18_{i}.png')

# if __name__ == '__main__':

#     config_path = '/home/m288756/stable-diffusion/models/ldm/semantic_synthesis256/config_mine_inference.yaml'
#     ckpt_path = '/home/m288756/stable-diffusion/logs_semantic/2024-01-17T14-10-19_config_train/checkpoints/epoch=000032.ckpt'

#     dataset = HistoVal(size=256)

#     ldm_cond_sample(config_path, ckpt_path, dataset, 4)
    
    
if __name__ == '__main__':

    config_path = '/home/m288756/stable-diffusion/models/ldm/semantic_synthesis256/config_mine_inference_prompt.yaml'
    # ckpt_path = '/home/m288756/stable-diffusion/logs_semantic_prompt/2024-01-17T18-48-41_config_train_prompt/checkpoints/epoch=000018.ckpt'
    ckpt_path = '/home/m288756/stable-diffusion/logs_semantic_prompt/2024-01-20T14-12-43_config_train_prompt/checkpoints/epoch=000014.ckpt'
    # outputDir = 'outputs/semantic/prompts_updated'
    outputDir = 'outputs'
    # create output directory
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    dataset = HistoValPrompt(size=256)

    ldm_cond_sample_prompt(config_path, ckpt_path, dataset, 1, outputDir)
