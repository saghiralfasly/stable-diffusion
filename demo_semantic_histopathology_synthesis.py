import sys
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from omegaconf import OmegaConf
from einops import repeat
import albumentations

from scripts.sample_diffusion import load_model
# from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T

# from imwatermark import WatermarkEncoder

# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


# def put_watermark(img, wm_encoder=None):
#     if wm_encoder is not None:
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         img = wm_encoder.encode(img, 'dwtDct')
#         img = Image.fromarray(img[:, :, ::-1])
#     return img


@st.cache(allow_output_mutation=True)
# @st.cache_data(allow_output_mutation=True)
# @st.cache_resource(allow_output_mutation=True)
def initialize_model(config, ckpt):
    # config = OmegaConf.load(config)
    # model = instantiate_from_config(config.model)

    # model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)
    # sampler = DDIMSampler(model)

    return None #sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = sampler.model

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            # c = model.cond_stage_model.encode(batch["txt"])

            # c_cat = list()
            # for ck in model.concat_keys:
            #     cc = batch[ck].float()
            #     if ck != model.masked_image_key:
            #         bchw = [num_samples, 4, h // 8, w // 8]
            #         cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            #     else:
            #         cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            #     c_cat.append(cc)
            # c_cat = torch.cat(c_cat, dim=1)

            # # cond
            # cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # # uncond cond
            # uc_cross = model.get_unconditional_conditioning(num_samples, "")
            # uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            # shape = [model.channels, h // 8, w // 8]
            # samples_cfg, intermediates = sampler.sample(
            #     ddim_steps,
            #     num_samples,
            #     shape,
            #     cond,
            #     verbose=False,
            #     eta=1.0,
            #     unconditional_guidance_scale=scale,
            #     unconditional_conditioning=uc_full,
            #     x_T=start_code,
            # )
            # x_samples_ddim = model.decode_first_stage(samples_cfg)

            # result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
            #                      min=0.0, max=1.0)

            # result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    # return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def run():
    
    config_path = '/home/m288756/stable-diffusion/models/ldm/semantic_synthesis256/config_mine_inference_prompt.yaml'
    ckpt_path = '/home/m288756/stable-diffusion/logs_semantic_prompt/2024-01-20T14-12-43_config_train_prompt/checkpoints/epoch=000014.ckpt'
    n_labels = 2
    size = 256
    sample_to_generate = 3
    
    st.title("Visual-Prompted Latent Diffusion Model for Histopathology Image Generation")

    # sampler = initialize_model(sys.argv[1], sys.argv[2])
    sampler = initialize_model("yes", "yes")
    
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    # image = st.file_uploader("Image", ["jpg", "png"])
    # if True:
    # image = Image.open(image)
    # create a PIL Image that is all black
    image = Image.new('RGB', (512, 512))
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((width, height))

    # prompt = st.text_input("Prompt")

    # seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
    # num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
    # scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=10., step=0.1)
    ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=200, value=200, step=1)

    fill_color = "rgba(255, 255, 255, 0.0)"
    # stroke_width = st.number_input("Brush Size", value=100, min_value=1, max_value=100)
    stroke_width = st.slider("Brush Size", value=100, min_value=1, max_value=100)
    stroke_color = "rgba(255, 255, 255, 1.0)"
    bg_color = "rgba(0, 0, 0, 1.0)"
    drawing_mode = "freedraw"

    st.write("Histopathology Semantic Map")
    st.caption("Draw a region that to be the tumor region on the output generated image.")
    main_canvas_results = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=False,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        key="canvasMain",
    )
    

    cropped_image = None
    cropped_image2 = None
    
    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Upload a tumor histopathology image", type=["jpg", "jpeg", "png", "tiff", "tif"])

    if uploaded_image is not None:
        # Convert the uploaded image to PIL Image
        uploaded_image = Image.open(uploaded_image)
        
        # print(uploaded_image.size)
        # print("-------------")

        # Convert the uploaded image to NumPy array
        image_np = np.array(uploaded_image)
        w, h = image_np.shape[1], image_np.shape[0]

        # Create a canvas for drawing rectangles
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange color for the drawn rectangle
            stroke_width=2,
            stroke_color="rgb(255, 165, 0)",
            background_image=uploaded_image,
            drawing_mode="rect",
            key="canvas1",
            width=w,  # Set a specific width
            height=h,  # Set a specific height
        )

        # Check if the user has drawn a rectangle
        x, y, width, height = 0, 0, 0, 0
        if canvas_result.image_data is not None:
            # Extract rectangle coordinates and dimensions from JSON data
            objects = canvas_result.json_data.get('objects', [])
            
            # print(objects)
            # print(len(objects))
            
            if objects:
                last_object = objects[-1]
                x, y, width, height = (
                    int(last_object.get('left', 0)),
                    int(last_object.get('top', 0)),
                    int(last_object.get('width', 0)),
                    int(last_object.get('height', 0))
                )
                
                # crop the image using the rectangle coordinates from PIL Image
                cropped_image = uploaded_image.crop((x, y, x+width, y+height))

                # # Crop the image using the rectangle coordinates
                # cropped_image = image_np[y:y+height, x:x+width]

                # # Display the cropped image
                # st.image(cropped_image, caption="Normal Prompt", width=200) # ************************************

                # # Display rectangle coordinates and dimensions
                # st.write(f"Normal tissue prompt size: Width X Height: {width} X {height}")
                # st.write("Rectangle Coordinates:")
                # st.write(f"X: {x}, Y: {y}")
                # st.write(f"Width: {width}, Height: {height}")
                
                # # Print the canvas size
                # canvas_height, canvas_width, _ = canvas_result.image_data.shape
                # st.write(f"Canvas Size: {canvas_width} x {canvas_height}")


    # Upload image through Streamlit
    uploaded_image2 = st.file_uploader("Upload a normal histopathology image", type=["jpg", "jpeg", "png", "tiff", "tif"])
    x2, y2, width2, height2 = 0, 0, 0, 0
    if uploaded_image2 is not None:
        # Convert the uploaded image to PIL Image
        uploaded_image2 = Image.open(uploaded_image2)
        
        # print(uploaded_image.size)
        # print("-------------")
        # Convert the uploaded image to NumPy array
        image_np2 = np.array(uploaded_image2)
        w2, h2 = image_np2.shape[1], image_np2.shape[0]

        # Create a canvas for drawing rectangles
        canvas_result2 = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange color for the drawn rectangle
            stroke_width=2,
            stroke_color="rgb(255, 165, 0)",
            background_image=uploaded_image2,
            drawing_mode="rect",
            key="canvas2",
            width=w2,  # Set a specific width
            height=h2,  # Set a specific height
        )

        # Check if the user has drawn a rectangle
        if canvas_result2.image_data is not None:
            # Extract rectangle coordinates and dimensions from JSON data
            objects2 = canvas_result2.json_data.get('objects', [])
            
            # print(objects)
            # print(len(objects))
            
            if objects2:
                last_object2 = objects2[-1]
                x2, y2, width2, height2 = (
                    int(last_object2.get('left', 0)),
                    int(last_object2.get('top', 0)),
                    int(last_object2.get('width', 0)),
                    int(last_object2.get('height', 0))
                )
                
                # crop the image using the rectangle coordinates from PIL Image
                cropped_image2 = uploaded_image2.crop((x2, y2, x2+width2, y2+height2))

                # # Crop the image using the rectangle coordinates
                # cropped_image = image_np[y:y+height, x:x+width]

                # # Display the cropped image
                # st.image(cropped_image2, caption="Tumor Prompt", width=200) # ************************************

                # # Display rectangle coordinates and dimensions
                # st.write(f"Tumor prompt size:  Width X Height:  {width2} X {height2}")
                # st.write(f"X: {x2}, Y: {y2}")
                # st.write(f"Width: {width2}, Height: {height2}")


    if main_canvas_results and cropped_image is not None and cropped_image2 is not None:
    # get the center of the drawn region
        mask = main_canvas_results.image_data
        # print(mask)
        mask = mask[:, :, -1] == 255
        mask_for_inference = mask.copy()
        mask_for_inference = Image.fromarray(mask_for_inference)
        mask_for_inference = np.array(mask_for_inference)
        mask_for_inference = mask_for_inference.astype(np.uint8)
        if mask.sum() > 0:
            mask = Image.fromarray(mask)
            mask = np.array(mask)
            mask = np.where(mask==True)
            xn = int(np.mean(mask[0]))
            yn = int(np.mean(mask[1]))
            print(f"center of the Normal region: ({xn}, {yn})")
            
            # create a mask_normal that is the same size as the mask2
            mask_normal = np.zeros((512, 512, 3))
            # convert mask_normal to PIL Image
            mask_normal = Image.fromarray(mask_normal.astype(np.uint8))

            x_cornern = xn - width//2
            y_cornern = yn - height//2
            mask_normal.paste(cropped_image, (y_cornern, x_cornern))
            # # now show the mask_normal
            # st.image(mask_normal, caption="Normal Prompt", width=512) # ************************************

            # get the center of the background region
            mask2 = main_canvas_results.image_data
            mask2 = mask2[:, :, -1] == 0
            if mask2.sum() > 0:
                mask2_PIL = Image.fromarray(mask2)
                mask2_array = np.array(mask2_PIL)
                mask2 = np.where(mask2_array==True)
                x2in = int(np.mean(mask2[0]))
                y2in = int(np.mean(mask2[1]))
                print(f"center of the normal region: ({x2in}, {y2in})")
                
                # create a mask_tumor that is the same size as the mask2
                mask_tumor = np.zeros((512, 512, 3))
                # convert mask_tumor to PIL Image
                mask_tumor = Image.fromarray(mask_tumor.astype(np.uint8))
                x_corner = x2in - width2//2
                y_corner = y2in - height2//2
                # mask_tumor.paste(cropped_image2, (x_corner, y_corner))
                mask_tumor.paste(cropped_image2, (y_corner, x_corner))
                # # now show the mask_tumor
                # st.image(mask_tumor, caption="Tumor Prompt", width=512) # ************************************

                if mask_tumor is not None and mask_normal is not None:
                    
                    # seg = x['segmentation']
                    # seg2 = x['segmentation']
                    # img = x['image']
                    
                    # print(type(image))
                    # if not image.mode == "RGB":
                    image = image.convert("RGB")
                    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        
                    image = np.array(image).astype(np.uint8)
                    # print(image)
                    new_tumor_crop = np.array(mask_tumor) 
                    new_not_tumor_crop = np.array(mask_normal)
                    
                    # print(type(image))
                    # print(image.shape)
                    # print(type(new_tumor_crop))
                    # print(new_tumor_crop.shape)
                    # print(type(new_not_tumor_crop))
                    # print(new_not_tumor_crop.shape)
                    # # print(len(mask_for_inference))
                    # print(type(mask_for_inference))
                    # # print(mask_for_inference)
                    # print(mask_for_inference.shape)
                    # print("------------------------------")

                    # image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation="bicubic")
                    # segmentation_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_NEAREST)


                    segmentation = mask_for_inference
                    if size is not None:
                        # resie image with shape (512, 512, 3) to (256, 256, 3)
                        image = Image.fromarray(image)
                        segmentation = Image.fromarray(segmentation)
                        new_not_tumor_crop = Image.fromarray(new_not_tumor_crop)
                        new_tumor_crop = Image.fromarray(new_tumor_crop)
                        
                        # resize image and segmentation where they are numpy to 256
                        image = image.resize((size, size), Image.BICUBIC)
                        # print(image)
                        # print(image.size)
                        segmentation = segmentation.resize((size, size), Image.NEAREST)
                        new_not_tumor_crop = new_not_tumor_crop.resize((size, size), Image.BICUBIC)
                        new_tumor_crop = new_tumor_crop.resize((size, size), Image.BICUBIC)
                        # image = image_rescaler(image=image)["image"]
                        # segmentation = segmentation_rescaler(image=segmentation)["image"]
                    # convert all to numpy again
                    image = np.array(image)
                    segmentation = np.array(segmentation)
                    new_not_tumor_crop = np.array(new_not_tumor_crop)
                    new_tumor_crop = np.array(new_tumor_crop)
                    
                    new_tumor_crop = (new_tumor_crop/127.5 - 1.0).astype(np.float32)
                    new_not_tumor_crop = (new_not_tumor_crop/127.5 - 1.0).astype(np.float32)
                    

            
                    # print(type(image))
                    # print(image.shape)
                    # print(type(new_tumor_crop))
                    # print(new_tumor_crop.shape)
                    # print(type(new_not_tumor_crop))
                    # print(new_not_tumor_crop.shape)
                    # # print(len(mask_for_inference))
                    # print(type(mask_for_inference))
                    # print(mask_for_inference.shape)
                    
                    # img = (image/127.5 - 1.0).astype(np.float32)
                    img = image
                    # print(img)

                    onehot = np.eye(n_labels)[segmentation]    
                    
                    # # print(onehot.shape)                
                    # # print(new_not_tumor_crop[:,:,0].tolist())
                    # # save the new_not_tumor_crop and new_tumor_crop to the txt file
                    # np.savetxt("new_not_tumor_crop.txt", new_not_tumor_crop.reshape(1, -1))
                    # np.savetxt("new_tumor_crop.txt", new_tumor_crop.reshape(1, -1))
                    
                    # convert each -1 to 0
                    new_not_tumor_crop[new_not_tumor_crop == -1] = 0
                    new_tumor_crop[new_tumor_crop == -1] = 0
                    

                    # now I want to create segment_regions which contains last dimension as follows [:,:,-l]: 
                    #        the first onehot segmentation with index 0, then the tumor_crop, 
                    #        then the second onehot segmentation with index 1, then the not_tumor_crop
                    seg = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), new_tumor_crop, np.expand_dims(onehot[:,:,1], axis=-1), new_not_tumor_crop), axis=-1) 
                    
                    # convert seg and img to torch tensors
                    seg = torch.from_numpy(seg).permute(2, 0, 1).unsqueeze(0)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    # print(img)
            
            
                    with torch.no_grad():        
                        # img = rearrange(img, 'b h w c -> b c h w')
                        # img = img.to('cuda').float()
                        
                        # print(img.shape)
                        # print(type(img))
                        # print(seg.shape)
                        print("++++++++++++++++++q+++++++++++++++++++++")
                        
                        # seg = rearrange(seg, 'h w c -> c h w')
                        # print(seg[ 0, :, :].unsqueeze(0).unsqueeze(0).shape)
                        # condition = model.to_rgb(seg[0, 0, :, :].unsqueeze(0).unsqueeze(0))
                        # convert the condition to numpy with shape (256, 256, 3)
                        condition = segmentation#.unsqueeze(0).numpy().transpose(2,1,0)
                        
                        # condition = model.to_rgb(seg[0, 4, :, :].unsqueeze(0).unsqueeze(0))
                        tumor_prompt = seg[0, 1:4, :, :]
                        Non_tumor_prompt = seg[0, 5:, :, :]

                        seg = seg.to('cuda').float()
                        seg = model.get_learned_conditioning(seg)
                        

                        output_samples = []
                        for _ in range(sample_to_generate):
                            samples, _ = model.sample_log(cond=seg, batch_size=1, ddim=True,
                                    ddim_steps=ddim_steps, eta=1.)
                            sample = model.decode_first_stage(samples)
                            output_samples.append(sample)
                                                

                        # Save images in two rows using plt
                        plt.figure(figsize=(12, 4))

                        # # First row: Original Image, Generated Mask, Masked Image
                        # plt.subplot(1, 5, 1)
                        # # Transpose the image data if needed
                        # # image_data = np.transpose(image_data, (1, 2, 0))
                        # print(np.transpose(img[0], (1, 2, 0)))
                        # plt.imshow(np.transpose(img[0], (1, 2, 0)))
                        # plt.title('Original Image')
                        # plt.axis('off')
                        
                        # print(seg2.transpose(1,2,0)[0].shape)
                        plt.subplot(1, 3, 1)
                        # plt.imshow(seg2.transpose(1,2,0)[0])
                        plt.imshow(condition)
                        # plt.imshow(condition.squeeze().numpy().transpose(2,1,0))
                        plt.title('Segmentation Map')
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        # print(tumor_prompt.numpy().transpose(1,2,0)[:,:,-1].tolist())
                        plt.imshow(tumor_prompt.numpy().transpose(1,2,0) * 255)
                        plt.title('Normal Prompt')
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.imshow(Non_tumor_prompt.numpy().transpose(1,2,0) * 255)
                        plt.title('Tumor Prompt')
                        plt.axis('off')

                        # plt.subplot(1, 4, 4)
                        # # plt.imshow(samples[0].cpu().numpy().transpose(1,2,0))
                        # plt.imshow(output2)
                        # plt.title('Generated Image')
                        # plt.axis('off')
                        
                        # show the plot on the screen
                        st.pyplot(plt)
                        
                        for i in range(sample_to_generate):
                            samples = output_samples[i]
                            output = samples[0].cpu().numpy().transpose(1,2,0)
                            
                            # save output to png file
                            # first denormalize the output
                            output = ((output + 1.0)/2) * 255
                            # clip the output to be between 0 and 255
                            output2 = np.clip(output.astype(np.uint8), 0, 255)
                            # then convert output to PIL Image
                            output = Image.fromarray(output2)
                            # convert output to RGB
                            # output = output.convert("RGB")
                            # save the output to png file
                            output.save(f"output{i}.png")
                            st.image(output, caption=f"Output Sample {i+1}", width=512)
                            # cv2.imwrite("aoutput.png", output)
                            # print(output)
                        
                        # print(type(output))
                        # print(output.shape)
                        
                        
                        
                        # plt.show()

                        # plt.savefig(f"{outputDir}/sample_{i}_.png")
                        # plt.close()
                        # print(x['file_path_'])
                    
if __name__ == "__main__":
    run()
