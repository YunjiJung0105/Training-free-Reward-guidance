from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from rewards import RFUNCTIONS
import random
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import argparse

DNUM = 0
devices = ["cuda:0"]
device = torch.device(devices[DNUM] if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
class SequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 
        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]   # [1, 21, 41, 61, 81, 101, ... 901, 921, 941, 961, 981]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]  # [0, 1, 21, 41, 61, 81, ... 901, 921, 941, 961]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]  #  [tensor(0.0017), tensor(0.0196), tensor(0.0391), tensor(0.0602), tensor(0.0829), tensor(0.1070), tensor(0.1326), tensor(0.1596), tensor(0.1879), tensor(0.2173), tensor(0.2479), tensor(0.2793), tensor(0.3115), tensor(0.3443), tensor(0.3776), tensor(0.4112), tensor(0.4449), tensor(0.4785), tensor(0.5118), tensor(0.5448), tensor(0.5771), tensor(0.6087), tensor(0.6395), tensor(0.6692), tensor(0.6977), tensor(0.7250), tensor(0.7510), tensor(0.7755), tensor(0.7986), tensor(0.8201), tensor(0.8402), tensor(0.8587), tensor(0.8757), tensor(0.8913), tensor(0.9054), tensor(0.9181), tensor(0.9295), tensor(0.9396), tensor(0.9486), tensor(0.9565), tensor(0.9635), tensor(0.9695), tensor(0.9746), tensor(0.9790), tensor(0.9828), tensor(0.9860), tensor(0.9887), tensor(0.9909), tensor(0.9927), tensor(0.9942)]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]  #  [tensor(0.0009), tensor(0.0017), tensor(0.0196), tensor(0.0391), tensor(0.0602),  tensor(0.9887), tensor(0.9909), tensor(0.9927)]
        # print('self.scheduler_timesteps: ', self.scheduler_timesteps)
        # print('scheduler_prev_timesteps: ', scheduler_prev_timesteps)
        # print('alphas_cumprod: ', alphas_cumprod)
        # print('alphas_cumprod_prev: ', alphas_cumprod_prev)

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)   #same with alphas_cumprod
        next_coeff = torch.clamp(next_coeff, min = 0)   #same with alphas_cumprod_prev
        # print('now_coeff: ',now_coeff)
        # print('next_coeff: ', next_coeff)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)   # tensor([0.9983, 0.9804, 0.9609, 0.9398, 0.9171, 0.8930, 0.8674, 0.8404, 0.8121,0.7827, 0.7521, 0.7207, 0.6885, 0.6557, 0.6224, 0.5888, 0.5551, 0.5215,0.4882, 0.4552, 0.4229, 0.3913, 0.3605, 0.3308, 0.3023, 0.2750, 0.2490,0.2245, 0.2014, 0.1799, 0.1598, 0.1413, 0.1243, 0.1087, 0.0946, 0.0819,0.0705, 0.0604, 0.0514, 0.0435, 0.0365, 0.0305, 0.0254, 0.0210, 0.0172,0.0140, 0.0113, 0.0091, 0.0073, 0.0058])
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)  # tensor([0.9991, 0.9983, 0.9804, 0.9609, 0.9398, 0.9171, 0.8930, 0.8674, 0.8404,0.8121, 0.7827, 0.7521, 0.7207, 0.6885, 0.6557, 0.6224, 0.5888, 0.5551,0.5215, 0.4882, 0.4552, 0.4229, 0.3913, 0.3605, 0.3308, 0.3023, 0.2750,0.2490, 0.2245, 0.2014, 0.1799, 0.1598, 0.1413, 0.1243, 0.1087, 0.0946,0.0819, 0.0705, 0.0604, 0.0514, 0.0435, 0.0365, 0.0305, 0.0254, 0.0210,0.0172, 0.0140, 0.0113, 0.0091, 0.0073])
        # print('m_now_coeff: ', m_now_coeff)
        # print('m_next_coeff: ', m_next_coeff)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.  # tensor([0.0000, 0.0395, 0.0999, 0.1194, 0.1323, 0.1428, 0.1521, 0.1608, 0.1691,0.1771, 0.1849, 0.1926, 0.2002, 0.2077, 0.2151, 0.2226, 0.2299, 0.2372,0.2445, 0.2518, 0.2590, 0.2662, 0.2734, 0.2806, 0.2877, 0.2948, 0.3019,0.3089, 0.3159, 0.3229, 0.3298, 0.3368, 0.3436, 0.3505, 0.3573, 0.3641,0.3708, 0.3775, 0.3841, 0.3908, 0.3973, 0.4038, 0.4103, 0.4167, 0.4231,0.4295, 0.4358, 0.4421, 0.4483, 0.4545])
        # print('self.nl: ', self.nl)
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x
        self.sqrt_m_now_coeff = torch.sqrt(m_now_coeff)
        self.sqrt_now_coeff = torch.sqrt(now_coeff)
        print('m_nl_next_coeff: ', m_nl_next_coeff)  #  tensor([8.5002e-04, 1.4534e-04, 9.6418e-03, 2.4866e-02, 4.2706e-02, 6.2468e-02,8.3878e-02, 1.0678e-01, 1.3104e-01, 1.5654e-01, 1.8316e-01, 2.1077e-01,2.3922e-01, 2.6836e-01, 2.9804e-01, 3.2809e-01, 3.5832e-01, 3.8857e-01,4.1867e-01, 4.4842e-01, 4.7767e-01, 5.0624e-01, 5.3398e-01, 5.6074e-01,5.8639e-01, 6.1081e-01, 6.3388e-01, 6.5554e-01, 6.7569e-01, 6.9431e-01,7.1134e-01, 7.2677e-01, 7.4062e-01, 7.5288e-01, 7.6360e-01, 7.7282e-01,7.8059e-01, 7.8698e-01, 7.9208e-01, 7.9595e-01, 7.9868e-01, 8.0037e-01,8.0109e-01, 8.0094e-01, 8.0000e-01, 7.9835e-01, 7.9607e-01, 7.9325e-01,7.8993e-01, 7.8618e-01])
        print('self.coeff_x:', self.coeff_x)  # tensor([1.0004, 1.0091, 1.0101, 1.0112, 1.0123, 1.0134, 1.0147, 1.0159, 1.0173,1.0186, 1.0201, 1.0216, 1.0231, 1.0247, 1.0264, 1.0281, 1.0299, 1.0317,1.0336, 1.0355, 1.0376, 1.0396, 1.0417, 1.0439, 1.0462, 1.0485, 1.0508,1.0532, 1.0557, 1.0583, 1.0609, 1.0635, 1.0663, 1.0691, 1.0719, 1.0748,1.0778, 1.0809, 1.0840, 1.0872, 1.0904, 1.0938, 1.0972, 1.1006, 1.1041,1.1077, 1.1114, 1.1151, 1.1190, 1.1229])
        print('self.coeff_d:', self.coeff_d)  # tensor([-0.0121, -0.1293, -0.1016, -0.0904, -0.0847, -0.0816, -0.0799, -0.0791,-0.0790, -0.0792, -0.0799, -0.0808, -0.0819, -0.0833, -0.0848, -0.0865,-0.0883, -0.0903, -0.0924, -0.0947, -0.0971, -0.0996, -0.1023, -0.1051,-0.1081, -0.1112, -0.1144, -0.1179, -0.1214, -0.1251, -0.1290, -0.1330,-0.1372, -0.1416, -0.1461, -0.1508, -0.1556, -0.1606, -0.1658, -0.1711,-0.1766, -0.1823, -0.1881, -0.1941, -0.2002, -0.2064, -0.2129, -0.2194,-0.2261, -0.2329])
        print('self.sqrt_m_now_coeff: ', self.sqrt_m_now_coeff) # tensor([0.9991, 0.9901, 0.9802, 0.9694, 0.9577, 0.9450, 0.9313, 0.9167, 0.9012,0.8847, 0.8673, 0.8489, 0.8298, 0.8097, 0.7889, 0.7673, 0.7451, 0.7222,0.6987, 0.6747, 0.6503, 0.6255, 0.6005, 0.5752, 0.5498, 0.5244, 0.4990,0.4738, 0.4488, 0.4241, 0.3998, 0.3759, 0.3525, 0.3298, 0.3076, 0.2862,0.2655, 0.2457, 0.2266, 0.2085, 0.1912, 0.1748, 0.1593, 0.1447, 0.1311,0.1183, 0.1065, 0.0955, 0.0853, 0.0760])
        print('self.sqrt_now_coeff: ', self.sqrt_now_coeff) # tensor([0.0413, 0.1401, 0.1978, 0.2454, 0.2879, 0.3271, 0.3642, 0.3995, 0.4335,0.4662, 0.4979, 0.5285, 0.5581, 0.5868, 0.6145, 0.6412, 0.6670, 0.6917,0.7154, 0.7381, 0.7597, 0.7802, 0.7997, 0.8180, 0.8353, 0.8515, 0.8666,0.8806, 0.8936, 0.9056, 0.9166, 0.9267, 0.9358, 0.9441, 0.9515, 0.9582,0.9641, 0.9694, 0.9740, 0.9780, 0.9816, 0.9846, 0.9872, 0.9895, 0.9914,0.9930, 0.9943, 0.9954, 0.9964, 0.9971])

        self.grads = []

    def get_tweedie(self,z,noise_pred,t):
        z0 = z - self.sqrt_now_coeff[t] * noise_pred
        z0 = z0 / self.sqrt_m_now_coeff[t]
        return z0


    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
        #torch.set_grad_enabled(True)
        
        # _sample = self._samples[0].detach().requires_grad(True)
        _sample = self._samples[0]

        model_kwargs = {
            "sample": [_sample,  _sample],
            "timestep": torch.tensor(t, device = self.device),
            "encoder_hidden_states": prompt_embeds
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

        return model_kwargs


    def step(self, sample, pipeline,timestep,temb, loss_fn=None,condition=None):
        
        torch.set_grad_enabled(True)
        sample = sample.detach()
        sample.requires_grad_(True)
        model_output = pipeline.unet(sample, timestep, temb)   
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)

        t = self.num_steps - len(self._samples)

        grad = 0
        tweedie_decode = None
        if loss_fn != None:
            s = sample[:sample.shape[0]//2]
            tweedie = self.get_tweedie(s,direction,t)   
            tweedie_decode = decode_latent(pipeline.vae,tweedie)


            loss = loss_fn(tweedie_decode, condition)
            grad = torch.autograd.grad(loss,s)[0]
            self.grads.append(grad[0,0].flatten().detach())

            
        torch.set_grad_enabled(False)

        if t <= self.opt_timesteps:   # By KJY
        # if t >= self.opt_timesteps:     # By JYJ
            now_sample = self.coeff_x[t] * self._samples[0] +  self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t] - 2 * self.sqrt_now_coeff[t] * grad   
            # print('now_sample')
            # print(now_sample.shape)   #[2,4,64,64]
            # print(torch.min(now_sample), torch.max(now_sample))
            # print(self.coeff_x[t]) 
            # print('self._samples[0]')
            # print(self._samples[0].shape)   [2,4,64,64]
            # print(torch.min(self._samples[0]), torch.max(self._samples[0]))
            # print(self.coeff_d[t])   
            # print('direction')
            # print(direction.shape)    #[2,4,64,64]
            # print(torch.min(direction), torch.max(direction))
            # print(self.nl[t])    
            # print('self.noise_vectors[t]')
            # print(self.noise_vectors[t].shape)  [4,64,64]
            # print(torch.min(self.noise_vectors[t]), torch.max(self.noise_vectors[t]))
            # print(self.sqrt_now_coeff[t])   
            # print('grad')
            # print(grad.shape)   #[2,4,64,64]
            # print(torch.min(grad), torch.max(grad))

        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] +  self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]  - 2 * self.sqrt_now_coeff[t] * grad  
            # print(now_sample.shape)
            # print(self.coeff_x[t].shape)
            # print(self._samples[0].shape)
            # print(self.coeff_d[t].shape)
            # print(direction.shape)
            # print(self.nl[t].shape)
            # print(self.noise_vectors[t].shape)
            # print(self.sqrt_now_coeff[t].shape)
            # print(grad.shape)
          

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True
        
        return tweedie_decode, grad  #, now_sample, self._samples[1], direction, self.noise_vectors[t], grad
    
    def initialize(self, noise_vectors, batch_size_doub):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]
        
        self._samples[0] = self._samples[0].unsqueeze(0).expand(batch_size_doub//2,-1,-1,-1)
        self.tweedies = []

def sequential_sampling(pipeline, sampler, prompt_embeds, noise_vectors, loss_fn=None, condition=None): 
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    sampler.initialize(noise_vectors, prompt_embeds.shape[0])

    model_time = 0
    step = 0
    coss = []

    # tweedie_decode_list = []
    # now_sample_list = []
    # prev_sample_list = []
    # direction_list = []
    # noise_vector_list = []
    grad_list = []
    while not sampler.is_finished():
        step += 1
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        #model_output = pipeline.unet(**model_kwargs)
        ##my impl to batch
        sample = model_kwargs["sample"]
        sample = torch.cat(sample)
        timestep = model_kwargs["timestep"]
        timestep = timestep.expand(model_kwargs["encoder_hidden_states"].shape[0])

        
        tweedie_decode, grad = sampler.step(sample, pipeline=pipeline, timestep=timestep, temb=model_kwargs["encoder_hidden_states"],loss_fn=loss_fn,condition=condition)
        # tweedie_decode, now_sample, prev_sample, direction, noise_vector, grad = sampler.step(sample, pipeline=pipeline, timestep=timestep, temb=model_kwargs["encoder_hidden_states"],loss_fn=loss_fn,condition=condition)
        if step % 10 == 0:
            # tweedie_decode_list.append(tweedie_decode)
            # now_sample_list.append(now_sample)
            # prev_sample_list.append(prev_sample)
            # direction_list.append(direction)
            # noise_vector_list.append(noise_vector.unsqueeze(0))
            grad_list.append(grad)


        samp = sampler.get_last_sample().detach()
        samp = sample = decode_latent(pipeline.vae, samp)
        img = to_img(samp)
        if step>1:
            coss.append(cos(sampler.grads[step-2],sampler.grads[step-1]))
        # tweedie = sampler.tweedies[0]
        # tweedie_decode = tweedie_decode[0].unsqueeze(0)
        # tweedie_decode = to_img(tweedie_decode)
        # tweedie_decode = Image.fromarray(tweedie_decode[0].astype(np.uint8))
        # tweedie_decode.save("tweedie.png")
    coss = torch.tensor(coss)
    print(coss.mean(),coss.std())

    # tweedie_decode_list = torch.cat(tweedie_decode_list, dim=0)
    # now_sample_list= torch.cat(now_sample_list, dim=0)
    # prev_sample_list= torch.cat(prev_sample_list, dim=0)
    # direction_list= torch.cat(direction_list, dim=0)
    # noise_vector_list= torch.cat(noise_vector_list, dim=0)
    grad_list= torch.cat(grad_list, dim=0)
    return sampler.get_last_sample(), grad_list  #, now_sample_list, prev_sample_list, direction_list, noise_vector_list, grad_list


def decode_latent(decoder, latent):
    img = decoder.decode(latent / 0.18215).sample
    return img

def to_img(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img



def main(timesteps, opt_timesteps, cfg_scale, model_id, output_dir):
    # model_ids = ["/raid/workspace/cvml_user/kjy/ImageReward/checkpoint/refl_pick_b32_1/"]
    # model_names = ["refl_pick_b32_1"]  
    # model_ids = ["/raid/workspace/cvml_user/kjy/ImageReward/checkpoint/refl_tmask_b32/"]
    # model_names = ["refl_tmask_b32"]  
    # model_ids = ["/raid/workspace/cvml_user/kjy/ImageReward/checkpoint/refl_pick_new/"]
    # model_names = ["refl_pick_new"]  


    losses = []
    # model_id = "../../ImageReward/checkpoint/refl_pick_new"
    #model_id = "../tmp-sd15-v2"
    # model_id = "CompVis/stable-diffusion-v1-4"
    #model_id = "runwayml/stable-diffusion-v1-5"
    #scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet"
    )
    

    pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
    #pipe = StableDiffusionXLPipeline.from_pretrained(
    #    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    #)

    #pipe.unet.load_attn_procs(model_path+"checkpoint_199/",weight_name="pytorch_lora_weights.bin")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # set the number of steps
    pipe.scheduler.set_timesteps(50)
    pipe = pipe.to(device)
    pipe.vae = pipe.vae.to(device)

    pipe.unet = pipe.unet.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    # pipe.enable_attention_slicing()

    #f = open("prompts_save.txt",'r')
    #texts = f.readlines()

    df = pd.read_json("/raid/workspace/cvml_user/kjy/ImageReward/data/test.json")
    texts = df['prompt'][:50]
    # texts = df['text'][:200]
    print(texts)
    
    n_per_device = len(texts) // len(devices)
    print(n_per_device)
    if DNUM < len(devices) - 1:
        texts = texts[DNUM * n_per_device : (DNUM+1) * n_per_device]
    else:
        texts = texts[DNUM * n_per_device :]
    texts = texts.reset_index(drop=True)
    # df = pd.read_csv("../../cocoval/subset.csv")
    # texts = df['caption']
    # texts = texts[:300]
    #df = pd.read_json("../photo.json")
    #texts = df[0]
    cnt = DNUM * n_per_device
    
    output_dir += f"/timesteps{timesteps}_opt_timesteps{opt_timesteps}"
    os.makedirs(output_dir, exist_ok=True)
    text_dataset = TextDataset(texts)
    
    dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)
    loss_fn = RFUNCTIONS['pick'](inference_dtype = torch.float, device = device)

    start_time = time.perf_counter()
    for prompts in dataloader:
        #prompt, path = txt.split('\t ,')
        #prompt = txt
        noise_vectors = torch.randn(50 + 1, 4, 64, 64, device = device)
        # noise_vectors.requires_grad_(True)

        encoder_hidden_states= pipe._encode_prompt(
            prompts,
            device,
            1,
            True,
        )
        # print(prompts)
        ddim_sampler = SequentialDDIM(timesteps = timesteps,
                                scheduler = pipe.scheduler, 
                                eta = 1, 
                                cfg_scale = cfg_scale, 
                                device = device,
                                opt_timesteps = opt_timesteps)
        
        # sample, now_sample_list, prev_sample_list, direction_list, noise_vector_list, grad_list = sequential_sampling(pipe, ddim_sampler, prompt_embeds = encoder_hidden_states, noise_vectors = noise_vectors, loss_fn = loss_fn, condition=prompts)        
        sample, grad_list = sequential_sampling(pipe, ddim_sampler, prompt_embeds = encoder_hidden_states, noise_vectors = noise_vectors, loss_fn = loss_fn, condition=prompts)
        # save_list = [now_sample_list, prev_sample_list, direction_list, noise_vector_list, grad_list]
        # save_list_name = ['now_sample', 'prev_sample', 'direction', 'noise_vector', 'grad']
        save_list = [grad_list]
        save_list_name = ['grad']
  
        sample = decode_latent(pipe.vae, sample)
        loss = loss_fn(sample, prompts)
        losses.append(loss.detach().cpu().numpy())
        print(loss.detach().cpu().numpy())
        # print('img')
        # print(torch.min(sample), torch.max(sample))
        images = to_img(sample)
        os.makedirs(output_dir + "/img", exist_ok=True)

        for image in images:
            # print(image.shape)
            image = Image.fromarray(image.astype(np.uint8))
            image.save(output_dir + "/img/" +str(cnt)+".png")

            
            for s in range(len(save_list)):
                os.makedirs(f'{output_dir}/{save_list_name[s]}', exist_ok=True)
                # For others
                # save_item_mean = save_list[s].mean(dim=1)

                # For tweedie
                save_item_mean = save_list[s]
                for step in range(5):
                    # for channel in range(4):
                    # For others
                    # save_item_img = to_img(save_item_mean[step][channel].unsqueeze(0).unsqueeze(0))
                    # save_item_img = save_item_mean[step][channel]
                    # save_item_img = (save_item_img - torch.min(save_item_img)) / (torch.max(save_item_img)-torch.min(save_item_img))
                    # save_item_img = Image.fromarray(torch.clamp(255 * save_item_img.cpu().float(), 0, 255).to(dtype=torch.uint8).numpy())
                    
                    torch.save(save_item_mean[step], f'{output_dir}/{save_list_name[s]}/img{cnt}_step{(4-step)*10}.pt')
                    
                    # For tweedie
            
                    # save_item_img = to_img(save_item_mean[0])
                    # save_item_img = to_img(save_item_mean[0].unsqueeze(0))
                    # save_item_img = Image.fromarray(save_item_img[0].astype(np.uint8))

                    # save_item_img.save(f'{output_dir}/{save_list_name[s]}/img{cnt}_step47.png')
                    # save_item_img.save(f'{output_dir}/{save_list_name[s]}/img{cnt}_step{(4-step)*10}_channel{channel}.png')

            cnt+=1
  
    end_time = time.perf_counter()
    print("time consumption : ", end_time - start_time)
    
    losses = np.array(losses)
    print(losses.mean())
    print(losses.std())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--opt_timesteps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="SD14-full")

    args = parser.parse_args()

    main(args.timesteps, args.opt_timesteps, args.cfg_scale, args.model_id, args.output_dir)

