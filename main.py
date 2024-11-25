from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
import torch
import torchvision
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
from diffusers.image_processor import VaeImageProcessor
import pickle
import argparse

DNUM = 0
devices = ['cuda:6']
device = torch.device(devices[DNUM] if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


global_norms = dict()
global_loss = dict()
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
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]
        
        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        self.m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(self.m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x
        self.sqrt_m_now_coeff = torch.sqrt(m_now_coeff)
        self.sqrt_now_coeff = torch.sqrt(now_coeff)
        
        self.sqrt_m_next_coeff = torch.sqrt(m_next_coeff)
        self.sqrt_next_coeff = torch.sqrt(next_coeff)
        # self.betas = [1 - self.scheduler.alphas[t] for t in self.scheduler_timesteps]
        # self.betas = [1 - self.scheduler.alphas[50-t] for t in self.scheduler_timesteps]
        # self.betas = [torch.tensor(0.001,device=self.scheduler.alphas[t].device) for t in self.scheduler_timesteps]
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

    def inner_optim_dno(self, grad):
        torch.set_grad_enabled(True)
        grad = grad.detach()
        grad.requires_grad_(True)
        
        inner_steps = 20
        lr = 0.1
        for inner_step in range(inner_steps):
            # grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            # grad = grad / grad_norm * ref_norm # noise norm

            loss = [compute_probability_regularization(grad[i], 1) for i in range(grad.shape[0])]
            
            loss = sum(loss) / len(loss)
            # print(f"step : {t}, inner opt : {inner_step}, reg loss : {loss.item()}")
            reg_dir = torch.autograd.grad(loss,grad)[0]
            grad = grad - lr * reg_dir

        torch.set_grad_enabled(False)
        return grad
    
    def inner_optim_cosguide(self, opt_dir, guidance):
        torch.set_grad_enabled(True)
        opt_dir = opt_dir.detach()
        opt_dir.requires_grad_(True)
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # coss = cos(tmp.flatten(),self.noise_vectors[t].flatten())

        inner_steps = 30
        lr = 0.5
       
        for inner_step in range(inner_steps):

            # loss = [compute_probability_regularization(opt_dir[i], 1) for i in range(opt_dir.shape[0])]
            # print(opt_dir.shape,guidance.shape)
            loss = -cos(opt_dir[0].flatten(), -guidance[0].flatten())
            
            
            # loss = sum(loss) / len(loss)
            # print(loss,"hm")
            # if inner_step < 1:
                # print(f"inner opt : {inner_step}, reg loss : {loss.item()}")
            reg_dir = torch.autograd.grad(loss,opt_dir)[0]
            opt_dir = opt_dir - lr * reg_dir
        # print(f"inner opt : {inner_step}, reg loss : {loss.item()}")
        torch.set_grad_enabled(False)
        return opt_dir
    def step(self, sample, pipeline,timestep,temb, loss_fn=None,condition=None, use_forward=True, use_tilde=True, gamma_bar=0.1, use_backward=True, use_norm_adj=True):
        
        sample = sample.detach()
        now_sample = None
        max_iter = 1
        for i in range(max_iter):
            tweedie = None
            updated_tweedie = None
            tweedie_decode = None
            torch.set_grad_enabled(True)
            sample.requires_grad_(True)
            if sample.shape[0] == 1:
                sample = torch.cat(self.scheduler.scale_model_input([sample,sample],timestep))
            model_output = pipeline.unet(sample, timestep, temb)   
            model_output_uncond, model_output_text = model_output[0].chunk(2)
            direction = model_output_text + self.cfg_scale * (model_output_text - model_output_uncond)

            t = self.num_steps - len(self._samples)
            grad = 0
            tweedie_decode = None
            if loss_fn != None:
                s = sample[:sample.shape[0]//2]
                tweedie = self.get_tweedie(s,direction,t)   
                tweedie_decode = decode_latent(pipeline.vae,tweedie)
                ### Forward Guidance
                if use_forward:
                    if use_tilde:
                        delta = torch.randn_like(tweedie_decode)
                        loss = loss_fn(tweedie_decode + gamma_bar * torch.sqrt(self.sqrt_now_coeff[t]) * delta, condition)
                    else:
                        loss = loss_fn(tweedie_decode, condition)
                        grad = torch.autograd.grad(loss,s)[0]

            ### Backward Guidance (implemented in latent version (TFG style))
            # if use_backward:
            #     back_iters = 0
            #     opt_tweedie = tweedie.detach().requires_grad_(True)
            #     lr = 1e-2
            #     optimizer = torch.optim.SGD([opt_tweedie], lr=lr)
            #     weights = torch.ones_like(opt_tweedie).to(self.device)
            #     ones = torch.ones_like(opt_tweedie).to(self.device)
            #     zeros = torch.zeros_like(opt_tweedie).to(self.device)
                
            #     for _ in range(back_iters):
            #         optimizer.zero_grad()
            #         loss = loss_fn(decode_latent(opt_tweedie), condition)   
            #         m_loss.backward()
            #         optimizer.step()
            #     opt_tweedie.requires_grad = False
            torch.set_grad_enabled(False)
            

            ### Visualize

            # tweedie_decode = torch.clamp(tweedie_decode, -1, 1)
            # image = to_img(tweedie_decode)
            # image = Image.fromarray(image[0].astype(np.uint8))
            # updated_tweedie = encode_img(pipeline.vae,tweedie_decode)
            # updated_tweedie_decode = decode_latent(pipeline.vae,updated_tweedie)
            
            # image = to_img(tweedie_decode)
            # image = Image.fromarray(image[0].astype(np.uint8))
            # image.save("tweedie_uped.png")
            # self.tweedies.insert(0,tweedie.detach())

            ### Norm adjust (by kjy)
            if use_norm_adj:
                ref_norm = torch.norm(self.noise_vectors[t,0]) 
                grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                grad = grad / grad_norm * ref_norm # noise norm
            
            ### Choose sampling method
            
            ### Update tweedie (for Backward guidance)
            # direction = direction - self.sqrt_m_now_coeff[t] / self.sqrt_now_coeff[t] * (opt_tweedie-tweedie)
            #now_sample = opt_tweedie * self.sqrt_m_next_coeff[t] + direction * torch.sqrt(self.m_nl_next_coeff[t]) + self.nl[t] *  torch.randn((1,4,64,64),device=sample.device) - 0.3 * self.sqrt_now_coeff[t] * grad
            
            ### Only Forward guidance
            if use_forward:
                now_sample = tweedie * self.sqrt_m_next_coeff[t] + direction * torch.sqrt(self.m_nl_next_coeff[t]) + \
                    self.nl[t] *  torch.randn((1,4,64,64),device=sample.device) - 3 * self.sqrt_now_coeff[t] * grad
            else:
                now_sample = tweedie * self.sqrt_m_next_coeff[t] + direction * torch.sqrt(self.m_nl_next_coeff[t]) + \
                    self.nl[t] *  torch.randn((1,4,64,64),device=sample.device) 
           

            ## ULA dynamics with the gamma-product distribution
            ## To use this, ensure the above is DDIM dynamics
            # K = 0
            # gamma = 0
            # print(self.coeff_d[t],self.betas[t],torch.sqrt(self.betas[t]),(2*self.sqrt_now_coeff[t]))
            # direction = self.coeff_d[t] * direction #+ self.nl[t] * self.noise_vectors[t]
            # for k in range(K):
            #     epsilon = torch.randn(1, 4, 64, 64, device = self.noise_vectors[t].device)
            #     now_sample  = now_sample - (self.betas[t] / 2) * ( (1-gamma) * direction + gamma * grad ) + torch.sqrt(self.betas[t]) * epsilon
            
            if i < max_iter - 1:
                
                coeff1 = self.sqrt_m_now_coeff[t] /  self.sqrt_m_next_coeff[t]
                # For check
                # torch.sqrt(self.scheduler.alphas[t])
                # print(torch.sqrt(self.scheduler.alphas_cumprod[self.scheduler_timesteps[t]] / self.scheduler.alphas_cumprod[self.scheduler_timesteps[t-1]]),coeff1)
                coeff2 = torch.sqrt(1 - coeff1 * coeff1)
                now_sample = coeff1 * now_sample + coeff2 * torch.randn((1,4,64,64),device=sample.device)
                sample = now_sample

        self._samples.insert(0, now_sample)
        if len(self._samples) > self.timesteps:
            self._is_finished = True
        
        return tweedie_decode
    
    def initialize(self, noise_vectors, batch_size_doub):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]
        
        self._samples[0] = self._samples[0]#.unsqueeze(0).expand(batch_size_doub//2,-1,-1,-1)
        self.tweedies = []

def sequential_sampling(pipeline, sampler, prompt_embeds, noise_vectors, loss_fn=None, condition=None, use_forward=True, use_tilde=True, gamma_bar=0.1, use_backward=True, use_norm_adj=True): 
    
    sampler.initialize(noise_vectors, prompt_embeds.shape[0])

    model_time = 0
    step = 0
    coss = []
    while not sampler.is_finished():
        step += 1
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        #model_output = pipeline.unet(**model_kwargs)
        ##my impl to batch
        sample = model_kwargs["sample"]
        sample = torch.cat(sample)
        timestep = model_kwargs["timestep"]
        timestep = timestep.expand(model_kwargs["encoder_hidden_states"].shape[0])
        # model_output = unet(sample, timestep, model_kwargs["encoder_hidden_states"])   
        
        
        # model_output = checkpoint.checkpoint(unet, sample, timestep, model_kwargs["encoder_hidden_states"],  use_reentrant=False)
        
        with torch.no_grad():
            tweedie_decode = sampler.step(sample, pipeline=pipeline, timestep=timestep, temb=model_kwargs["encoder_hidden_states"],loss_fn=loss_fn,condition=condition, use_forward=use_forward, use_tilde=use_tilde, gamma_bar=gamma_bar, use_backward=use_backward, use_norm_adj=use_norm_adj)

        # samp = sampler.get_last_sample().detach()
        # samp = sample = decode_latent(pipeline.vae, samp)
        # img = to_img(samp)
        # # if step > 1:
        #     # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        #     # coss = cos(sampler.grads[step-1].flatten(),sampler.grads[step-2].flatten())
        #     # coss = torch.tensor(coss)
        #     # print(coss)
        # tweedie = sampler.tweedies[0]
        # tweedie_decode = tweedie_decode[0].unsqueeze(0)
        # tweedie_decode = to_img(tweedie_decode)
        # tweedie_decode = Image.fromarray(tweedie_decode[0].astype(np.uint8))
        # tweedie_decode.save("tweedie.png")
    return sampler.get_last_sample()


def decode_latent(decoder, latent):
    img = decoder.decode(latent / 0.18215).sample
    return img
def encode_img(encoder, x):
    latent = encoder.encode(x).latent_dist.mean
    return 0.18215 * latent

def to_img(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img
def to_imgtensor(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1)
    return img
def compute_probability_regularization(noise_vectors, subsample, shuffled_times = 3):
    noise_vectors = noise_vectors.flatten()
    dim = noise_vectors.shape[0]

    # use for computing the probability regularization
    subsample_dim = round(4 ** subsample)
    subsample_num = dim // subsample_dim
    noise_vectors_seq = noise_vectors.view(subsample_num, subsample_dim)

    seq_mean = noise_vectors_seq.mean(dim = 0)
    noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
    seq_cov = noise_vectors_seq.T @ noise_vectors_seq
    seq_var = seq_cov.diag()
    
    # compute the probability of the noise
    seq_mean_M = torch.norm(seq_mean)
    seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
    
    seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
    seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
    seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
    seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
    seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
    seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
    seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)

    shuffled_mean_prob_list = []
    shuffled_cov_prob_list = [] 
    
    shuffled_mean_log_prob_list = []
    shuffled_cov_log_prob_list = [] 
    
    shuffled_mean_M_list = []
    shuffled_cov_M_list = []

    for _ in range(shuffled_times):
        noise_vectors_flat_shuffled = noise_vectors[torch.randperm(dim)]   
        noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
        
        shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
        noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
        shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
        shuffled_var = shuffled_cov.diag()
        
        # compute the probability of the noise
        shuffled_mean_M = torch.norm(shuffled_mean)
        shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
        

        shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
        shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
        shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
        shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
        
        shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
        shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
        shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
        
        
        shuffled_mean_prob_list.append(shuffled_mean_prob.item())
        shuffled_cov_prob_list.append(shuffled_cov_prob.item())
        
        shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
        shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
        
        shuffled_mean_M_list.append(shuffled_mean_M.item())
        shuffled_cov_M_list.append(shuffled_cov_M.item())
        
    reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)
    
    return reg_loss


def main(args):
    losses = []
    model_id = args.model_id

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
    texts = df['prompt'][:200]
    # texts = df['text'][:200]
    # print(texts[46])
    
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
    
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(output_path+'/0')
    text_dataset = TextDataset(texts)
    
    dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)
    loss_fn = RFUNCTIONS['pick'](inference_dtype = torch.float, device = device)

    start_time = time.perf_counter()
    for prompts in dataloader:
        #prompt, path = txt.split('\t ,')
        #prompt = txt
        # prompts[0] = ""
        # print(prompts)
        noise_vectors = torch.randn(50 + 1, len(prompts), 4, 64, 64, device = device)
        # noise_vectors.requires_grad_(True)

        encoder_hidden_states= pipe._encode_prompt(
            prompts,
            device,
            1,
            True,
        )
        # print(prompts)
        ddim_sampler = SequentialDDIM(timesteps = 50,
                                scheduler = pipe.scheduler, 
                                eta = 1, 
                                cfg_scale = args.cfg_scale, 
                                device = device,
                                opt_timesteps = args.opt_timesteps)
        # print(self.noise_vectors[-1],self.noise_vectors.shape)
        sample = sequential_sampling(pipe, ddim_sampler, prompt_embeds = encoder_hidden_states, noise_vectors = noise_vectors, loss_fn = loss_fn, condition=prompts, use_forward=args.use_forward, use_tilde=args.use_tilde, gamma_bar=args.gamma_bar, use_backward=args.use_backward, use_norm_adj=args.use_norm_adj)
        
        sample = decode_latent(pipe.vae, sample)
        loss = loss_fn(sample, prompts)
        losses.append(loss.detach().cpu().numpy())
        print(loss.detach().cpu().numpy(),"ddd")
        images = to_img(sample)
        for image in images:
            # print(image.shape)
            image = Image.fromarray(image.astype(np.uint8))
        
            image.save(output_path+'/0/'+str(cnt)+".png")
            cnt+=1
        #print(prompt,path)
        # for image in images:
            
        #     image = Image.fromarray(image.astype(np.uint8))
        #     image.save('generations_rewardguidance'+'/0/'+str(cnt)+".png")
        #     cnt += 1
    end_time = time.perf_counter()
    print("time consumption : ", end_time - start_time)
    # print(global_loss)
    # with open('norms_example.pickle','wb') as fw:
        # pickle.dump(global_norms, fw)
    # with open('loss_noguide.pickle','wb') as fw:
        # pickle.dump(global_loss, fw)

    losses = np.array(losses)
    print(losses.mean())
    print(losses.std())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--opt_timesteps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_path", type=str, default="SD14")
    parser.add_argument('--use_forward', action='store_true', default=False)
    parser.add_argument('--use_tilde', action='store_true', default=False)
    parser.add_argument("--gamma_bar", type=float, default=0.1)
    parser.add_argument('--use_backward', action='store_true', default=False)
    parser.add_argument('--use_norm_adj', action='store_true', default=False)


    args = parser.parse_args()

    main(args)