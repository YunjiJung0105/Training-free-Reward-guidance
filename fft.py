import torch.fft as fft
import torch 
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob 
import natsort

def high_low_fft(x, cutoff_ratio):
    # FFT   
    orig_dtype = x.dtype
    with torch.autocast(device_type='cuda', dtype=torch.float):
        x = x.to(torch.float)
        _, _, h, w = x.shape
        cx, cy = w // 2, h // 2  # Center coordinates
        x_freq = fft.fftn(x)
        x_freq = fft.fftshift(x_freq)
        

        # # # Compute amplitude (magnitude of the complex numbers)
        # amplitude = torch.abs(x_freq[0])
        # # print(amplitude.shape)
        
        # # Create frequency bins based on the distance from the center
        # frequency = np.zeros((h, w))
        # for y in range(h):
        #     for x in range(w):
        #         frequency[y, x] = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # # Flatten the arrays to compute radial means
        # amplitude_flat = amplitude.flatten().cpu().numpy()
        # frequency_flat = frequency.flatten()

        # # Sort by frequency
        # sorted_indices = np.argsort(frequency_flat)
        # frequency_sorted = frequency_flat[sorted_indices]
        # amplitude_sorted = amplitude_flat[sorted_indices]
        
        # # Group into bins and compute the log amplitude
        # bin_size = 1
        # max_frequency = int(np.max(frequency_sorted))
        # bin_centers = []
        # log_amplitudes = []
        
        # for i in range(0, max_frequency, bin_size):
        #     # Find indices of frequencies within this bin
        #     indices_in_bin = np.where((frequency_sorted >= i) & (frequency_sorted < i + bin_size))
            
        #     if len(indices_in_bin[0]) > 0:
        #         bin_center = i + bin_size / 2
        #         avg_amplitude = np.mean(amplitude_sorted[indices_in_bin])
                
        #         bin_centers.append(bin_center)
        #         log_amplitudes.append(np.log1p(avg_amplitude))  # Log scale, log(1 + amplitude)
        
        # # Plot frequency vs log amplitude
        # plt.figure(figsize=(8, 6))
        # plt.plot(bin_centers, log_amplitudes)
        # x_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        # x_tick_labels = ['0', r'$\pi /2$', r'$\pi$',r'$3\pi/2$']
        # plt.xticks(x_ticks, x_tick_labels)
        # plt.title('Frequency vs Log Amplitude')
        # plt.xlabel('Frequency (radial distance)')
        # plt.ylabel('Log Amplitude')
        # plt.grid(True)
        # plt.savefig("aaasdf.png")
        # raise
        
        
        low_mask = torch.zeros_like(x_freq) 
        high_mask = torch.ones_like(x_freq)
        radius = min(h, w) * cutoff_ratio
        cnt = 0
        for y in range(h):
            for x in range(w):
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    low_mask[:, :, y, x] = 1
                    high_mask[:, :, y, x] = 0
                    cnt += 1
        
        
        # print(cnt,h*w)
        low_freq = x_freq * low_mask
        high_freq = x_freq * high_mask
        high_freq = fft.ifftshift(high_freq)
        high_filtered = fft.ifftn(high_freq).real

        low_freq = fft.ifftshift(low_freq)
        low_filtered = fft.ifftn(low_freq).real
    high_filtered = high_filtered.to(orig_dtype)
    low_filtered = low_filtered.to(orig_dtype)
    return high_filtered, low_filtered



def draw_grid(img_path, img, ffts, ratio): 
    # check if tens is numpy 
    if isinstance(ffts, torch.Tensor): 
        ffts = ffts.cpu().detach().numpy()
    fig, ax = plt.subplots(2, 4, figsize=(13, 5))

    ax[0][0].imshow(img.permute(1, 2, 0))
    ax[0][0].set_title("img")

    for i in range(2):
        if i == 0:
            title = 'high_filter'
        else:
            title = 'low filter'
        for j in range(3):
            ax[i][j+1].imshow(ffts[i, j, :, :], cmap='gray')
            ax[i][j+1].set_title(f"{title}[{i}][{j}]")

    plt.tight_layout()

    if 'tweedie' in img_path:
        plt.savefig(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}/img{img_path.split('/')[-1].split('_')[0][3:]}_tweedie_{img_path.split('/')[-1].split('_')[1].split('.')[0]}.png")
    elif 'direction' in img_path:
        plt.savefig(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}/img{img_path.split('/')[-1].split('_')[0][3:]}_direction_{img_path.split('/')[-1].split('_')[1].split('.')[0]}.png")
    else:
        plt.savefig(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}/img{img_path.split('/')[-1].split('.')[0]}_clean.png")



ratio = 0.05

# img_paths = glob.glob("SD14-nograd-2/direction/*.pt")
# img_paths = natsort.natsorted(img_paths)

# for img_path in img_paths:
#     print(img_path)
#     os.makedirs(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}", exist_ok=True)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     img = torch.load(img_path, map_location="cpu").to(device)
#     img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).to(dtype=torch.uint8)

#     ret1, ret2 = high_low_fft(img.unsqueeze(0), ratio)
#     ret = torch.concat([ret1, ret2], dim=0)
#     draw_grid(img_path, img, ret, ratio)


img_paths = glob.glob("SD14-nograd-2/timesteps50_opt_timesteps50/tweedie_decode/*.png")
img_paths = natsort.natsorted(img_paths)

for img_path in img_paths:
    print(img_path)
    os.makedirs(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}", exist_ok=True)

    to_tensor = transforms.ToTensor()
    img = Image.open(img_path)
    img = to_tensor(img)

    ret1, ret2 = high_low_fft(img.unsqueeze(0), ratio)
    ret = torch.concat([ret1, ret2], dim=0)
    draw_grid(img_path, img, ret, ratio)



# img_paths = glob.glob("SD14-nograd-2/img/*.png")
# img_paths = natsort.natsorted(img_paths)

# for img_path in img_paths:
#     print(img_path)
#     os.makedirs(f"{img_path.split('/')[0]}/fft/ratio_0_{str(ratio).split('.')[-1]}", exist_ok=True)

#     to_tensor = transforms.ToTensor()
#     img = Image.open(img_path)
#     img = to_tensor(img)

#     ret1, ret2 = high_low_fft(img.unsqueeze(0), ratio)
#     ret = torch.concat([ret1, ret2], dim=0)
#     draw_grid(img_path, img, ret, ratio)