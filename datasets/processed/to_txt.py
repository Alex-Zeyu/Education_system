import os
import numpy as np
import torch

def save_txt(dataset: str, idx: int, mode: str) -> None:
    """export the edge index for SBGNN usage
    """
    data_path = os.path.join(dataset, f'{mode}_{idx}.pt')

    if mode == 'train':
        mode = 'training'
    elif mode == 'val':
        mode = 'validation'
    elif mode == 'test':
        mode = 'testing'
    save_path = f'{dataset}-{idx}_{mode}.txt'

    # read dataset
    # dat = torch.load(data_path).t().numpy()
    dat = np.insert(torch.load(data_path).t().numpy(), 0, np.array([-1, -1, -1]), axis=0)
    dat[1:,1] -= min(dat[1:,1])
    # print(np.min(dat[1:,], axis=0), np.max(dat[1:,], axis=0))
    np.savetxt(save_path, dat, fmt='%d', delimiter='\t')

for dataset in ['Biology', 'Law', 'Psychology', 'Sydney', 'Cardiff0', 'Cardiff1']:
    for i in range(10):
        save_txt(dataset, i, 'train')
        save_txt(dataset, i, 'val')
        save_txt(dataset, i, 'test')
    print(f'{dataset} done.')

# save_txt('Biology', 0, 'train')
