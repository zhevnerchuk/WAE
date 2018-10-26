from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor

import os
import PIL
from skimage import io


class ImagesUpDownWrapper(Dataset):
    def __init__(self, dataset, top_pixels_to_cut):
        self.dataset = dataset
        self.top_pixels_to_cut = top_pixels_to_cut


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        pic, _ = self.dataset.__getitem__(idx)
        X = pic[:, :self.top_pixels_to_cut]
        Y = pic[:, self.top_pixels_to_cut:]
        return X, Y

class CelebaDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.mode = mode        
        
        self.root_dir = path
        self.img_dir = os.path.join(self.root_dir, self.mode)
        self.img_ids = os.listdir(self.img_dir)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = PIL.Image.fromarray(io.imread(os.path.join(self.img_dir, img_id)))  
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, img

def get_train_valid_loader(batch_size=100, random_seed=42, shuffle=False):
    transforms = Compose([CenterCrop((140, 140)),
                          Resize((64, 64)),
                          ToTensor()])
    
    train_dataset = CelebaDataset(mode="train", transforms=transforms)
    valid_dataset = CelebaDataset(mode="valid", transforms=transforms)
    
    train_sampler = None
    if shuffle:
        indices = np.arange(len(train_dataset))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,  sampler=train_sampler)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader