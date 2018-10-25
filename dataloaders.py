from torch.utils.data import Dataset


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
