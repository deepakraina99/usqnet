import torch
from torch.utils.data import Dataset
import glob, re, os, cv2, collections

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_data(img_dir, cls_labels):
    all_images, all_labels = [], []
    print('loading data ...')
    for lb in cls_labels:
        all_imgs = sorted(glob.glob(img_dir+'/'+lb+'/'+'*.png'), key=numericalSort)
        # print('class:{}, #images:{}'.format(lb, len(all_imgs)))
        for img in all_imgs:
            all_images.append(cv2.imread(img))
            all_labels.append(int(lb)-1)
    print('total images: {}'.format(len(all_images)))
    counter=collections.Counter(all_labels)
    print('class images:', collections.OrderedDict(sorted(counter.items())))
    return all_images, all_labels

def collate_fn(batch):
    imgs_batch, labels_batch = [], []
    imgs_batch = [bitem[0] for bitem in batch]
    labels_batch = [bitem[1] for bitem in batch]
    input_tensor = torch.stack(imgs_batch)
    ouput_tensor = torch.stack(labels_batch)
    return input_tensor, ouput_tensor

class USQDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.as_tensor(self.labels[idx], dtype=torch.int64)
        return image, label
    
    def __len__(self):
        return len(self.images)


