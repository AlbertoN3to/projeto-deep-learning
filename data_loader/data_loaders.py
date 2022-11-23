from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
import os

class AlzheimerDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

class AlzheimerDataLoader:
    """
    Alzheimer data loading using BaseDataLoader
    """
    def __init__(self, data_dir, test_dir, batch_size, shuffle=True, validation_split=0.2,num_workers=2):
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1,0.1,0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        self.data_dir = data_dir
        
        dataset = datasets.ImageFolder(self.data_dir)
        train_count = int((1.0 - validation_split) * len(dataset))
        valid_count = len(dataset) - train_count
        
        test = datasets.ImageFolder(test_dir)
        
        train,val = random_split(dataset, (train_count,valid_count))

        train_dataset = AlzheimerDataset(train,data_transforms["train"])
        val_dataset = AlzheimerDataset(val,data_transforms["val"])
        test_dataset = AlzheimerDataset(test,data_transforms["val"])

        
        train_loader = DataLoader(train_dataset , batch_size=batch_size, 
                                                num_workers=num_workers,  shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                                num_workers=num_workers )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                                num_workers=num_workers)
        print("Data Loaded!!")
        
        self.dataset_sizes = {"train":train_count, "val":valid_count, "test":len(test_dataset)}
        self.data_loaders = {"train":train_loader, "val":val_loader,"test":test_loader}
        
        self.classes = os.listdir(self.data_dir)
