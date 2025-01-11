import medmnist
import torchvision
from torchvision import transforms

__all__   = ["DataSet"]
Data_Path = '/home/username/DataSet/'

DataSet = {
    # MedMNIST
    "PathMNIST"     : lambda size=28 : MedMNIST2D(size=size, name="PathMNIST"),
    "DermaMNIST"    : lambda size=28 : MedMNIST2D(size=size, name="DermaMNIST"),
    "OCTMNIST"      : lambda size=28 : MedMNIST2D(size=size, name="OCTMNIST"),
    "PneumoniaMNIST": lambda size=28 : MedMNIST2D(size=size, name="PneumoniaMNIST"),
    "RetinaMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="RetinaMNIST"),
    "BreastMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="BreastMNIST"),
    "BloodMNIST"    : lambda size=28 : MedMNIST2D(size=size, name="BloodMNIST"),
    "OrganAMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganAMNIST"),
    "OrganCMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganCMNIST"),
    "OrganSMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganSMNIST"),
}


def MedMNIST2D(size, name:str):
    MEAN, STD = [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]  # Use Cifar-10 statistics
    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        torchvision.transforms.AugMix(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    valid_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(), 
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    if name == "PathMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.PathMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.PathMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "DermaMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.0),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.DermaMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.DermaMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "OCTMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OCTMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.OCTMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "PneumoniaMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.PneumoniaMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.PneumoniaMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "RetinaMNIST":
        Train_Data = medmnist.RetinaMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.RetinaMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "BreastMNIST":
        Train_Data = medmnist.BreastMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.BreastMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "BloodMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.BloodMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.BloodMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "OrganAMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganAMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.OrganAMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "OrganCMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganCMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.OrganCMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    elif name == "OrganSMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganSMNIST(size=size, split='train', download=False, root=Data_Path+'MEDMNIST', transform=train_transform)
        Test_Data  = medmnist.OrganSMNIST(size=size, split='test', download=False, root=Data_Path+'MEDMNIST', transform=valid_transform)
    else:
        raise ValueError("Wrong Name of Dataset")

    num_class  = len(Train_Data.info['label'])
    return Train_Data, Test_Data, num_class

