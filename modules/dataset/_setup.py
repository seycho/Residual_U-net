from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
from numpy.random import permutation
from numpy import zeros
from cv2 import imread, resize
from PIL.Image import fromarray


def MakeEqualDatasetList(_dic, dataShuffle=False):
    equalList = []
    
    lenList = []
    for key in _dic.keys():
        lenList.append(len(_dic[key]))
    dataMin = min(lenList)
    
    for key in _dic.keys():
        equalList.extend(permutation(_dic[key])[:dataMin].tolist())
    
    if dataShuffle:
        equalList = permutation(equalList).tolist()
    
    return equalList

class ImageNetDataset(Dataset):
    
    def __init__(self, dataInfosDictionaries, transform=None):
        self.dataInfosDictionaries = dataInfosDictionaries
        self.transform = transform
        
    def __getitem__(self, index):
        self.dataInfosDic = self.dataInfosDictionaries[index]
        self.dataPath = self.dataInfosDic["dataPath"]
        self.dataLabelNum, self.dataLabelMax = self.dataInfosDic["dataLabelInfo"]
        
        self.data = imread(self.dataPath, 1)
        self.data = self.transform(fromarray(self.data))
        self.dataLabel = zeros(self.dataLabelMax)
        self.dataLabel[self.dataLabelNum] = 1
        
        return self.dataPath, self.data, self.dataLabel
    
    def __len__(self):
        return len(self.dataInfosDictionaries)

def MakeDataLoader(dataList, datasetClass=ImageNetDataset, batchSize=8, numWorkers=4, dataShuffle=True, dataTransform=None):
    
    if dataTransform is None:
        
        normalize = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                )

        dataTransform = Compose([
            Resize((256, 256)),
            ToTensor(),
            normalize,
            ])
        
    _datasetClass = datasetClass(dataList, dataTransform)

    dataLoader = DataLoader(_datasetClass, batch_size=batchSize, shuffle=dataShuffle, drop_last=False, num_workers=numWorkers)
    
    return dataLoader