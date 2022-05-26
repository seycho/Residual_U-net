from modules.models import *
from modules.dataset import *
from modules.process import *

from torchvision import transforms
import numpy as np
import argparse, pickle, torch, os

def PrintInputVariable(modelPath, learningRate, imageSize, batchSize, epochs, deviceName, rootPath, classNamePath):

    print("--- input variables ---")
    print("model path : %s"%(modelPath if os.path.isfile(modelPath) else "None"))
    print("learning rate : %f"%learningRate)
    print("image size : %d"%imageSize)
    print("batch size : %d"%batchSize)
    print("epochs : %d"%epochs)
    print("use cuda device : %s"%(deviceName if torch.cuda.is_available() else "cpu"))
    print("data root path : %s"%rootPath)
    print("class name path : %s"%classNamePath)
    print()
    
    return None

def main():

    parser = argparse.ArgumentParser(description="variable file select")
    parser.add_argument("--csv", type=str, help="variable csv file")
    args = parser.parse_args()
    variableFile = args.csv
    variableFileName = variableFile.split('.')[0]

    # get argments
    inputValue = {}
    for key, value in np.loadtxt(variableFile, dtype=str, delimiter=','):
        inputValue[key] = value

    modelPath = str(inputValue["modelPath"])
    learningRate = float(inputValue["learningRate"])
    imageSize = int(inputValue["imageSize"])
    batchSize = int(inputValue["batchSize"])
    epochs = int(inputValue["epochs"])
    deviceName = str(inputValue["deviceName"])
    rootPath = str(inputValue["rootPath"])
    classNamePath = str(inputValue["classNamePath"])

    # model setting
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
    if "cuda" in deviceName:
        os.environ['CUDA_LAUNCH_BLOCKING'] = deviceName.split(':')[-1]

    model = ResUNext()
    model.load_state_dict(torch.load(modelPath))
    model = model.to(device)

    PrintInputVariable(modelPath, learningRate, imageSize, batchSize, epochs, deviceName, rootPath, classNamePath)

    # prepare dataset
    classDic = VariableDumpSaveNLoad("classDic", GetClassDictionary, rootPath+"train")
    validDic = VariableDumpSaveNLoad("validDic", GetDataInfoDictionary, rootPath+"val", classDic, len(classDic))

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )

    dataTransform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor(),
        normalize,
        ])

    # learning model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    epoch = 0
    validList = MakeEqualDatasetList(validDic, dataShuffle=True)

    validDataLoader = MakeDataLoader(validList, batchSize=batchSize, dataTransform=dataTransform)

    _loss, _acc, _dic = DeeplearningProcessing(model, device, validDataLoader, criterion, optimizer, epoch, training=False, announceBatchStep=50)
    pickle.dump(_dic, open("%s_loss_%.4f_acc_%.2f%%_epoch_%d.dump"%("test", _loss, _acc, epoch), "wb"))

    return None

if __name__ == "__main__":
    main()