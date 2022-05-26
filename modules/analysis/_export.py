from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, roc_auc_score, recall_score, roc_curve, f1_score, auc
from matplotlib.pyplot import suptitle, savefig, subplot, figure, imshow, axis, clf
from torchvision.transforms import Normalize, ToTensor, Compose, Resize
from numpy.random import choice
from numpy import savetxt, argsort, arange, array, uint8, max, min
from PIL.Image import fromarray
from cv2 import COLOR_BGR2RGB, COLORMAP_JET, applyColorMap, cvtColor, imread, resize
from os.path import isdir
from os import mkdir


def ExportNumericalResults(savePath, resultsDic, resultsSubjects):
    
    dataPathList = resultsDic["dataPathList"]
    originalClasses = resultsDic["originalClasses"]
    predictionClasses = resultsDic["predictionClasses"]
    probability = resultsDic["probability"]

    resultsInfos = []
    if "Accuracy" in resultsSubjects:
        resultsInfos.append(str("Accuracy : %f\n"%accuracy_score(originalClasses, predictionClasses)))
    if "F1 score" in resultsSubjects:
        resultsInfos.append(str("F1 score : %f\n"%f1_score(originalClasses, predictionClasses, average='macro')))
    if "Precision score" in resultsSubjects:
        resultsInfos.append(str("Precision score : %f\n"%precision_score(originalClasses, predictionClasses, average='macro')))
    if "Sensitivity score" in resultsSubjects:
        resultsInfos.append(str("Sensitivity score : %f\n"%recall_score(originalClasses, predictionClasses, average='macro')))
    if "Classification report" in resultsSubjects:
        resultsInfos.append(str("Classification report : \n" + classification_report(originalClasses, predictionClasses)))

    savetxt(savePath+"numerical_results.txt", array(resultsInfos), fmt="%s", delimiter="\n")

    return None

def ExportClassActivationMapResults(savePath, model, resultsDic, device, className, imageSize=256, topRank=False, exportNum=20):
    
    model.eval()

    dataPathList = array(resultsDic["dataPathList"])
    originalClasses = array(resultsDic["originalClasses"])
    predictionClasses = array(resultsDic["predictionClasses"])
    probability = array(resultsDic["probability"])

    parameters = model.state_dict()
    weightFullyConnected = parameters["fully_connected.weight"].numpy()

    normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )

    dataTransform = Compose([
        Resize((imageSize, imageSize)),
        ToTensor(),
        normalize,
        ])

    if topRank:
        idxsCorrect = originalClasses == predictionClasses
        dataPathList = dataPathList[idxsCorrect]
        originalClasses = originalClasses[idxsCorrect]
        predictionClasses = predictionClasses[idxsCorrect]
        probability = probability[idxsCorrect]
        ranks = argsort(probability[:,1])
        idxList = arange(len(probability))[ranks <= exportNum-1]
    else:
        idxList = choice(arange(len(probability)), exportNum)
    
    i = 0
    
    pathDirCAM = savePath+"CAM/"
    if not isdir(pathDirCAM):
        mkdir(pathDirCAM)

    for idx in idxList:
        i += 1
        img = imread(dataPathList[idx], 1)
        imgSizeW = img.shape[1]
        imgSizeH = img.shape[0]

        FCLs, labels = model(dataTransform(fromarray(img)).unsqueeze(dim=0).to(device))
        FCL = FCLs[0].detach().numpy()
        _sizeC = FCL.shape[0]
        _sizeW = FCL.shape[1]
        _sizeH = FCL.shape[2]
        CAM = weightFullyConnected[originalClasses[idx]].dot(FCL.reshape(_sizeC, _sizeW*_sizeH)).reshape(_sizeW, _sizeH)

        CAM -= min(CAM)
        CAM /= max(CAM)
        CAM *= 255
        CAM = resize(CAM, dsize=(imgSizeW, imgSizeH))
        CAM = applyColorMap(CAM.astype(uint8), COLORMAP_JET)

        figure(figsize=(15, 4 * (img.shape[0] / img.shape[1] + 0.3)))
        suptitle("original %.4f%% : %s\n predict %.4f%% : %s"
                     %(probability[idx][0], className[originalClasses[idx]], probability[idx][1], className[predictionClasses[idx]]))
        subplot(1,3,1)
        imshow(cvtColor(img, COLOR_BGR2RGB))
        axis("off")
        subplot(1,3,2)
        imshow((cvtColor(img, COLOR_BGR2RGB)*0.7+CAM*0.3).astype(uint8))
        axis("off")
        subplot(1,3,3)
        imshow(CAM)
        axis("off")
        savefig(pathDirCAM+"/%3d.png"%i)
        clf()

    return None