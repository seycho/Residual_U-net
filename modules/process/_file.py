from os.path import isfile, isdir
from os import mkdir
from pickle import dump, load
from numpy import savetxt, array


def VariableDumpSaveNLoad(key, func, *args):

    pathDump = "dump_dir/"
    if not isdir(pathDump):
        mkdir(pathDump)

    fileName = pathDump+key+".dump"
    if isfile(fileName):
        value = load(open(fileName, "rb"))
    else:
        value = func(*args)
        dump(value, open(fileName, "wb"))
    key = value

    return key

def RecordCSV(variableFileName, processRecord, imageSize, classNamePath):

    recordLoss = 1e3
    recordAccuracy = 1e3
    recordEpoch = 1000

    for epoch in processRecord["loss"].keys():
        if recordLoss > processRecord["loss"][epoch]["valid"]:
            recordEpoch = epoch
            recordLoss = processRecord["loss"][epoch]["valid"]
            recordAccuracy = processRecord["accuracy"][epoch]["valid"]

    modelPath = "results/%s/loss_%.4f_acc_%.2f%%_epoch_%d.pt"%(variableFileName, recordLoss, recordAccuracy, recordEpoch)
    dataInfosPath = "results/%s/valid_loss_%.4f_acc_%.2f%%_epoch_%d.dump"%(variableFileName, recordLoss, recordAccuracy, recordEpoch)
    resultsSubjects = "Accuracy/F1 score/Precision score/Sensitivity score/Classification report"

    saveCSV = array([["modelPath", modelPath],
                     ["dataInfosPath", dataInfosPath],
                     ["classNamePath", classNamePath],
                     ["imageSize", imageSize],
                     ["resultsSubjects", resultsSubjects]])
    
    savetxt("%s_analysis_recommended.csv"%(variableFileName), saveCSV, delimiter=',', fmt="%s")

    return None
