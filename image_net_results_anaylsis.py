from modules.models import *
from modules.analysis import *

import numpy as np
import argparse, pickle, torch, os


def main():

    parser = argparse.ArgumentParser(description="variable file select")
    parser.add_argument("--csv", type=str, help="variable csv file")
    args = parser.parse_args()
    variableFile = args.csv
    variableFileName = variableFile.split('.')[0]

    if "_analysis_recommended" in variableFileName:
        variableFileName = variableFileName.split("_analysis_recommended")[0]

    inputValue = {}
    for key, value in np.loadtxt(variableFile, dtype=str, delimiter=','):
        inputValue[key] = value

    modelPath = str(inputValue["modelPath"])
    dataInfosPath = str(inputValue["dataInfosPath"])
    classNamePath = str(inputValue["classNamePath"])
    imageSize = int(inputValue["imageSize"])
    resultsSubjects = str(inputValue["resultsSubjects"]).split('/')

    dataInfosDic = pickle.load(open(dataInfosPath, "rb"))
    classDic = pickle.load(open("dump_dir/classDic.dump", "rb"))
    className = {}
    for stringLine in np.loadtxt(classNamePath, dtype=str, delimiter='\n'):
        key, value = stringLine.split(' ', 1)
        className[classDic[key]] = value

    if not os.path.isdir("analysis/"):
        os.mkdir("analysis/")

    pathAnalysisDir = "analysis/"+variableFileName+'/'
    if not os.path.isdir(pathAnalysisDir):
        os.mkdir(pathAnalysisDir)

    deviceName = "cpu"
    device = torch.device(deviceName)
    model = ResUNext(export_FCL=True)
    model.load_state_dict(torch.load(modelPath, map_location=device))

    ExportNumericalResults(pathAnalysisDir, dataInfosDic, resultsSubjects)

    ExportClassActivationMapResults(pathAnalysisDir, model, dataInfosDic, device, className, topRank=True)

if __name__ == "__main__":
    main()