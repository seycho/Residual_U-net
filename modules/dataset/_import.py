from os import listdir


def GetClassDictionary(path):
    classDic = {}
    classList = listdir(path)
    classList.sort()
    for _idx, className in enumerate(classList):
        classDic[className] = _idx
    return classDic

def GetDataInfoDictionary(path, classDic, classNum):
    dataDic = {}
    for LD1 in listdir(path):
        tempList = []
        for LD2 in listdir(path+'/'+LD1):
            tempList.append({"dataPath" : path+'/'+LD1+'/'+LD2,
                             "dataLabelInfo" : [classDic[LD1], classNum]})
        dataDic[LD1] = tempList
    return dataDic