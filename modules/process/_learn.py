from modules.analysis import ConvertLabelNOutput2Softmax

from torch.autograd import Variable
from torch.cuda import current_device
from torch import no_grad, max
from matplotlib.pyplot import savefig, subplot, legend, figure, title, xlabel, ylabel, plot, clf


def DeeplearningProcessing(model, device, dataLoader, criterion, optimizer, epoch, training=False, announceBatchStep=100):

    if training:
        model.train()
        status = "Train"
    else:
        model.eval()
        status = "Valid"

    correct, current = 0, 0
    accSum, lossSum = 0, 0
    total = len(dataLoader.dataset)

    outDataPathList = []
    outOriginalClasses = []
    outPredictionClasses = []
    outProbability = []

    for _idx, (dataPathList, datas, labels) in enumerate(dataLoader):

        datas, labels = datas.to(device), labels.to(device)

        if training:
            datas, labels = Variable(datas), Variable(labels)
            optimizer.zero_grad()
        else:
            with no_grad():
                datas = datas
                labels = labels

        outputs = model(datas)

        loss = criterion(outputs, max(labels, 1)[1])
        if training:
            loss.backward()
            optimizer.step()
        lossSum += loss.cpu().item()

        labelClasses = max(labels, 1)[1].detach().cpu().numpy()
        predClasses = max(outputs, 1)[1].detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        outProbability.extend(ConvertLabelNOutput2Softmax(outputs, labelClasses, predClasses))
        outDataPathList.extend(dataPathList)
        outOriginalClasses.extend(labelClasses)
        outPredictionClasses.extend(predClasses)

        correct += (predClasses == labelClasses).sum()
        current += dataLoader.batch_size

        if _idx%announceBatchStep == 0:
            print("%s Epoch %d | [%d/%d (%.2f%%)] Loss %.4f, Accuaracy %.2f%%"
                  %(status, epoch, current, total, (100*current/total), lossSum/current, 100*correct/current)
                 , end='\r')

    del dataPathList, datas, labels
    current_device()

    lossAvg = lossSum / total
    accAvg = 100 * correct / total
    print("%s Epoch %d | Avarage Loss %.4f, Average Accuracy %.2f%%"%(status, epoch, lossAvg, accAvg))

    infos = {"dataPathList" : outDataPathList,
             "originalClasses" : outOriginalClasses,
             "predictionClasses" : outPredictionClasses,
             "probability" : outProbability}

    return lossAvg, accAvg, infos

def PlotProcess(processRecord):

    epochs = []
    trainLoss = []
    validLoss = []
    trainAccuracy = []
    validAccuracy = []

    for epoch in processRecord["loss"]:

        epochs.append(epoch)
        trainLoss.append(processRecord["loss"][epoch]["train"])
        validLoss.append(processRecord["loss"][epoch]["valid"])
        trainAccuracy.append(processRecord["accuracy"][epoch]["train"])
        validAccuracy.append(processRecord["accuracy"][epoch]["valid"])

    figure(figsize=(15, 5))
    subplot(1,2,1)
    title("loss")
    plot(epochs, trainLoss, label="train", color="red")
    plot(epochs, validLoss, label="valid", color="blue")
    xlabel("epoch")
    ylabel("loss")
    legend()
    subplot(1,2,2)
    title("accuracy")
    plot(epochs, trainAccuracy, label="train", color="red")
    plot(epochs, validAccuracy, label="valid", color="blue")
    xlabel("epoch")
    ylabel("accuracy")
    legend()
    savefig("PlotProcess")
    clf()

    return None