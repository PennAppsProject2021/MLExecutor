import json

def getLearningRate():
    eturn getAttribute("learning_rate")

def getTrainingIters():
    return getAttribute("training_iters")

def getReportIter():
    return getAttribute("report_iter")

def setLearningRate(val):
    setAttribute("learning_rate", val)

def setTrainingIters(val):
    setAttribute("training_iters", val)

def setReportIter(val):
    setAttribute("report_iter", val)

def getAttribute(attribute):
    with open("hyperparameter_args.txt", "r") as jsonFile:
        data = json.load(jsonFile)
        return data[attribute]

def setAttribute(attribute, val):
    with open("hyperparameter_args.txt", "r") as jsonFile:
        data = json.load(jsonFile)

    data[attribute] = val

    with open("hyperparameter_args.txt", "w") as jsonFile:
        json.dump(data, jsonFile)
