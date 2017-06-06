import matplotlib.pyplot as plt

errors = []
itteractions = []


def addErrorLog(error, itteraction):
    global errors
    global itteractions

    error = map(abs, error)
    er = sum(error) / len(error)
    er = sum(er) / len(er)
    errors = errors + [er]
    itteractions = itteractions + [itteraction]


def saveLog(folder):
    errorsFile = open(folder + 'logErrors.txt', 'w+')
    errorsFile.write(str(errors))
    errorsFile.close()


def draw(fileToSave="", show=1):
    global errors
    global itteractions

    # prepare figure
    plt.figure()

    plt.plot(itteractions, errors)

    plt.draw()
    # save
    if (fileToSave != ""):
        plt.savefig(fileToSave + '.jpg')
    if (show == 1):
        plt.show()
    plt.close()
