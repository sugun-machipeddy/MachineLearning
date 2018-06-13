import argparse
import numpy as np

import utils
import linear_model

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        print (XBin.shape)
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)

        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL1(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL0(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.5":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        print("Scikit learns L2 regularization")
        L2RegSk = LogisticRegression(penalty= 'l2', C=1.0, fit_intercept= False, verbose=0)
        L2RegSk.fit(XBin,yBin)
        print("\nTraining error %.3f" % utils.classification_error(L2RegSk.predict(XBin), yBin))
        print("Validation error %.3f" % utils.classification_error(L2RegSk.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (L2RegSk.coef_ != 0).sum())

        print(" our implemetnation of L2 regularization")
        L2Reg = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        L2Reg.fit(XBin, yBin)
        print("\nlogRegL1 Training error %.3f" % utils.classification_error(L2Reg.predict(XBin), yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(L2Reg.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (L2Reg.w != 0).sum())

        print("Scikit learns L1 regularization")
        L1RegSk = LogisticRegression(penalty='l1', C=1.0, fit_intercept=False, verbose=0)
        L1RegSk.fit(XBin, yBin)
        print("\nTraining error %.3f" % utils.classification_error(L1RegSk.predict(XBin), yBin))
        print("Validation error %.3f" % utils.classification_error(L1RegSk.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (L1RegSk.coef_ != 0).sum())

        print(" our implemetnation of L1 regularization")
        L1Reg = linear_model.logRegL1(lammy=1.0, maxEvals=400)
        L1Reg.fit(XBin, yBin)
        print("\nlogRegL1 Training error %.3f" % utils.classification_error(L1Reg.predict(XBin), yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(L1Reg.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (L1Reg.w != 0).sum())


    elif question == "3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMulti)))


    elif question == "3.2":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # TODO
        print("Scikit learns one vs all implementation")
        L2RegSk = LogisticRegression(penalty='l2', C=1000.0, fit_intercept=False, verbose=0)
        L2RegSk.fit(XMulti, yMulti)
        print("\nTraining error %.3f" % utils.classification_error(L2RegSk.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(L2RegSk.predict(XMultiValid), yMultiValid))

        print("our one vs all implementation")
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)
        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid),
                                                                                       yMultiValid))
        print("Scikit learns softmax implementation")
        L2RegSk = LogisticRegression(penalty='l2', C=1000.0, fit_intercept=False, verbose=0, multi_class = 'multinomial',
                                     solver = 'lbfgs')
        L2RegSk.fit(XMulti, yMulti)
        print("\nTraining error %.3f" % utils.classification_error(L2RegSk.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(L2RegSk.predict(XMultiValid), yMultiValid))

        print("our softmax implementation")
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)
        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))