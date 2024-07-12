import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as le_me
import scipy
from scipy import stats
from scipy.stats import mannwhitneyu
from skimage import feature, transform

import json
import itertools

def readConf(confFile):
    """
    Read configuration
    :param confFile: file name
    :return: configuration dictionary
    """

    config = None
    with open(confFile, 'r') as f:
        config = json.load(f)


    return config


def sigTestAUC(data1, data2, disp='long'):
    '''
    return a string with AUC and significance based on the Mann Whitney test
    disp= short|long|auc
    '''
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    # p_value *= 2 # no longer required

    p_val_str = ''
    pValStars = ''
    if (p_value <= 0.001):
        p_val_str = '***p<0.001'
        pValStars = '***'
    elif (p_value <= 0.01):
        p_val_str = '**p<0.01'
        pValStars = '**'
    elif (p_value <= 0.05):
        p_val_str = '*p<0.05'
        pValStars = '*'
    else:
        p_val_str = 'not sig. p={:0.3f}'.format(p_value)
        pValStars = ''

    aucVal = 1 - u / (len(data1) * len(data2))

    if disp == 'short':
        strOut = '{:0.3f}{:}'.format(aucVal, pValStars)
    elif disp == 'long':
        strOut = '{:0.3f} ({:})'.format(aucVal, p_val_str)
    else:
        strOut = '{:0.3f}'.format(aucVal)

    return strOut

def mannAUC( data1, data2 ):
    """
    Compute AUC using the Mann Whitney test
    """
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    aucVal = 1 - u / (len(data1) * len(data2))
    
    return aucVal

def classSigTests( yIn, yPredProbArrIn, classesNamesIn ):
    """

    :param yIn: ground truth y, assumes classes are zero based indexed
    :param yPredProbArrIn:
    :param classesNamesIn:
    :return:
    """
    classIdArr = np.unique(yIn)
    for classId in classIdArr:
        # get probabilities 1 vs all
        probClass = yPredProbArrIn[ yIn==classId, classId ]
        probNoClass = yPredProbArrIn[yIn != classId, classId]
        # significance test
        testStr = sigTestAUC(probNoClass, probClass, disp='long')

        print( classesNamesIn[classId], ': ', testStr )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def bootStrapMetrics( y, yPred, dataRatio=0.8 ):
    BOOT_NUM = 1000 # number of bootstraps

    classesArr = np.unique(y)
    assert( np.max(classesArr)+1 == len(classesArr) )

    smplNum = len( y )
    bootSmplNum = int(smplNum * dataRatio)
    # create bootstraps indices with replacement
    rndIdx = np.random.randint(len(y), size=(BOOT_NUM, bootSmplNum))

    # select samples/labels
    yPredBoot = yPred[rndIdx]
    yBoot = y[rndIdx]
    #-- for each bootsrap
    resLst = []
    for bIdx in range(yBoot.shape[0]):
        yTmp = yBoot[bIdx,:]
        yPredTmp = yPredBoot[bIdx, :]

        # compute accuracy
        acc = (1.0 * np.sum(yTmp == yPredTmp)) / len(yTmp)

        # compute precision/recall/fscore
        prec, rec, fscore, _ = le_me.precision_recall_fscore_support(yTmp, yPredTmp, average='weighted')
        resLst.append( [acc, prec, rec, fscore] )

    resArr = np.array(resLst)
    # --
    # compute average with full set
    fullPrec, fullRec, fullFscore, _ = le_me.precision_recall_fscore_support(y, yPred, average='weighted')
    # compute accuracy with full set
    fullAcc = (1.0 * np.sum(y == yPred)) / len(y)

    med = np.median(resArr, axis=0)
    upConf = np.percentile(resArr, 95, axis=0)
    lowConf = np.percentile(resArr, 5, axis=0)

    print( 'Accuracy: {:.3f}, [{:.3f}-{:.3f}]'.format(fullAcc, lowConf[0], upConf[0]))
    print( 'Precision: {:.3f}, [{:.3f}-{:.3f}]'.format(fullPrec, lowConf[1], upConf[1]))
    print( 'Recall: {:.3f}, [{:.3f}-{:.3f}]'.format(fullRec, lowConf[2], upConf[2]))
    print( 'fscore: {:.3f}, [{:.3f}-{:.3f}]'.format(fullFscore, lowConf[3], upConf[3]))

    pass

def prBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute Precision-Recall curve using bootstrap.
    See plotPrAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, recallGridVec, precisionGridMat
    """
    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootPrLst = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    # create recall grid for interpolation
    recallGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all precision corresponding to recallGridVec
    precisionGridMat = np.zeros((len(recallGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.average_precision_score(y_true[indices], y_pred[indices])
        tmpPrecision, tmpRecall, _ = le_me.precision_recall_curve(y_true[indices], y_pred[indices])
#         tmpRecall = np.concatenate(([0], tmpRecall, [1]))
#         tmpPrecision = np.concatenate(([0], tmpPrecision, [1]))

        # interpolate for comparable ROCs
        fInter = scipy.interpolate.interp1d(tmpRecall, tmpPrecision, kind='nearest')
        precisionGridMat[:, i] = fInter(recallGridVec)

        bootstrapped_scores.append(score)
        bootPrLst.append([tmpRecall, tmpPrecision])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    precisionMean = np.mean(precisionGridMat, axis=1)
    averagePrecision = np.mean(precisionMean)

    return (averagePrecision, confidence_lower, confidence_upper, recallGridVec, precisionGridMat)

def plotPrAndConf(recallGridVec, precisionGridMat, labelIn=''):
    """
    Plot PR curve with confidence interval (estimated with Bootstrap). See prBootstrap function
    :param recallGridVec:
    :param precisionGridMat:
    :param labelIn:
    :return:
    """
    n_bootstraps = precisionGridMat.shape[1]

    # confidence interval for ROC
    precisionGridMatS = np.sort(precisionGridMat, axis=1)
    precisionLow025 = precisionGridMatS[:, int(0.025 * n_bootstraps)]
    precisionTop975 = precisionGridMatS[:, int(0.975 * n_bootstraps)]
    precisionMean = np.mean(precisionGridMat, axis=1)

    # plt.hold(True)
    ax = plt.gca()  # kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(recallGridVec[1:-1], precisionMean[1:-1], '-', linewidth=4, label=labelIn)
    ax.fill_between(recallGridVec, precisionLow025, precisionTop975, facecolor=base_line.get_color(), alpha=0.2)

def rocBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute ROC bootstrap.
    See plotRocAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, fprGridVec, tprGridMat
    """

    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootRocLst = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    # create fpr grid for interpolation
    fprGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all tpr corresponding to fprGridVec
    tprGridMat = np.zeros((len(fprGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.roc_auc_score(y_true[indices], y_pred[indices])
        tmpFpr, tmpTpr, _ = le_me.roc_curve(y_true[indices], y_pred[indices])
        tmpFpr = np.concatenate(([0], tmpFpr, [1]))
        tmpTpr = np.concatenate(([0], tmpTpr, [1]))

        # interpolate for comparable ROCs
        fInter = scipy.interpolate.interp1d(tmpFpr, tmpTpr, kind='nearest')
        tprGridMat[:, i] = fInter(fprGridVec)

        bootstrapped_scores.append(score)
        bootRocLst.append([tmpFpr, tmpTpr])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    tprMean = np.mean(tprGridMat, axis=1)
    averageTPR = np.mean(tprMean)

    return (averageTPR, confidence_lower, confidence_upper, fprGridVec, tprGridMat)

def plotRocAndConf(fprGridVec, tprGridMat, labelIn=''):
    """
    Plot ROC curve with confidence interval (estimated with Bootstrap). See rocBootstrap function
    :param fprGridVec:
    :param tprGridMat:
    :param labelIn:
    :return:
    """
    n_bootstraps = tprGridMat.shape[1]

    # confidence interval for ROC
    tprGridMatS = np.sort(tprGridMat, axis=1)
    tprLow025 = tprGridMatS[:, int(0.025 * n_bootstraps)]
    tprTop975 = tprGridMatS[:, int(0.975 * n_bootstraps)]
    tprMean = np.mean(tprGridMat, axis=1)

    # plt.hold(True)
    ax = plt.gca()  # kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(fprGridVec, tprMean, '-', linewidth=4, label=labelIn)
    ax.fill_between(fprGridVec, tprLow025, tprTop975, facecolor=base_line.get_color(), alpha=0.2)
    
def sensitivitySpecificityBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute sensitivity and specificity bootstrap.
    See plotRocAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, fprGridVec, tprGridMat
    """

    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    sensList = []
    specList = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.roc_auc_score(y_true[indices], y_pred[indices])
        tn, fp, fn, tp = le_me.confusion_matrix(y_true[indices], y_pred[indices]).ravel()
        sens = tp/(tp + fn)
        spec = tn/(tn + fp)

        specList.append(spec)
        sensList.append(sens)

    return (np.mean(sensList), np.mean(specList))

# Taken from DeepExplain
def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis




def pearsonr_ci(x,y,alpha=0.05):
    """ 
    calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    """

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi