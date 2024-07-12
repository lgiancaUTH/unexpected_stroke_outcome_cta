# %% [markdown]
# # Test AUC
# Feb 8, 2023

# %%
import pandas as pd
import os 
import utils
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.multitest as sm_mul
from statsmodels.stats.power import tt_ind_solve_power
import scipy.stats as stats
import sklearn.metrics as sl_me

GT_FILE = 'gt.csv' 
# PROB_DIR = 'out_prob/'
PROB_DIR = 'output_probs/'

# %%
gtFr = pd.read_csv(GT_FILE)


#only 0’s and 1’s at presentation

gtFr.columns

gtFr[['dc_mrs', 'day_mrs', 'premorbid_mrs']]


#only 0’s and 1’s at presentation
len(gtFr[(gtFr['premorbid_mrs'] == 0) | (gtFr['premorbid_mrs'] == 1)])


# %% [markdown]
# ## Cohort 1 
# Patients with small FIV (< 30 mL) with mRS 3-6 and premorbid_mrs < 2 
# vs
# FIV < 30 with mRS 0-2 

# %%
cntFr =  gtFr[(gtFr[ 'day_mrs' ] <= 2) & (gtFr[ 'fiv' ] <= 30) ]
casesFr =  gtFr[(gtFr[ 'day_mrs' ] > 2) & (gtFr[ 'fiv' ] <= 30) & ((gtFr['premorbid_mrs'] == 0) | (gtFr['premorbid_mrs'] == 1))]

len(cntFr), len(casesFr)

# %% [markdown]
# ## Load probabilities
# 

# %%
fLst = os.listdir(PROB_DIR)

probFr = None
i = 0
for f in fLst:
    resFr = pd.read_csv(PROB_DIR + f)
    # rename coluns
    resFr = resFr.rename(columns={'pred':'pred'+str(i), 'image': 'study_id'})
    # fix types
    resFr['label'] = resFr['label'].astype(int)
    # fix ID
    resFr['study_id'] = 'SAS-'+resFr['study_id'].str[-8:-4]

    if probFr is None:
        probFr = resFr
    else:
        del resFr['label']
        probFr = probFr.merge(resFr, on='study_id')

    i += 1
# aggregate probabilities
probFr['probMean'] = probFr[['pred'+str(i)  for i in range(len(fLst)) ] ].mean(axis=1)
probFr['probStd'] = probFr[['pred'+str(i)  for i in range(len(fLst)) ] ].std(axis=1)
probFr

# %% [markdown]
# ## ROC and AUC with bootstrap 

# %%
def formatROC():
    plt.legend( loc='lower right', prop={'size':13} ) 
    plt.xlabel('1-Specificity' )
    plt.ylabel('Sensitivity')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.tick_params(axis="y", labelsize=12 )
    plt.tick_params(axis="x", labelsize=12 )

probB = probFr[probFr['label']==0].probMean
probA = probFr[probFr['label']==1].probMean


_, aucConfDown, aucConfUp, fprGridVec, tprGridMat = utils.rocBootstrap(probB, probA)

lbl = 'AUC {:} [CI {:0.2f}-{:0.2}]'.format(  utils.sigTestAUC( probB, probA, 'auc' ), 
                                        aucConfDown, aucConfUp )

utils.plotRocAndConf(fprGridVec, tprGridMat, lbl)
formatROC()

# %% [markdown]
# ## ROC AUC without bootstrap

# %%
fpr, tpr, thresholds = sl_me.roc_curve( probFr['label'], probFr.probMean )
plt.plot(fpr, tpr, '-')
formatROC()
aucVal = sl_me.roc_auc_score( probFr['label'], probFr.probMean  )
print('AUC (no bootstrap)', aucVal)

# %%
fpr, tpr, thresholds = sl_me.roc_curve( probFr['label'], probFr.pred0 )
plt.plot(fpr, tpr, '-')
formatROC()
aucVal = sl_me.roc_auc_score( probFr['label'], probFr.pred0  )
print('AUC (no bootstrap)', aucVal)

# %%



