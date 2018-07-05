import os
import pandas as pd
import pdb
import numpy as np
import math
import shelve
xls = pd.ExcelFile('dataset1.xls')
sheet1 = xls.parse(0) #2 is the sheet number
var1 = sheet1['File Name']
var2 = sheet1['Sample ID']
D = []
for i in range(1222):
    A = 'gcdDataset/'+var1[i]
    data = pd.read_csv(A, sep="\t", header=None)
    D.append(np.array(data[1]))
    print i
D = np.asarray(D)
Transcriptome = D.T
data = pd.read_csv(A, sep="\t", header=None)
Transcriptome_labels=np.array(data[0]);

xls = pd.ExcelFile('Dataset_Labels.xls', header=None)
sheet2 = xls.parse(0)
var1_2 = sheet2['Complete_TCGA_ID']
var2_2 = sheet2['PAM50_mRNA']
fatto = 0
Y = []
for i in range(1222):
    for j in range(825):
        if cmp(var2[i], var1_2[j])==0:
            fatto = 1
            if cmp(var2_2[j],'HER2-enriched')==0:
                Y.append('HER2-enriched')
            elif  cmp(var2_2[j],'Luminal A')==0:
                Y.append('Luminal A    ')
            elif  cmp(var2_2[j],'Basal-like')==0:
                Y.append('Basal-like   ')
            elif  cmp(var2_2[j],'Luminal B')==0:
                Y.append('Luminal B    ')
            elif  cmp(var2_2[j],'Normal-like')==0:
                Y.append('Normal-like  ')
            elif  math.isnan(var2_2[j]):
                Y.append('NA           ')

    if fatto == 0:
        if var2[i].find('-11') > -1:
            Y.append('Healty       ')
        else:
            Y.append('Not present  ')
    fatto = 0
Labels = ['Dataset_Labels2.xls', 'Dataset_Labels3.xls']
Dim = [1148,467]
z=0
for dataset in Labels:
    xls = pd.ExcelFile('Dataset_Labels2.xls', header=None)
    sheet2 = xls.parse(0)
    var1_2 = sheet2['Sample ID']
    var2_2 = sheet2['PAM50']
    for i in range(1222):
        for j in range(Dim[z]):
            if cmp(var2[i], var1_2[j])==0 and (cmp(Y[i],'Not present  ')==0 or cmp(Y[i],'NA           ')==0):
                if cmp(var2_2[j],'HER2-enriched')==0:
                    Y[i]='HER2-enriched'
                elif  cmp(var2_2[j],'Luminal A')==0:
                    Y[i]='Luminal A    '
                elif  cmp(var2_2[j],'Basal-like')==0:
                    Y[i]='Basal-like   '
                elif  cmp(var2_2[j],'Luminal B')==0:
                    Y[i]='Luminal B    '
                elif  cmp(var2_2[j],'Normal-like')==0:
                    Y[i]='Normal-like  '
                elif   math.isnan(var2_2[j]):
                    Y[i]='NA           '
    z = z+1
fil= shelve.open('Dataset.dat')
fil['Features'] = Transcriptome
fil['Features_labels'] = Transcriptome_labels
fil['Y'] = Y
fil.close()
pdb.set_trace()
