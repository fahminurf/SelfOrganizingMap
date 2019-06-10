# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:55:12 2019

@author: King
"""

#importing library
import numpy as num
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data1 = pd.read_csv('Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv',header=None)

#scatter data tanpa label
#ax1 = data1.plot.scatter(x=0,y=1,c='navy')

lr0=0.1
tau=3
epoch=100
wlistx0=[]
wlisty0=[]


for u in range(15):
#    random.seed(4)
    randa = num.random.randint(0,599)
    wlistx0.append(data1.iloc[randa][0])
    wlisty0.append(data1.iloc[randa][1])

wSerx0=pd.DataFrame(wlistx0)
wSery0=pd.DataFrame(wlisty0)
cluster=[]

for i in range(epoch):
    for j in range(len(data1)):
        listjarak=[]
        a=data1.iloc[j][0]
        b=data1.iloc[j][1]
        for k in range(len(wlistx0)):
            jarak = num.sqrt( (a-wlistx0[k])**2 + (b-wlisty0[k])**2 )
            listjarak.append(jarak)
            min = listjarak.index(num.min(listjarak))
        tempx = lr0*(a-wlistx0[min])+wlistx0[min]
        tempy = lr0*(b-wlisty0[min])+wlisty0[min]
        lrtemp=lr0
        wlistx0[min]=tempx
        wlisty0[min]=tempy
        if(i==epoch-1):
            cluster.append(min)
    
    if lrtemp <= 0 :
        lr0=lrtemp
    else:
        lr0=lrtemp*num.exp(-i/tau)
    

# plot after label
        
plt.rc('font', size=11)


sns.set_style('white')

colors= ('red','green','blue','yellow','orange','magenta','black','darkgrey','darkmagenta','cyan','skyblue','pink','darkcyan','maroon','brown')
sns.set_palette(colors)

left= pd.read_csv('Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv',header=None,names=['x','y'])
right = pd.DataFrame(cluster,columns=['label'])
hamber = pd.concat([left,right], axis=1)
#print(hamber)

facet = sns.lmplot(data=hamber,x='x',y='y',hue='label',fit_reg=False, legend=True, legend_out=True)