#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
from underworld import function as fn
#import UWGeodynamics as GEO
#from underworld.scaling import units as u
#from underworld.scaling import dimensionalise, non_dimensionalise
import math
import numpy as np
import underworld.visualisation as glucifer
#import glucifer
import os
import matplotlib.pyplot as pyplot
from cycler import cycler

rank = uw.mpi.rank


# In[2]:


nmodels = 5
nseries = 4
inputPath = 'IceDripTxts/'
outputPath = 'IceDripFigure/'

width = np.empty([1,nmodels])
x1 = np.empty([nseries, nmodels])
x2 = np.empty([nseries, nmodels])

width[:] = np.loadtxt(inputPath + 'dripwidth.txt')
width=np.transpose(width)
print(width)
x1[:,:] = np.loadtxt(inputPath + 'x1.txt')
x1 = np.transpose(x1)
print(x1)
x2[:,:] = np.loadtxt(inputPath + 'x2.txt')
x2 = np.transpose(x2)
print(x2)



    


# In[3]:


stylecycle = (cycler(linestyle = ['-.',':','-.',':']) +
              cycler(color = ['g','g','b','b']) +
              cycler(lw = [2., 2., 2., 2.]) +
              cycler(marker = ['s','v','s','v']) +
              cycler(ms = [7., 7., 7., 7.]))


Figure = pyplot.figure()
Figure.set_size_inches(12,6)
ax1 = Figure.add_subplot(1,2,1)
ax1.set_prop_cycle(stylecycle)
#ax1.plot(tTracer, yTracer)
ax1.plot(width, x1)
ax1.set_ylim([0, 1.0])
ax1.set_title('a) Edge Tracer Movement',loc = 'left')
ax1.set_xlabel('Nucleus Width (km)')
ax1.set_ylabel('Surface Movement (km)')
ax1.legend(['$f_{d\_2}$ , Square',
            '$f_{d\_2}$ , Taper',
            '$f_{d\_3}$ , Square',
            '$f_{d\_3}$ , Taper',])


ax3 = Figure.add_subplot(1,2,2)
ax3.set_prop_cycle(stylecycle)
#ax3.plot(tTracer, sTracer)
ax3.plot(width, x2)
ax3.set_title('b) Interior Tracer Movement', loc = 'left')
ax3.set_xlabel('Nucleus Width (km)')
ax3.set_ylabel('Surface Movement (km)')
ax3.legend(['$f_{d\_2}$ , Square',
            '$f_{d\_2}$ , Taper',
            '$f_{d\_3}$ , Square',
            '$f_{d\_3}$ , Taper',])


Figure.show()
Figure.savefig(outputPath + 'Fig2.png')


# In[ ]:




