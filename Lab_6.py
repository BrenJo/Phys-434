#!/usr/bin/env python
# coding: utf-8

# ## Confidence intervals
# For this week we will be using the real data you are working on, but more as a background for the main activity rather than in depth explorations. Don't worry, more in-depth exploration the next two weeks!
# 
# #### Setup
# First, we want to have some `background' data (presumably signal free) onto which we can inject a simulated signal. So the first step is to isolate some appropriate data from your data files. 
# 
# #### LHC
# For the LHC data, we are going to have to fake a background. Make a Poisson background with mean 100. 

# In[84]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats


# In[85]:


bkgd = stats.poisson.rvs(100, loc=0, size=1000000)


# ### Problem 1
# The first problem is to look at your background distribution and determine where the 5σ sensitivity threshold is. [Hint, you've done this several times now, look back at the first few labs.]

# In[86]:


P5 = stats.norm.sf(5, loc=0, scale=1)
sig5_down = stats.poisson.ppf(P5, 100, loc=0)
sig5_up = stats.poisson.isf(P5, 100, loc=0)
print(P5)
print(sig5_down)
print(sig5_up)


# From our calculations above, we know that our 5$\sigma$ sensitivity threshold occurs at $\pm54$ from the mean (100) of our background distribution. This means that background events of 5$\sigma$ or more occur at or below a magnitude of 54 and at or above a magnitude of 154.

# ### Problem 2
# Now inject a signal of known strength. You will want to make your signal moderately strong, say somewhere in the 8-30σ range. Inject this signal into your background data many times.
# 
# a) Histogram how bright the observed signal appears to be, and discuss it's shape. Say in words what this histogram is telling you.
# 
# b) Is your observed signal biased? (e.g. is the observed value equally likely to be stronger or weaker than the true injected signal?) Is it symmetric?

# In[87]:


P8 = stats.norm.sf(8, loc=0, scale=1)
sig8_down = stats.poisson.ppf(P8, 100, loc=0)
sig8_up = stats.poisson.isf(P8, 100, loc=0)
print(P8)
print(sig8_down)
print(sig8_up)


# From the calculations above, we can see that for a signal with a strength above 8$\sigma$, we need it to be lower than 31 or higher than 190. For our purposes, then, we will choose a signal with a strength of 200.

# In[88]:


signal1 = np.full((1000000), 200)
data1 = bkgd + signal1


# In[89]:


plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(1, 2)
ax[0].hist(data1, 49, density=True, label='Observed Data')
ax[1].hist(data1, 49, density=True, label='Observed Data')
ax[0].set_title('Distribution of Observed Data')
ax[1].set_title('Distribution of Observed Data (Log Plot)')
ax[0].set_xlabel('Observed Data')
ax[1].set_xlabel('Observed Data')
ax[0].set_ylabel('Probability')
ax[1].set_ylabel('Probability (Log Scale)')
ax[1].semilogy()
ax[0].legend()
ax[1].legend()
plt.show()


# The plots above show the probability of each observed value we get in data collection. It appears to have a brightness which is symmetrically distributed around 300, and this distribution occurs entirely the upper 5-sigma threshold.
# 
# This distribution does show a bias, since it is not equally probable that we will measure a value lower or higher than our injected true signal. Since our signal is 200, and we have a Poisson distribution (which includes only positive integer values), our entire distribution spans values which are larger than our true signal. Therefore, we have zero probability of measuring an observed value less than 200 (our true signal).

# ### Problem 3
# Now make a suite of injected signals. You will want to have a range of injected signal strengths, starting at zero and extending well above 5σ (30σ or more). You will want to follow the work you did for Homework 5 very closely.
# 
# a) Clearly state what you are simulating, and make a 2D histogram of injected signal vs. observed data
# 
# b) For the same injected signal power as in problem 2, show that you get the same answer.
# 
# c) Now reverse the problem, select an observed data value (pick something quite a bit stronger than 5σ) and create a 1D histogram of the true signal probability given the observed data. Describe the meaning of this histogram.
# 
# d) For your observed signal, what is the 1σ uncertainty on the true signal strength?
# 
# e) Discuss the answer to part d in some depth. Is it symmetric? Is it biased? Does this make sense?

# In[90]:


P30 = stats.norm.sf(30, loc=0, scale=1)
sig30_down = stats.poisson.ppf(P30, 100, loc=0)
sig30_up = stats.poisson.isf(P30, 100, loc=0)
print(P30)
print(sig30_down)
print(sig30_up)


# Above, we tried to calculate exaclty where 30$\sigma$ would occur, but this calculation was beyond the limits of python to accomplish without information loss. So, we made a guess that a value above 500 would be above 30$\sigma$, and then produced a uniform true signal distribution from 0 to 550.

# In[91]:


x = np.linspace(0,550,1000000)
signal2 = np.random.choice(x, size=1000000, replace=True, p=None)

data2 = bkgd + signal2
max = np.max(data2)

signaledges = np.linspace(0,550,100)
dataedges = np.linspace(0,max,100)

Psd, temp, temp2= np.histogram2d(data2, signal2, bins=[dataedges,signaledges], density=True)

datacenters = (dataedges[:-1] + dataedges[1:]) / 2
signalcenters = (signaledges[:-1] + signaledges[1:]) / 2

plt.rcParams["figure.figsize"] = (15,15)

plt.pcolormesh(datacenters, signalcenters, Psd.T, shading='auto')
plt.ylabel('True signal, $P(s|d)$', fontsize = 24)
plt.xlabel('Observed data, $P(d|s)$', fontsize = 24)


# The above plot shows the density distribution of our observed data as we sweep through our range of true signals. Each horizontal slice represents the probability distribution of observed data we might measure if we were to inject the particular true signal associated with that slice into our system. This is $P(d|s)$. With each increased signal value, the distribution shifts to an increased mean. Likewise, each vertical slice represents the probability distribution of true signals that could have been injected if it were the case that we measured or observed a particular value. This is $P(s|d)$. As our observed values increase, the distributions of possible true signals associated with those observations occur around an increased mean.

# In[92]:


print(max)
print(np.where(signaledges == 200))


# In[93]:


plt.rcParams["figure.figsize"] = (15,10)

plt.step(dataedges[1:], Psd[:,36], label='Probability of observed data')
#plt.stairs(dataedges[1:], Psd[:,36], label='Probability of observed data', fill=True)
plt.ylabel('Probability', fontsize = 18)
plt.xlabel('Observed data', fontsize = 18)
plt.vlines(x=signaledges[36], ymin=0, ymax=0.00007, color='r', label='Signal')
plt.legend()
plt.title('Distribution of Observed Data Given a Signal of 200, $P(d|s=200)$', fontsize = 20)


# The plot above shows that we get the same result from taking a horizontal slice of our probability density distribution at the true signal value of 200. We would get a distribution of observed data symmetric around a mean of 300, and we still have zero probability that the observed value we measure will be less than our true signal.

# In[94]:


print(dataedges[68])


# In[96]:


plt.rcParams["figure.figsize"] = (15,10)

plt.step(signaledges[1:], Psd[68,:], label='Probability of true signal')
plt.ylabel('Probability', fontsize = 18)
plt.xlabel('True Signal', fontsize = 18)
plt.vlines(x=dataedges[68], ymin=0, ymax=0.00007, color='r', label='Observation')
plt.legend()
plt.title('Distribution of True Signal w/ a Given Observation, $P(s|d)$', fontsize = 20)


# The plot above shows a vertical slice of the probability density plot from before, where the slice occurs at a select observed value. From this plot, we can see that the probability distribution of true signals corresponding to a given measured data value will occur around a mean that is less than the observed measurement. This is exactly what we would expect, given that the background distribution has a positive-value bias and therefore we can expect that every measurement will occur at a higher magnitude than the range of true signals that may have produced it.

# In[97]:


mu = dataedges[68] - 100

P1 = stats.norm.sf(1, loc=0, scale=1)
sig1_down = stats.poisson.ppf(P1, mu, loc=0)
sig1_up = stats.poisson.isf(P1, mu, loc=0)
print(P1)
print(mu)
print(sig1_down)
print(sig1_up)
print(mu - sig1_down)
print(sig1_up - mu)


# From the calculations above, we find that the 1$\sigma$ uncertainty for true signal strength is approximately $\pm19$ (give or take a few decimal places, based on our background population of random numbers).
# 
# This shows that, to a certain number of decimal places, we will have symmetric true-signal uncertainty without a bias, because those values above the mean of the true signal distribution associated with our measurement will be just as probable as their corresponding values below the mean.
# 
# This result makes sense, because our original background distribution is a Poisson distribution, which is a symmetrical distribution. Therefore, since our observational data is the combination of the background and the true signal (which provides the bias discussed earlier between the true signal and observation distribution), we can expect every true signal distribution to be a Poisson distribution with symmetrical uncertainty around that distribution's mean.
# 
# The bias that occurs will be in the relationship of the observed value to the uncertainty in the true-value distribution, because all values in the true-value distribution will be lower than the observed value.

# In[ ]:




