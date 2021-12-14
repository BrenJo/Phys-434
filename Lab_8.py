#!/usr/bin/env python
# coding: utf-8

# ## Higgs Simulation Data Analysis

# In[347]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py
import pickle


# In[348]:


infile_1 = open ("higgs_100000_pt_250_500.pkl",'rb')
infile_2 = open ("higgs_100000_pt_1000_1200.pkl",'rb')
infile_3 = open ("qcd_100000_pt_250_500.pkl",'rb')
infile_4 = open ("qcd_100000_pt_1000_1200.pkl",'rb')

new_dict_1 = pickle.load(infile_1)
new_dict_2 = pickle.load(infile_2)
new_dict_3 = pickle.load(infile_3)
new_dict_4 = pickle.load(infile_4)


# Below are the expected counts for Higgs and QCD events out of a sample size of 100,000, followed by the normalization coefficients associated with each type of event. For the purposes of the lab, only the high momentum datasets and expected yields will be used.

# In[349]:


NH_high = 50
NQ_high = 2000

NH_Hnorm = NH_high / 100000
NQ_Hnorm = NQ_high / 100000


# ## Event selection optimization 
# 
# You and your lab partner should pick different pT (transverse momentum) samples for this lab. In each pT sample, there are dedicated training samples for event selection optimization. All studies should be carried out by normalizing Higgs and QCD samples in each pT sample to give expected yields accordingly (See Dataset descriptions).
# 
# #### 1. Make a stacked histogram plot for the feature variable: mass
# Evaluate expected significance without any event selection.
# 
# Use Poisson statistics for significance calculation
#     
# Compare the exact significance to the approximation  NHiggs/(√NQCD) . If they are equivalent, explain your findings.

# In[350]:


counts1, bins1 = np.histogram(new_dict_2['mass'], bins=100000)
counts2, bins2 = np.histogram(new_dict_4['mass'], bins=100000)


# In[351]:


NQ_Hnorm1 = np.full(100000, NQ_Hnorm)
NH_Hnorm1 = np.full(100000, NH_Hnorm)
countsA = [NQ_Hnorm1, NH_Hnorm1]
print(np.sum(counts2*NQ_Hnorm))
print(np.sum(counts1*NH_Hnorm))


# The "counts" variable above defines weights by which the counts of our datasets can be normalized such that all counts in all bins for each type of event should match the expected number of events. Taking the sums for each to double check if this is true, we can see that it is and our normalization has been done properly.
# 
# Below is a stacked histogram plot of the mass variable distributions for the background (QCD) and signal (Higgs).

# In[352]:


plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(1, 1)
colors = ['lightgrey','cadetblue']

mass = ['QCD Distribution', 'Higgs Distribution']
ax.hist((new_dict_4['mass'], new_dict_2['mass']), 100, density=False, histtype='bar', stacked=True, color=colors, weights=countsA, label=mass)
ax.set_xlabel('Mass')
ax.set_ylabel('Counts')
ax.set_title('Normalized Stacked Mass Distribution')
ax.legend()


# As we can see, there is overlap between the two distributions and if we were to use the entirety of these datasets we would not get significant discrimination between the background and the signal. However, there is a broad range of masses over which only the background is distributed. This suggests that making a cut of the dataset to exclude these masses may improve our results. First, however, we should do some calculations to see what level of discrimination we have currently, and also what thresholds would be required for a significant measurement if we left the dataset as is.
# 
# If we treat the QCD background as a Poisson distribution with a mean of 2000 (equal to the expected yield for the QCD background in a sample of 100,000 points), we can judge the thresholds required for 5 sigma significance. Likewise, we can calculate what significance we have currently.

# In[353]:


P51 = stats.norm.sf(5, loc=0, scale=1)
P52 = stats.norm.cdf(-5, loc=0, scale=1)


# In[354]:


q_norm = NQ_high

q_down = stats.poisson.ppf(P51, q_norm, loc=0)
q_up = stats.poisson.isf(P52, q_norm, loc=0)

print(q_down)
print(q_up)


# The calculations above show that in order for our observation to have 5 sigma significance, we would need to measure the signal plus the background as lower than or equal to 1780 or higher than or equal to 2228. Since tha sum of our expected values for the background and the signal is 2050, we already know that it is not significant as is.
# 
# Below are calculations to find exactly how significant it is.

# In[355]:


NQ_sd = np.sqrt(NQ_high)
s = NH_high / NQ_sd

s1 = stats.poisson.sf(2050, q_norm, loc=0)
s2 = stats.norm.isf(s1, loc=0, scale=1)

print(s)
print(s2)


# The second value is the significance of a Higgs signal in a QCD background using the entirety of the mass datasets for each. The first number is the approximation $\frac{N_{Higgs}}{\sqrt{N_{QCD}}}$, which is the number of standard deviations the expected signal is from the mean of the expected background. Since they are equivalent down to one tenth of a standard deviation, this shows that it is a pretty fair approximation for our purposes.

# #### 2. Identify mass cuts to optimize the expected significance.
# 
# Try different mass cuts systematically
# 
# Evaluate expected significance for each set of mass cuts
# 
# Identify the set of mass cuts which give you the highest significance.
# 
# Make two sets of stacked histogram plots for the rest of the features

# From the plot above, we know that the QCD background has a large right-hand tail of higher masses which the Higgs distribution does not occur over, so the first cut we will make will be to exclude masses above the highest occuring Higgs mass. After this cut, we can re-calculate the significance of our newly-shortened datasets and go from there.

# In[356]:


print(np.max(new_dict_2['mass']))


# In[357]:


new_Q = np.empty([])
new_Q = new_dict_4[(new_dict_4['mass'] <= 155)]
new_H = np.empty([])
new_H = new_dict_2[(new_dict_2['mass'] <= 155)]
print(new_Q['mass'].size)
print(new_H['mass'].size)
print(new_Q['pt'].size)
print(new_H['pt'].size)


# In[358]:


NQ_new = NQ_high * new_Q['mass'].size / 100000
NH_new = NH_high * new_H['mass'].size / 100000

s3 = stats.poisson.sf(NQ_new + NH_new, NQ_new, loc=0)
s4 = stats.norm.isf(s3, loc=0, scale=1)

print(s4)


# As the result above shows, the calculated significance did improve with our cut, but not enough to allow significant discrimination between event types. Therefore, we need to make further cuts of our datasets according to mass in order to maximize the significance we can achieve by limiting this variable.
# 
# Below is a series of different cuts of both distributions and calculations of the achieved significance for each.

# In[359]:


new_Q1 = np.empty([])
new_Q1 = new_dict_4[(new_dict_4['mass'] <= 155) & (new_dict_4['mass'] >= 90)]
new_H1 = np.empty([])
new_H1 = new_dict_2[(new_dict_2['mass'] <= 155) & (new_dict_2['mass'] >= 90)]

new_Q2 = np.empty([])
new_Q2 = new_dict_4[(new_dict_4['mass'] <= 145) & (new_dict_4['mass'] >= 105)]
new_H2 = np.empty([])
new_H2 = new_dict_2[(new_dict_2['mass'] <= 145) & (new_dict_2['mass'] >= 105)]

new_Q3 = np.empty([])
new_Q3 = new_dict_4[(new_dict_4['mass'] <= 140) & (new_dict_4['mass'] >= 120)]
new_H3 = np.empty([])
new_H3 = new_dict_2[(new_dict_2['mass'] <= 140) & (new_dict_2['mass'] >= 120)]

new_Q4 = np.empty([])
new_Q4 = new_dict_4[(new_dict_4['mass'] <= 135) & (new_dict_4['mass'] >= 125)]
new_H4 = np.empty([])
new_H4 = new_dict_2[(new_dict_2['mass'] <= 135) & (new_dict_2['mass'] >= 125)]

new_Q5 = np.empty([])
new_Q5 = new_dict_4[(new_dict_4['mass'] <= 135) & (new_dict_4['mass'] >= 124)]
new_H5 = np.empty([])
new_H5 = new_dict_2[(new_dict_2['mass'] <= 135) & (new_dict_2['mass'] >= 124)]

xQ = np.array([new_Q1['mass'].size, new_Q2['mass'].size, new_Q3['mass'].size, new_Q4['mass'].size, new_Q5['mass'].size])
xH = np.array([new_H1['mass'].size, new_H2['mass'].size, new_H3['mass'].size, new_H4['mass'].size, new_H5['mass'].size])

NH = xH * NH_Hnorm
NQ = xQ * NQ_Hnorm

s5 = stats.poisson.sf(NQ[0] + NH[0], NQ[0], loc=0)
s6 = stats.norm.isf(s5, loc=0, scale=1)

s7 = stats.poisson.sf(NQ[1] + NH[1], NQ[1], loc=0)
s8 = stats.norm.isf(s7, loc=0, scale=1)

s9 = stats.poisson.sf(NQ[2] + NH[2], NQ[2], loc=0)
s10 = stats.norm.isf(s9, loc=0, scale=1)

s11 = stats.poisson.sf(NQ[3] + NH[3], NQ[3], loc=0)
s12 = stats.norm.isf(s11, loc=0, scale=1)

s13 = stats.poisson.sf(NQ[4] + NH[4], NQ[4], loc=0)
s14 = stats.norm.isf(s13, loc=0, scale=1)

print('Cut 1:', s6)
print('Cut 2:', s8)
print('Cut 3:', s10)
print('Cut 4:', s12)
print('Cut 5:', s14)


# The results displayed above show that the fifth cut is the cut at which significance is maximized. Though not shown, narrower cuts were attempted after the fifth cut, but rather than increasing significance, they decreased it and therefore were not included as they did not improve upon the fifth cut.
# 
# This is a much better result than we had before, but it is still not good enough to allow significant discrimination between the distributions. Therefore, we need to look at the rest of the variables in our datasets in order to determine if there are others which we can narrow down in order to improve the resultant significance.

# #### 3. Make stacked histogram plots for the rest of features
# 
# ##### Set A without any event selection
# 
# Can you identify another feature as discriminative as mass? (i.e. equal or better significance after feature cut)

# Below, we will plot each variable without any event selection. This may show us variables which we can use later on to optimize our selections.

# In[360]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)
labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','cadetblue']

ax[0,0].hist((new_dict_4['pt'], new_dict_2['pt']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Normalized Stacked Momentum Distribution')
ax[0,0].legend()

ax[0,1].hist((new_dict_4['eta'], new_dict_2['eta']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Normalized Stacked Pseudorapidity Distribution')
ax[0,1].legend()

ax[1,0].hist((new_dict_4['phi'], new_dict_2['phi']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Normalized Stacked Angle Distribution')
ax[1,0].legend()

ax[1,1].hist((new_dict_4['ee2'], new_dict_2['ee2']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Normalized Stacked E2-Correlation Distribution')
ax[1,1].legend()

ax[2,0].hist((new_dict_4['ee3'], new_dict_2['ee3']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Normalized Stacked E3-Correlation Distribution')
ax[2,0].legend()

ax[2,1].hist((new_dict_4['d2'], new_dict_2['d2']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Normalized Stacked Jet Discrimination Distribution')
ax[2,1].legend()

ax[3,0].hist((new_dict_4['angularity'], new_dict_2['angularity']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Normalized Stacked Angularity Distribution')
ax[3,0].legend()

ax[3,1].hist((new_dict_4['t1'], new_dict_2['t1']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Normalized Stacked 1-Sub-jettiness Distribution')
ax[3,1].legend()

ax[4,0].hist((new_dict_4['t2'], new_dict_2['t2']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Normalized Stacked 2-Sub-jettiness Distribution')
ax[4,0].legend()

ax[4,1].hist((new_dict_4['t3'], new_dict_2['t3']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Normalized Stacked 3-Sub-jettiness Distribution')
ax[4,1].legend()

ax[5,0].hist((new_dict_4['t21'], new_dict_2['t21']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Normalized Stacked 21-Sub-jettiness Ratio Distribution')
ax[5,0].legend()

ax[5,1].hist((new_dict_4['t32'], new_dict_2['t32']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Normalized Stacked 32-Sub-jettiness Ratio Distribution')
ax[5,1].legend()

ax[6,0].hist((new_dict_4['KtDeltaR'], new_dict_2['KtDeltaR']), 100, density=False, histtype='bar', stacked=True, weights=countsA, color=colors, label=labels)
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Normalized Stacked Angular Distance Distribution')
ax[6,0].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# As these plots show, it is unlikely that the variables of momentum, pseudorapidity, angle, e3-correlation, jet discrimination, angularity, 1-sub-jettiness, or the 32-sub-jettiness ratio will provide any further discrimination ability between events. This is evident because the maxima and minima of both distributions occur in roughly the same places, and they follow similar distribution curves. However, the e2-correlation, 2-sub-jettiness, 3-sub-jettiness, 21-sub-jettiness ratio, and angular distance variables show enough difference that they could be candidates for further cuts which would improve the significance we can calculate.
# 
# Using the plots above, we can make reasonable cuts to each candidate variable and check to see if any give us a better significance than our initial mass cut, which returned approximately 1.538 sigma. If they do, perhaps we should change our strategy; if not, we know we are on the right track and should use mass as our first variable cut.

# In[361]:


new_Q51 = np.empty([])
new_Q51 = new_dict_4[(new_dict_4['ee2'] <= 0.075) & (new_dict_4['ee2'] >= 0.02)]
new_H51 = np.empty([])
new_H51 = new_dict_2[(new_dict_2['ee2'] <= 0.075) & (new_dict_2['ee2'] >= 0.02)]

new_Q52 = np.empty([])
new_Q52 = new_dict_4[(new_dict_4['t2'] <= 0.6) & (new_dict_4['t2'] >= 0.0)]
new_H52 = np.empty([])
new_H52 = new_dict_2[(new_dict_2['t2'] <= 0.6) & (new_dict_2['t2'] >= 0.0)]

new_Q53 = np.empty([])
new_Q53 = new_dict_4[(new_dict_4['t3'] <= 0.45) & (new_dict_4['t3'] >= 0.0)]
new_H53 = np.empty([])
new_H53 = new_dict_2[(new_dict_2['t3'] <= 0.45) & (new_dict_2['t3'] >= 0.0)]

new_Q54 = np.empty([])
new_Q54 = new_dict_4[(new_dict_4['t21'] <= 0.75) & (new_dict_4['t21'] >= 0.0)]
new_H54 = np.empty([])
new_H54 = new_dict_2[(new_dict_2['t21'] <= 0.75) & (new_dict_2['t21'] >= 0.0)]

new_Q55 = np.empty([])
new_Q55 = new_dict_4[(new_dict_4['KtDeltaR'] <= 0.46) & (new_dict_4['KtDeltaR'] >= 0.0)]
new_H55 = np.empty([])
new_H55 = new_dict_2[(new_dict_2['KtDeltaR'] <= 0.46) & (new_dict_2['KtDeltaR'] >= 0.0)]

xQ5 = np.array([new_Q51['t1'].size, new_Q52['t2'].size, new_Q53['t3'].size, new_Q54['t21'].size, new_Q55['KtDeltaR'].size])
xH5 = np.array([new_H51['t1'].size, new_H52['t2'].size, new_H53['t3'].size, new_H54['t21'].size, new_H55['KtDeltaR'].size])

NH5 = xH5 * NH_Hnorm
NQ5 = xQ5 * NQ_Hnorm

s55 = stats.poisson.sf(NQ5[0] + NH5[0], NQ5[0], loc=0)
s56 = stats.norm.isf(s55, loc=0, scale=1)

s57 = stats.poisson.sf(NQ5[1] + NH5[1], NQ5[1], loc=0)
s58 = stats.norm.isf(s57, loc=0, scale=1)

s59 = stats.poisson.sf(NQ5[2] + NH5[2], NQ5[2], loc=0)
s510 = stats.norm.isf(s59, loc=0, scale=1)

s511 = stats.poisson.sf(NQ5[3] + NH5[3], NQ5[3], loc=0)
s512 = stats.norm.isf(s511, loc=0, scale=1)

s513 = stats.poisson.sf(NQ5[4] + NH5[4], NQ5[4], loc=0)
s514 = stats.norm.isf(s513, loc=0, scale=1)

print('ee2:', s56)
print('t2:', s58)
print('t3:', s510)
print('t21:', s512)
print('KtDeltaR:', s514)


# The sigma values calculated above show that, though the significance of the E2-correlation comes close, none of the variables have the same significance. Certainly we can expect that these values might improve upon further cuts, but since none is markedly better that what we calculate for mass, we can safely go ahead and use mass as the variable to which we will apply our first cut.

# ##### Set B with optimal mass cuts
# 
# Can you identify another feature to further improve your expected signifiance?

# Before we make more cuts, we will plot each variable after the application of the mass cuts. This will give us an even better indication of which variabeles may give us good results.

# In[362]:


print(xH[4])
print(xQ[4])


# In[363]:


NQ_Hnorm2 = np.full(6978, NQ_Hnorm)
NH_Hnorm2 = np.full(89591, NH_Hnorm)
countsB = [NQ_Hnorm2, NH_Hnorm2]


# In[364]:


plt.rcParams["figure.figsize"] = (20,50)
fig, ax = plt.subplots(7, 2)
labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','cadetblue']

ax[0,0].hist((new_Q5['pt'], new_H5['pt']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Normalized Stacked Momentum Distribution After Mass Cut')
ax[0,0].legend()

ax[0,1].hist((new_Q5['eta'], new_H5['eta']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Normalized Stacked Pseudorapidity Distribution After Mass Cut')
ax[0,1].legend()

ax[1,0].hist((new_Q5['phi'], new_H5['phi']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Normalized Stacked Angle Distribution After Mass Cut')
ax[1,0].legend()

ax[1,1].hist((new_Q5['ee2'], new_H5['ee2']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Normalized Stacked E2-Correlation Distribution After Mass Cut')
ax[1,1].legend()

ax[2,0].hist((new_Q5['ee3'], new_H5['ee3']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Normalized Stacked E3-Correlation Distribution After Mass Cut')
ax[2,0].legend()

ax[2,1].hist((new_Q5['d2'], new_H5['d2']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Normalized Stacked Jet Discrimination Distribution After Mass Cut')
ax[2,1].legend()

ax[3,0].hist((new_Q5['angularity'], new_H5['angularity']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Normalized Stacked Angularity Distribution After Mass Cut')
ax[3,0].legend()

ax[3,1].hist((new_Q5['t1'], new_H5['t1']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Normalized Stacked 1-Sub-jettiness Distribution After Mass Cut')
ax[3,1].legend()

ax[4,0].hist((new_Q5['t2'], new_H5['t2']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Normalized Stacked 2-Sub-jettiness Distribution After Mass Cut')
ax[4,0].legend()

ax[4,1].hist((new_Q5['t3'], new_H5['t3']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Normalized Stacked 3-Sub-jettiness Distribution After Mass Cut')
ax[4,1].legend()

ax[5,0].hist((new_Q5['t21'], new_H5['t21']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Normalized Stacked 21-Sub-jettiness Ratio Distribution After Mass Cut')
ax[5,0].legend()

ax[5,1].hist((new_Q5['t32'], new_H5['t32']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Normalized Stacked 32-Sub-jettiness Ratio Distribution After Mass Cut')
ax[5,1].legend()

ax[6,0].hist((new_Q5['KtDeltaR'], new_H5['KtDeltaR']), 100, density=False, histtype='bar', stacked=True, weights=countsB, color=colors, label=labels)
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Normalized Stacked Angular Distance Distribution After Mass Cut')
ax[6,0].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# The plots shown above display the same relationships we guessed at earlier, but more dramatically . We can use these plots to make decisions about where to apply cuts for each variable in order to test which improves the significance by the largest amount.

# #### 4. Optimize event selections using multiple features (if necessary)
# 
# Find a set of feature cuts which achieve high expected significance.
# 
# Compare significance (before/after event selection) derived in your pT samples to your lab partner. Describe your findings.

# Below, secondary cuts (based on the most recent group of plots) for each of the promising variables listed previously are tested for the significance that such a cut would result in.

# In[365]:


new_Q6 = np.empty([])
new_Q6 = new_Q5[(new_Q5['ee2'] <= 0.08) & (new_Q5['ee2'] >= 0.02)]
new_H6 = np.empty([])
new_H6 = new_H5[(new_H5['ee2'] <= 0.08) & (new_H5['ee2'] >= 0.02)]

new_Q7 = np.empty([])
new_Q7 = new_Q5[(new_Q5['t2'] <= 0.6) & (new_Q5['t2'] >= 0.0)]
new_H7 = np.empty([])
new_H7 = new_H5[(new_H5['t2'] <= 0.6) & (new_H5['t2'] >= 0.0)]

new_Q8 = np.empty([])
new_Q8 = new_Q5[(new_Q5['t3'] <= 0.45) & (new_Q5['t3'] >= 0.0)]
new_H8 = np.empty([])
new_H8 = new_H5[(new_H5['t3'] <= 0.45) & (new_H5['t3'] >= 0.0)]

new_Q9 = np.empty([])
new_Q9 = new_Q5[(new_Q5['t21'] <= 0.75) & (new_Q5['t21'] >= 0.0)]
new_H9 = np.empty([])
new_H9 = new_H5[(new_H5['t21'] <= 0.75) & (new_H5['t21'] >= 0.0)]

new_Q10 = np.empty([])
new_Q10 = new_Q5[(new_Q5['KtDeltaR'] <= 0.7) & (new_Q5['KtDeltaR'] >= 0)]
new_H10 = np.empty([])
new_H10 = new_H5[(new_H5['KtDeltaR'] <= 0.7) & (new_H5['KtDeltaR'] >= 0)]

xQ1 = np.array([new_Q6['ee2'].size, new_Q7['t2'].size, new_Q8['t3'].size, new_Q9['t21'].size, new_Q10['KtDeltaR'].size])
xH1 = np.array([new_H6['ee2'].size, new_H7['t2'].size, new_H8['t3'].size, new_H9['t21'].size, new_H10['KtDeltaR'].size])

NH1 = xH1 * NH_Hnorm
NQ1 = xQ1 * NQ_Hnorm

s15 = stats.poisson.sf(NQ1[0] + NH1[0], NQ1[0], loc=0)
s16 = stats.norm.isf(s15, loc=0, scale=1)

s17 = stats.poisson.sf(NQ1[1] + NH1[1], NQ1[1], loc=0)
s18 = stats.norm.isf(s17, loc=0, scale=1)

s19 = stats.poisson.sf(NQ1[2] + NH1[2], NQ1[2], loc=0)
s20 = stats.norm.isf(s19, loc=0, scale=1)

s21 = stats.poisson.sf(NQ1[3] + NH1[3], NQ1[3], loc=0)
s22 = stats.norm.isf(s21, loc=0, scale=1)

s23 = stats.poisson.sf(NQ1[4] + NH1[4], NQ1[4], loc=0)
s24 = stats.norm.isf(s23, loc=0, scale=1)

print('ee2:', s16)
print('t2:', s18)
print('t3:', s20)
print('t21:', s22)
print('KtDeltaR:', s24)


# As we can see, the cut associated with 3-sub-jettiness results in the largest calculated significance, and it pushes the significance over the 5 sigma threshold, allowing us to make significant discriminations between events. It is possible that further variable cuts would improve it even more, but this should be enough to be able to confidently tell the difference between a QCD background event and a Higgs signal event.

# ## Pseudo-experiment data analysis 
# 
# Using your optimized event selection, hunt for your signal by using one of the pseduo-experiment dataset. For each task below, you will choose one of the observed data from your specific pT sample to perform the analysis.
# 
# #### 1. High luminosity data
# Focus on each feature of your event selection.
# Plot observed data, and overlap with expected signal and background (normalized to observed yields) without event selection.

# In[366]:


data_lo = pd.read_hdf('data_lowLumi_pt_1000_1200.h5')
data_hi = pd.read_hdf('data_highLumi_pt_1000_1200.h5')


# In[367]:


data_lo


# In[368]:


data_hi


# Below are some calculations to judge the size of the dataset and to ensure that the normalizations are working correctly.

# In[369]:


print(data_lo.shape,data_lo.size)


# In[370]:


print(data_hi.shape,data_hi.size)


# In[371]:


len_lo = len(data_lo)
len_hi = len(data_hi)


# In[372]:


counts2, bins2 = np.histogram(new_dict_4['pt'], bins=100000)
counts11, bins11 = np.histogram(new_dict_2['d2'], bins=100000)


# In[373]:


SQ_hi = 4066*((1+50/2000)**-1)
SQ_hi


# In[374]:


SH_hi = (50/2000)*SQ_hi
SH_hi


# In[375]:


SQ_hi+SH_hi


# Since the high luminosity dataset has 4066 entries, the calculations above confirm that our normalization is correct, since what we observe should be the sum of the background and signal events.

# In[376]:


norm_Q_hi = SQ_hi/100000
norm_H_hi = SH_hi/100000


# In[377]:


norm_Q_hi


# In[378]:


np.sum(counts2*norm_Q_hi)


# In[379]:


np.sum(counts11*norm_Q_hi)


# Below, we will plot all of the variables in the observed high luminosity dataset and compare them with the simulation distributions for Higgs events and QCD background events, which have been normalized for a sample size of the length of the high luminosity dataset.

# In[380]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)

labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','lightblue']

weightQ1 = np.full(100000, norm_Q_hi)
weightH1 = np.full(100000, norm_H_hi)
weight1 = [weightQ1, weightH1]

ax[0,0].hist((new_dict_4['pt'], new_dict_2['pt']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[0,0].hist(data_hi['pt'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Observed vs. Normalized Simulated Momentum Distribution')
ax[0,0].legend()

ax[0,1].hist((new_dict_4['eta'], new_dict_2['eta']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[0,1].hist(data_hi['eta'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Observed vs. Normalized Simulated Pseudorapidity Distribution')
ax[0,1].legend()

ax[1,0].hist((new_dict_4['phi'], new_dict_2['phi']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[1,0].hist(data_hi['phi'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Observed vs. Normalized Simulated Angle Distribution')
ax[1,0].legend()

ax[1,1].hist((new_dict_4['ee2'], new_dict_2['ee2']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[1,1].hist(data_hi['ee2'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Observed vs. Normalized Simulated E2-Correlation Distribution')
ax[1,1].legend()

ax[2,0].hist((new_dict_4['ee3'], new_dict_2['ee3']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[2,0].hist(data_hi['ee3'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Observed vs. Normalized Simulated E3-Correlation Distribution')
ax[2,0].legend()

ax[2,1].hist((new_dict_4['d2'], new_dict_2['d2']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[2,1].hist(data_hi['d2'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Observed vs. Normalized Simulated Jet Discrimination Distribution')
ax[2,1].legend()

ax[3,0].hist((new_dict_4['angularity'], new_dict_2['angularity']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[3,0].hist(data_hi['angularity'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Observed vs. Normalized Simulated Angularity Distribution')
ax[3,0].legend()

ax[3,1].hist((new_dict_4['t1'], new_dict_2['t1']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[3,1].hist(data_hi['t1'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Observed vs. Normalized Simulated 1-Sub-jettiness Distribution')
ax[3,1].legend()

ax[4,0].hist((new_dict_4['t2'], new_dict_2['t2']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[4,0].hist(data_hi['t2'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Observed vs. Normalized Simulated 2-Sub-jettiness Distribution')
ax[4,0].legend()

ax[4,1].hist((new_dict_4['t3'], new_dict_2['t3']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[4,1].hist(data_hi['t3'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Observed vs. Normalized Simulated 3-Sub-jettiness Distribution')
ax[4,1].legend()

ax[5,0].hist((new_dict_4['t21'], new_dict_2['t21']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[5,0].hist(data_hi['t21'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Observed vs. Normalized Simulated 21-Sub-jettiness Ratio Distribution')
ax[5,0].legend()

ax[5,1].hist((new_dict_4['t32'], new_dict_2['t32']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[5,1].hist(data_hi['t32'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Observed vs. Normalized Simulated 32-Sub-jettiness Ratio Distribution')
ax[5,1].legend()

ax[6,0].hist((new_dict_4['KtDeltaR'], new_dict_2['KtDeltaR']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[6,0].hist(data_hi['KtDeltaR'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Observed vs. Normalized Simulated Angular Distance Distribution')
ax[6,0].legend()

ax[6,1].hist((new_dict_4['mass'], new_dict_2['mass']), 100, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[6,1].hist(data_hi['mass'], 100, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,1].set_xlabel('Mass')
ax[6,1].set_ylabel('Counts')
ax[6,1].set_title('Observed vs. Normalized Simulated Mass Distribution')
ax[6,1].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# These plots show that our data simulation was a pretty good approximation of what we could expect to observe for most variables. Some simulation datasets, such as those for 1- and 2-sub-jettiness and jet discrimination, appear to have overshot the mark in comparison to what our observed dataset shows, but this does not matter much because none of these are variables by which we intend to discriminate between background and signal events.

# Plot observed data, and overlap with expected signal and background (normalized to observed yields) with optimal event selection.

# First, we must apply the optimizing cut we discovered earlier to the high luminosity data. Then we will need to normalize the simulation datasets which have already been cut to the new size of the cut hihg luminosity set.

# In[381]:


new_hi_1 = np.empty([])
new_hi_1 = data_hi[(data_hi['mass'] <= 135) & (data_hi['mass'] >= 124)]

new_hi_2 = np.empty([])
new_hi_2 = new_hi_1[(new_hi_1['t3'] <= 0.45) & (new_hi_1['t3'] >= 0.0)]

x1 = new_hi_2['t3'].size
print(x1)
print(new_hi_2.shape)
print(new_hi_2['d2'].size)


# In[382]:


new_Q8.shape


# In[383]:


new_H8.shape


# In[384]:


weightQ2= np.full(2704, norm_Q_hi)
weightH2 = np.full(87991, norm_H_hi)
weight2 = [weightQ2, weightH2]


# Below are plots of all the datasets that have been cut for optimization.

# In[385]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)

labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','lightblue']

#ax[0,0].hist((bins2[:-1], bins1[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,0].hist((new_Q8['pt'], new_H8['pt']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,0].hist(new_hi_2['pt'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Observed vs. Normalized Simulated Momentum After Cuts')
ax[0,0].legend()

#ax[0,1].hist((bins4[:-1], bins3[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,1].hist((new_Q8['eta'], new_H8['eta']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,1].hist(new_hi_2['eta'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Observed vs. Normalized Simulated Pseudorapidity After Cuts')
ax[0,1].legend()

#ax[1,0].hist((bins6[:-1], bins5[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,0].hist((new_Q8['phi'], new_H8['phi']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,0].hist(new_hi_2['phi'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Observed vs. Normalized Simulated Angle After Cuts')
ax[1,0].legend()

#ax[1,1].hist((bins8[:-1], bins7[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,1].hist((new_Q8['ee2'], new_H8['ee2']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,1].hist(new_hi_2['ee2'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Observed vs. Normalized Simulated E2-Correlation After Cuts')
ax[1,1].legend()

#ax[2,0].hist((bins10[:-1], bins9[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,0].hist((new_Q8['ee3'], new_H8['ee3']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,0].hist(new_hi_2['ee3'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Observed vs. Normalized Simulated E3-Correlation After Cuts')
ax[2,0].legend()

#ax[2,1].hist((bins12[:-1], bins11[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,1].hist((new_Q8['d2'], new_H8['d2']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,1].hist(new_hi_2['d2'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Observed vs. Normalized Simulated Jet Discrimination After Cuts')
ax[2,1].legend()

#ax[3,0].hist((bins14[:-1], bins13[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,0].hist((new_Q8['angularity'], new_H8['angularity']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,0].hist(new_hi_2['angularity'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Observed vs. Normalized Simulated Angularity After Cuts')
ax[3,0].legend()

#ax[3,1].hist((bins16[:-1], bins15[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,1].hist((new_Q8['t1'], new_H8['t1']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,1].hist(new_hi_2['t1'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Observed vs. Normalized Simulated 1-Sub-jettiness After Cuts')
ax[3,1].legend()

#ax[4,0].hist((bins18[:-1], bins17[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,0].hist((new_Q8['t2'], new_H8['t2']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,0].hist(new_hi_2['t2'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Observed vs. Normalized Simulated 2-Sub-jettiness After Cuts')
ax[4,0].legend()

#ax[4,1].hist((bins20[:-1], bins19[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,1].hist((new_Q8['t3'], new_H8['t3']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,1].hist(new_hi_2['t3'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Observed vs. Normalized Simulated 3-Sub-jettiness After Cuts')
ax[4,1].legend()

#ax[5,0].hist((bins22[:-1], bins21[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,0].hist((new_Q8['t21'], new_H8['t21']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,0].hist(new_hi_2['t21'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Observed vs. Normalized Simulated 21-Sub-jettiness Ratio After Cuts')
ax[5,0].legend()

#ax[5,1].hist((bins24[:-1], bins23[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,1].hist((new_Q8['t32'], new_H8['t32']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,1].hist(new_hi_2['t32'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Observed vs. Normalized Simulated 32-Sub-jettiness Ratio After Cuts')
ax[5,1].legend()

#ax[6,0].hist((bins26[:-1], bins25[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,0].hist((new_Q8['KtDeltaR'], new_H8['KtDeltaR']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,0].hist(new_hi_2['KtDeltaR'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Observed vs. Normalized Simulated Angular Distance After Cuts')
ax[6,0].legend()

#ax[6,1].hist((bins28[:-1], bins27[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,1].hist((new_Q8['mass'], new_H8['mass']), 30, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,1].hist(new_hi_2['mass'], 30, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,1].set_xlabel('Mass')
ax[6,1].set_ylabel('Counts')
ax[6,1].set_title('Observed vs. Normalized Simulated Mass After Cuts')
ax[6,1].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# We can see from the plots above that, though there is some irregularity present, most variables generally follow the same sort of distribution.

# Evaluate observed significance and compare your results to the expectation.

# We can use Poisson statistics to calculate the significance using the sum of the expected values for Higgs events and for QCD events (which is also the length of the dataset) as our observed value.

# In[399]:


new_Q8['t32'].size


# In[400]:


SQ_hi_2 = 220*((1+(SH_hi/SQ_hi))**-1)
SH_hi_2 = SQ_hi_2*(SH_hi/SQ_hi)

print(SQ_hi_2)
print(SH_hi_2)
print(SQ_hi_2+SH_hi_2)


# In[401]:


NQ2 = (220/4066)*SQ_hi
NH2 = (220/4066)*SH_hi

obs1 = NQ2 + NH2
mu1 = 4066*(2704/100000)

s65 = stats.poisson.sf(obs1, mu1, loc=0)
s66 = stats.norm.isf(s65, loc=0, scale=1)

print(obs1)
print(mu1)
print(s66)


# From the calculations above, we can see that our observed significance is actually higher than what we had calculated for our expected significance due to the cuts we made to the simulation datasets. While we expected a significance of around 5.431, we calculated an observed significance of 9.276.

# #### 2. Low luminosity data
# Focus on each feature of your event selection.
# Plot observed data, and overlap with expected signal and background (normalized to observed yields) without event selection.

# First, we need check the lengths and other aspects of the low luminosity dataset in order to properly normalize the simulation dataset with respect to a sample size equal to that of the low luminosity dataset.

# In[402]:


len(data_lo)


# In[403]:


SQ_lo = 442*((1+50/2000)**-1)
SQ_lo


# In[404]:


SH_lo = (50/2000)*SQ_lo
SH_lo


# In[405]:


SQ_lo+SH_lo


# Since this dataset has a sample-size of 442, the above calculations show that the expected values we have gotten are correct, and we can use them to normalize the simulation datasets.

# In[406]:


norm_Q_lo = SQ_lo/100000
norm_H_lo = SH_lo/100000


# Below are plots of each of the variables in the low luminosity dataset along with those same variables for each of the simulated distributions (after normalization to a 442-size set).

# In[407]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)

labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','lightblue']

weightQ1 = np.full(100000, norm_Q_lo)
weightH1 = np.full(100000, norm_H_lo)
weight1 = [weightQ1, weightH1]

ax[0,0].hist((new_dict_4['pt'], new_dict_2['pt']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[0,0].hist(data_lo['pt'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Observed vs. Normalized Simulated Momentum Distribution')
ax[0,0].legend()

ax[0,1].hist((new_dict_4['eta'], new_dict_2['eta']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[0,1].hist(data_lo['eta'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Observed vs. Normalized Simulated Pseudorapidity Distribution')
ax[0,1].legend()

ax[1,0].hist((new_dict_4['phi'], new_dict_2['phi']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[1,0].hist(data_lo['phi'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Observed vs. Normalized Simulated Angle Distribution')
ax[1,0].legend()

ax[1,1].hist((new_dict_4['ee2'], new_dict_2['ee2']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[1,1].hist(data_lo['ee2'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Observed vs. Normalized Simulated E2-Correlation Distribution')
ax[1,1].legend()

ax[2,0].hist((new_dict_4['ee3'], new_dict_2['ee3']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[2,0].hist(data_lo['ee3'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Observed vs. Normalized Simulated E3-Correlation Distribution')
ax[2,0].legend()

ax[2,1].hist((new_dict_4['d2'], new_dict_2['d2']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[2,1].hist(data_lo['d2'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Observed vs. Normalized Simulated Jet Discrimination Distribution')
ax[2,1].legend()

ax[3,0].hist((new_dict_4['angularity'], new_dict_2['angularity']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[3,0].hist(data_lo['angularity'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Observed vs. Normalized Simulated Angularity Distribution')
ax[3,0].legend()

ax[3,1].hist((new_dict_4['t1'], new_dict_2['t1']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[3,1].hist(data_lo['t1'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Observed vs. Normalized Simulated 1-Sub-jettiness Distribution')
ax[3,1].legend()

ax[4,0].hist((new_dict_4['t2'], new_dict_2['t2']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[4,0].hist(data_lo['t2'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Observed vs. Normalized Simulated 2-Sub-jettiness Distribution')
ax[4,0].legend()

ax[4,1].hist((new_dict_4['t3'], new_dict_2['t3']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[4,1].hist(data_lo['t3'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Observed vs. Normalized Simulated 3-Sub-jettiness Distribution')
ax[4,1].legend()

ax[5,0].hist((new_dict_4['t21'], new_dict_2['t21']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[5,0].hist(data_lo['t21'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Observed vs. Normalized Simulated 21-Sub-jettiness Ratio Distribution')
ax[5,0].legend()

ax[5,1].hist((new_dict_4['t32'], new_dict_2['t32']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[5,1].hist(data_lo['t32'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Observed vs. Normalized Simulated 32-Sub-jettiness Ratio Distribution')
ax[5,1].legend()

ax[6,0].hist((new_dict_4['KtDeltaR'], new_dict_2['KtDeltaR']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[6,0].hist(data_lo['KtDeltaR'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Observed vs. Normalized Simulated Angular Distance Distribution')
ax[6,0].legend()

ax[6,1].hist((new_dict_4['mass'], new_dict_2['mass']), 50, density=False, histtype='bar', stacked=True, weights=weight1, color=colors, label=labels)
ax[6,1].hist(data_lo['mass'], 50, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,1].set_xlabel('Mass')
ax[6,1].set_ylabel('Counts')
ax[6,1].set_title('Observed vs. Normalized Simulated Mass Distribution')
ax[6,1].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# Similarly to the high luminosity dataset, that for low luminosity shows that each of the variables generally follows the distribution laid out by the simulated distributions. However, the low luminosity variables show more irregularity, are more often than not distributed with lower magnitudes than predicted by the simulation data. This is not particularly surprising, since the low luminosity dataset is relatively small. Thus, it will not be leveled out in the same way it would if there were many more measurements.

# Plot observed data, and overlap with expected signal and background (normalized to observed yields) with optimal event selection.

# First, we will apply the optimization cut to the low luminosity data and then check the length of the new dataset.

# In[408]:


new_lo_1 = np.empty([])
new_lo_1 = data_lo[(data_lo['mass'] <= 135) & (data_lo['mass'] >= 124)]

new_lo_2 = np.empty([])
new_lo_2 = new_lo_1[(new_lo_1['t3'] <= 0.45) & (new_lo_1['t3'] >= 0.0)]

x2 = new_lo_2['t3'].size
print(x2)


# Below are plots of each of the variables in the low luminosity dataset after the optimization  cut, along with their corresponding variables in the simulation distributions (which have also been cut and normalized to the new length).

# In[409]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)

labels = ['QCD Distribution', 'Higgs Distribution']
colors = ['lightgrey','lightblue']

#ax[0,0].hist((bins2[:-1], bins1[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,0].hist((new_Q8['pt'], new_H8['pt']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,0].hist(new_lo_2['pt'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Observed vs. Normalized Simulated Momentum After Cuts')
ax[0,0].legend()

#ax[0,1].hist((bins4[:-1], bins3[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,1].hist((new_Q8['eta'], new_H8['eta']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[0,1].hist(new_lo_2['eta'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Observed vs. Normalized Simulated Pseudorapidity After Cuts')
ax[0,1].legend()

#ax[1,0].hist((bins6[:-1], bins5[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,0].hist((new_Q8['phi'], new_H8['phi']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,0].hist(new_lo_2['phi'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Observed vs. Normalized Simulated Angle After Cuts')
ax[1,0].legend()

#ax[1,1].hist((bins8[:-1], bins7[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,1].hist((new_Q8['ee2'], new_H8['ee2']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[1,1].hist(new_lo_2['ee2'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Observed vs. Normalized Simulated E2-Correlation After Cuts')
ax[1,1].legend()

#ax[2,0].hist((bins10[:-1], bins9[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,0].hist((new_Q8['ee3'], new_H8['ee3']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,0].hist(new_lo_2['ee3'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Observed vs. Normalized Simulated E3-Correlation After Cuts')
ax[2,0].legend()

#ax[2,1].hist((bins12[:-1], bins11[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,1].hist((new_Q8['d2'], new_H8['d2']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[2,1].hist(new_lo_2['d2'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Observed vs. Normalized Simulated Jet Discrimination After Cuts')
ax[2,1].legend()

#ax[3,0].hist((bins14[:-1], bins13[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,0].hist((new_Q8['angularity'], new_H8['angularity']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,0].hist(new_lo_2['angularity'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Observed vs. Normalized Simulated Angularity After Cuts')
ax[3,0].legend()

#ax[3,1].hist((bins16[:-1], bins15[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,1].hist((new_Q8['t1'], new_H8['t1']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[3,1].hist(new_lo_2['t1'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Observed vs. Normalized Simulated 1-Sub-jettiness After Cuts')
ax[3,1].legend()

#ax[4,0].hist((bins18[:-1], bins17[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,0].hist((new_Q8['t2'], new_H8['t2']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,0].hist(new_lo_2['t2'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Observed vs. Normalized Simulated 2-Sub-jettiness After Cuts')
ax[4,0].legend()

#ax[4,1].hist((bins20[:-1], bins19[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,1].hist((new_Q8['t3'], new_H8['t3']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[4,1].hist(new_lo_2['t3'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Observed vs. Normalized Simulated 3-Sub-jettiness After Cuts')
ax[4,1].legend()

#ax[5,0].hist((bins22[:-1], bins21[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,0].hist((new_Q8['t21'], new_H8['t21']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,0].hist(new_lo_2['t21'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Observed vs. Normalized Simulated 21-Sub-jettiness Ratio After Cuts')
ax[5,0].legend()

#ax[5,1].hist((bins24[:-1], bins23[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,1].hist((new_Q8['t32'], new_H8['t32']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[5,1].hist(new_lo_2['t32'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Observed vs. Normalized Simulated 32-Sub-jettiness Ratio After Cuts')
ax[5,1].legend()

#ax[6,0].hist((bins26[:-1], bins25[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,0].hist((new_Q8['KtDeltaR'], new_H8['KtDeltaR']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,0].hist(new_lo_2['KtDeltaR'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Observed vs. Normalized Simulated Angular Distance After Cuts')
ax[6,0].legend()

#ax[6,1].hist((bins28[:-1], bins27[:-1]), 100, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,1].hist((new_Q8['mass'], new_H8['mass']), 20, density=False, histtype='bar', stacked=True, weights=weight2, color=colors, label=labels)
ax[6,1].hist(new_lo_2['mass'], 20, density=False, fill=False, histtype='step', color='k', label='Observed Momenta')
ax[6,1].set_xlabel('Mass')
ax[6,1].set_ylabel('Counts')
ax[6,1].set_title('Observed vs. Normalized Simulated Mass After Cuts')
ax[6,1].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# Unfortunately, our "optimization" cut doesn't appear to have worked well enough for the low luminosity data; each observed variable shows counts much lower than predicted by our simulation data. Even increasing the bin size, we still don't get strong observational counts, and they are not smoothly or well-distributed. Therefore, we are unlikely to be able to get significant results using the optimization cut here.

# Evaluate observed significance and compare your results to expectation.

# In[410]:


NQ3 = (23/442)*SQ_lo
NH3 = (23/442)*SH_lo

print(NQ3)
print(NH3)


# In[411]:


obs2 = NQ3 + NH3
mu2 = 442*(2704/100000)

s75 = stats.poisson.sf(obs2, mu2, loc=0)
s76 = stats.norm.isf(s75, loc=0, scale=1)

print(obs2)
print(mu2)
print(s76)


# As expected, our observational significance falls short of our expected significance, and even the 5 sigma threshold, with a significance of only 2.989. This is not unusaul, considering that the length of the dataset remaining after the cuts was only 23. In general, this is too short a dataset (statistically speaking) to expect good, significant results over such a short breadth of measurement.

# #### 3. 95% Confidence Level of signal yields
# 
# In the low luminosity data, the observed significance is less than 5 σ. We will calculate the 95% confidence level upper limit of signal yield.
# Evaluate the expected 95% confidence level upper limit.

# Again, we will use Poisson statistics to calculate the upper value for which we can have confidence that it is higher than the mean 95% of the time.

# In[341]:


s86 = stats.poisson.ppf(0.95, mu2, loc=0)
print(s86)


# In[343]:


mu2


# Using the mean calculated from the scaling of the length done earlier, we calculate 18.0 as the expected 95% confidence level upper limit.

# Evaluate the observed 95% confidence level upper limit.

# Now, we will use the mean of the observed set (one half the length of the set).

# In[344]:


mu3 = obs2/2
s96 = stats.poisson.ppf(0.95, mu3, loc=0)
print(s96)


# This time, we calculate the observed 95% confidence level upper limit as 17.0.

# Compare expectation to observation. Comment on your finding.

# We can see that the upper level limit for the observation is lower than that for the expectation we had for it. This suggests that there is some uncertainty in our results, and that different observational periods resulting in differing datasets may return different results, especially when the datasets are small.

# In[ ]:




