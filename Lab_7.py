#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py
import pickle


# In[147]:


infile_1 = open ("higgs_100000_pt_250_500.pkl",'rb')
infile_2 = open ("higgs_100000_pt_1000_1200.pkl",'rb')
infile_3 = open ("qcd_100000_pt_250_500.pkl",'rb')
infile_4 = open ("qcd_100000_pt_1000_1200.pkl",'rb')

new_dict_1 = pickle.load(infile_1)
new_dict_2 = pickle.load(infile_2)
new_dict_3 = pickle.load(infile_3)
new_dict_4 = pickle.load(infile_4)


# Below are the expected counts for Higgs and QCD events out of a sample size of 100,000, followed by the normalization coefficients associated with each type of event. For the purposes of the lab, only the high momentum datasets and expected yields will be used.

# In[190]:


NH_high = 50
NQ_high = 2000

NH_Hnorm = NH_high / 100000
NQ_Hnorm = NQ_high / 100000


# ## Event selection optimization 
# 
# You and your lab partner should pick different pT (transverse momentum) samples for this lab. In each pT sample, there are dedicated training samples for event selection optimization. All studies should be carried out by normalizing Higgs and QCD samples in each pT sample to give expected yields accordingly (See Dataset descriptions).
# 
# 1. Make a stacked histogram plot for the feature variable: mass
# 
#     *Evaluate expected significance without any event selection.
#     
#     *Use Poisson statistics for significance calculation
#     
#     *Compare the exact significance to the approximation  NHiggs/(√NQCD) . If they are equivalent, explain your findings.

# In[191]:


counts1, bins1 = np.histogram(new_dict_2['mass'], bins=100000)
counts2, bins2 = np.histogram(new_dict_4['mass'], bins=100000)


# In[192]:


counts = [counts2*NQ_Hnorm, counts1*NH_Hnorm]
print(np.sum(counts2*NQ_Hnorm))
print(np.sum(counts1*NH_Hnorm))


# The "counts" variable above defines weights by which the counts of our datasets can be normalized such that all counts in all bins for each type of event should match the expected number of events. Taking the sums for each to double check if this is true, we can see that it is and our normalization has been done properly.
# 
# Below is a stacked histogram plot of the mass variable distributions for the background (QCD) and signal (Higgs).

# In[193]:


plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(1, 1)

mass = ['QCD Distribution', 'Higgs Distribution']
ax.hist((new_dict_4['mass'], new_dict_2['mass']), 100, density=False, histtype='bar', stacked=True, weights=counts, label=mass)
ax.set_xlabel('Mass')
ax.set_ylabel('Counts')
ax.set_title('Normalized Stacked Mass Distribution')
ax.legend()


# As we can see, there is overlap between the two distributions and if we were to use the entirety of these datasets we would not get significant discrimination between the background and the signal. However, there is a broad range of masses over which only the background is distributed. This suggests that making a cut of the dataset to exclude these masses may improve our results. First, however, we should do some calculations to see what level of discrimination we have currently, and also what thresholds would be required for a significant measurement if we left the dataset as is.
# 
# If we treat the QCD background as a Poisson distribution with a mean of 2000 (equal to the expected yield for the QCD background in a sample of 100,000 points), we can judge the thresholds required for 5 sigma significance. Likewise, we can calculate what significance we have currently.

# In[194]:


P51 = stats.norm.sf(5, loc=0, scale=1)
P52 = stats.norm.cdf(-5, loc=0, scale=1)


# In[195]:


q_norm = NQ_high

q_down = stats.poisson.ppf(P51, q_norm, loc=0)
q_up = stats.poisson.isf(P52, q_norm, loc=0)

print(q_down)
print(q_up)


# The calculations above show that in order for our observation to have 5 sigma significance, we would need to measure the signal plus the background as lower than or equal to 1780 or higher than or equal to 2228. Since tha sum of our expected values for the background and the signal is 2050, we already know that it is not significant as is.
# 
# Below are calculations to find exactly how significant it is.

# In[153]:


NQ_sd = np.sqrt(NQ_high)
s = NH_high / NQ_sd

s1 = stats.poisson.sf(2050, q_norm, loc=0)
s2 = stats.norm.isf(s1, loc=0, scale=1)

print(s)
print(s2)


# The second value is the significance of a Higgs signal in a QCD background using the entirety of the mass datasets for each. The first number is the approximation $\frac{N_{Higgs}}{\sqrt{N_{QCD}}}$, which is the number of standard deviations the expected signal is from the mean of the expected background. Since they are equivalent down to one tenth of a standard deviation, this shows that it is a pretty fair approximation for our purposes.

# 2. Identify mass cuts to optimize the expected significance.
# 
#     *Try different mass cuts systematically
# 
#     *Evaluate expected significance for each set of mass cuts
# 
#     *Identify the set of mass cuts which give you the highest significance.
# 
#     *Make two sets of stacked histogram plots for the rest of the features

# From the plot above, we know that the QCD background has a large right-hand tail of higher masses which the Higgs distribution does not occur over, so the first cut we will make will be to exclude masses above the highest occuring Higgs mass. After this cut, we can re-calculate the significance of our newly-shortened datasets and go from there.

# In[154]:


print(np.max(new_dict_2['mass']))


# In[155]:


new_Q = np.empty([])
new_Q = new_dict_4[(new_dict_4['mass'] <= 155)]
new_H = np.empty([])
new_H = new_dict_2[(new_dict_2['mass'] <= 155)]
print(new_Q['mass'].size)
print(new_H['mass'].size)
print(new_Q['pt'].size)
print(new_H['pt'].size)


# In[198]:


NQ_new = NQ_high * new_Q['mass'].size / 100000
NH_new = NH_high * new_H['mass'].size / 100000

s3 = stats.poisson.sf(NQ_new + NH_new, NQ_new, loc=0)
s4 = stats.norm.isf(s3, loc=0, scale=1)

print(s4)


# As the result above shows, the calculated significance did improve with our cut, but not enough to allow significant discrimination between event types. Therefore, we need to make further cuts of our datasets according to mass in order to maximize the significance we can achieve by limiting this variable.
# 
# Below is a series of different cuts of both distributions and calculations of the achieved significance for each.

# In[199]:


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
# This is a mch better result than we had before, but it is still not good enough to allow significant discrimination between the distributions. Therefore, we need to look at the rest of the variables in our datasets in order to determine if there are others which we can narrow down in order to improve the resultant significance.

# 3. Set A without any event selection
# 
#     *Can you identify another feature as discriminative as mass? (i.e. equal or better significance after feature cut)

# Below, the weights for each variable (as part of a 100,000 point dataset) are defined.

# In[201]:


counts1, bins1 = np.histogram(new_dict_2['pt'], bins=100000)
counts2, bins2 = np.histogram(new_dict_4['pt'], bins=100000)

counts3, bins3 = np.histogram(new_dict_2['eta'], bins=100000)
counts4, bins4 = np.histogram(new_dict_4['eta'], bins=100000)

counts5, bins5 = np.histogram(new_dict_2['phi'], bins=100000)
counts6, bins6 = np.histogram(new_dict_4['phi'], bins=100000)

counts7, bins7 = np.histogram(new_dict_2['ee2'], bins=100000)
counts8, bins8 = np.histogram(new_dict_4['ee2'], bins=100000)

counts9, bins9 = np.histogram(new_dict_2['ee3'], bins=100000)
counts10, bins10 = np.histogram(new_dict_4['ee3'], bins=100000)

counts11, bins11 = np.histogram(new_dict_2['d2'], bins=100000)
counts12, bins12 = np.histogram(new_dict_4['d2'], bins=100000)

counts13, bins13 = np.histogram(new_dict_2['angularity'], bins=100000)
counts14, bins14 = np.histogram(new_dict_4['angularity'], bins=100000)

counts15, bins15 = np.histogram(new_dict_2['t1'], bins=100000)
counts16, bins16 = np.histogram(new_dict_4['t1'], bins=100000)

counts17, bins17 = np.histogram(new_dict_2['t2'], bins=100000)
counts18, bins18 = np.histogram(new_dict_4['t2'], bins=100000)

counts19, bins19 = np.histogram(new_dict_2['t3'], bins=100000)
counts20, bins20 = np.histogram(new_dict_4['t3'], bins=100000)

counts21, bins21 = np.histogram(new_dict_2['t21'], bins=100000)
counts22, bins22 = np.histogram(new_dict_4['t21'], bins=100000)

counts23, bins23 = np.histogram(new_dict_2['t32'], bins=100000)
counts24, bins24 = np.histogram(new_dict_4['t32'], bins=100000)

counts25, bins25 = np.histogram(new_dict_2['KtDeltaR'], bins=100000)
counts26, bins26 = np.histogram(new_dict_4['KtDeltaR'], bins=100000)

counts_pt = [counts2*NQ_Hnorm, counts1*NH_Hnorm]
counts_eta = [counts4*NQ_Hnorm, counts3*NH_Hnorm]
counts_phi = [counts6*NQ_Hnorm, counts5*NH_Hnorm]
counts_ee2 = [counts8*NQ_Hnorm, counts7*NH_Hnorm]
counts_ee3 = [counts10*NQ_Hnorm, counts9*NH_Hnorm]
counts_d2 = [counts12*NQ_Hnorm, counts11*NH_Hnorm]
counts_ang = [counts14*NQ_Hnorm, counts13*NH_Hnorm]
counts_t1 = [counts16*NQ_Hnorm, counts15*NH_Hnorm]
counts_t2 = [counts18*NQ_Hnorm, counts17*NH_Hnorm]
counts_t3 = [counts20*NQ_Hnorm, counts19*NH_Hnorm]
counts_t21 = [counts22*NQ_Hnorm, counts21*NH_Hnorm]
counts_t32 = [counts24*NQ_Hnorm, counts23*NH_Hnorm]
counts_KtDeltaR = [counts26*NQ_Hnorm, counts25*NH_Hnorm]


# Now, we will plot each normalized variable without any event selection. This may show us variables which we can use later on to optimize our selections.

# In[202]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)
labels = ['QCD Distribution', 'Higgs Distribution']

ax[0,0].hist((new_dict_4['pt'], new_dict_2['pt']), 100, density=False, histtype='bar', stacked=True, weights=counts_pt, label=labels)
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Normalized Stacked Momentum Distribution')
ax[0,0].legend()

ax[0,1].hist((new_dict_4['eta'], new_dict_2['eta']), 100, density=False, histtype='bar', stacked=True, weights=counts_eta, label=labels)
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Normalized Stacked Pseudorapidity Distribution')
ax[0,1].legend()

ax[1,0].hist((new_dict_4['phi'], new_dict_2['phi']), 100, density=False, histtype='bar', stacked=True, weights=counts_phi, label=labels)
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Normalized Stacked Angle Distribution')
ax[1,0].legend()

ax[1,1].hist((new_dict_4['ee2'], new_dict_2['ee2']), 100, density=False, histtype='bar', stacked=True, weights=counts_ee2, label=labels)
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Normalized Stacked E2-Correlation Distribution')
ax[1,1].legend()

ax[2,0].hist((new_dict_4['ee3'], new_dict_2['ee3']), 100, density=False, histtype='bar', stacked=True, weights=counts_ee3, label=labels)
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Normalized Stacked E3-Correlation Distribution')
ax[2,0].legend()

ax[2,1].hist((new_dict_4['d2'], new_dict_2['d2']), 100, density=False, histtype='bar', stacked=True, weights=counts_d2, label=labels)
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Normalized Stacked Jet Discrimination Distribution')
ax[2,1].legend()

ax[3,0].hist((new_dict_4['angularity'], new_dict_2['angularity']), 100, density=False, histtype='bar', stacked=True, weights=counts_ang, label=labels)
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Normalized Stacked Angularity Distribution')
ax[3,0].legend()

ax[3,1].hist((new_dict_4['t1'], new_dict_2['t1']), 100, density=False, histtype='bar', stacked=True, weights=counts_t1, label=labels)
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Normalized Stacked 1-Sub-jettiness Distribution')
ax[3,1].legend()

ax[4,0].hist((new_dict_4['t2'], new_dict_2['t2']), 100, density=False, histtype='bar', stacked=True, weights=counts_t2, label=labels)
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Normalized Stacked 2-Sub-jettiness Distribution')
ax[4,0].legend()

ax[4,1].hist((new_dict_4['t3'], new_dict_2['t3']), 100, density=False, histtype='bar', stacked=True, weights=counts_t3, label=labels)
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Normalized Stacked 3-Sub-jettiness Distribution')
ax[4,1].legend()

ax[5,0].hist((new_dict_4['t21'], new_dict_2['t21']), 100, density=False, histtype='bar', stacked=True, weights=counts_t21, label=labels)
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Normalized Stacked 21-Sub-jettiness Ratio Distribution')
ax[5,0].legend()

ax[5,1].hist((new_dict_4['t32'], new_dict_2['t32']), 100, density=False, histtype='bar', stacked=True, weights=counts_t32, label=labels)
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Normalized Stacked 32-Sub-jettiness Ratio Distribution')
ax[5,1].legend()

ax[6,0].hist((new_dict_4['KtDeltaR'], new_dict_2['KtDeltaR']), 100, density=False, histtype='bar', stacked=True, weights=counts_KtDeltaR, label=labels)
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Normalized Stacked Angular Distance Distribution')
ax[6,0].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# As these plots show, it is unlikely that the variables of momentum, pseudorapidity, angle, e3-correlation, jet discrimination, angularity, 1-sub-jettiness, or the 32-sub-jettiness ratio will provide any further discrimination ability between events. This is evident because the maxima and minima of both distributions occur in roughly the same places, and they follow similar distribution curves. However, the e2-correlation, 2-sub-jettiness, 3-sub-jettiness, 21-sub-jettiness ratio, and angular distance variables show enough difference that they could be candidates for further cuts which would improve the significance we can calculate.

#     *Set B with your optimal mass cuts
# 
#     *Can you identify another feature to further improve your expected signifiance?

# Before we make more cuts, we will plot each variable after the application of the mass cuts. This will give us an even better indication of which variabeles may give us good results.

# In[203]:


print(xH[4])
print(xQ[4])


# In[204]:


counts1, bins1 = np.histogram(new_H5['pt'], bins=6978)
counts2, bins2 = np.histogram(new_Q5['pt'], bins=6978)

counts3, bins3 = np.histogram(new_H5['eta'], bins=6978)
counts4, bins4 = np.histogram(new_Q5['eta'], bins=6978)

counts5, bins5 = np.histogram(new_H5['phi'], bins=6978)
counts6, bins6 = np.histogram(new_Q5['phi'], bins=6978)

counts7, bins7 = np.histogram(new_H5['ee2'], bins=6978)
counts8, bins8 = np.histogram(new_Q5['ee2'], bins=6978)

counts9, bins9 = np.histogram(new_H5['ee3'], bins=6978)
counts10, bins10 = np.histogram(new_Q5['ee3'], bins=6978)

counts11, bins11 = np.histogram(new_H5['d2'], bins=6978)
counts12, bins12 = np.histogram(new_Q5['d2'], bins=6978)

counts13, bins13 = np.histogram(new_H5['angularity'], bins=6978)
counts14, bins14 = np.histogram(new_Q5['angularity'], bins=6978)

counts15, bins15 = np.histogram(new_H5['t1'], bins=6978)
counts16, bins16 = np.histogram(new_Q5['t1'], bins=6978)

counts17, bins17 = np.histogram(new_H5['t2'], bins=6978)
counts18, bins18 = np.histogram(new_Q5['t2'], bins=6978)

counts19, bins19 = np.histogram(new_H5['t3'], bins=6978)
counts20, bins20 = np.histogram(new_Q5['t3'], bins=6978)

counts21, bins21 = np.histogram(new_H5['t21'], bins=6978)
counts22, bins22 = np.histogram(new_Q5['t21'], bins=6978)

counts23, bins23 = np.histogram(new_H5['t32'], bins=6978)
counts24, bins24 = np.histogram(new_Q5['t32'], bins=6978)

counts25, bins25 = np.histogram(new_H5['KtDeltaR'], bins=6978)
counts26, bins26 = np.histogram(new_Q5['KtDeltaR'], bins=6978)


# In[205]:


counts_pt = [counts2*NQ_Hnorm, counts1*NH_Hnorm]
counts_eta = [counts4*NQ_Hnorm, counts3*NH_Hnorm]
counts_phi = [counts6*NQ_Hnorm, counts5*NH_Hnorm]
counts_ee2 = [counts8*NQ_Hnorm, counts7*NH_Hnorm]
counts_ee3 = [counts10*NQ_Hnorm, counts9*NH_Hnorm]
counts_d2 = [counts12*NQ_Hnorm, counts11*NH_Hnorm]
counts_ang = [counts14*NQ_Hnorm, counts13*NH_Hnorm]
counts_t1 = [counts16*NQ_Hnorm, counts15*NH_Hnorm]
counts_t2 = [counts18*NQ_Hnorm, counts17*NH_Hnorm]
counts_t3 = [counts20*NQ_Hnorm, counts19*NH_Hnorm]
counts_t21 = [counts22*NQ_Hnorm, counts21*NH_Hnorm]
counts_t32 = [counts24*NQ_Hnorm, counts23*NH_Hnorm]
counts_KtDeltaR = [counts26*NQ_Hnorm, counts25*NH_Hnorm]


# In[206]:


plt.rcParams["figure.figsize"] = (20,35)
fig, ax = plt.subplots(7, 2)
labels = ['QCD Distribution', 'Higgs Distribution']

ax[0,0].hist((new_Q5['pt'], bins1[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_pt, label=labels)
ax[0,0].set_xlabel('Momentum')
ax[0,0].set_ylabel('Counts')
ax[0,0].set_title('Normalized Stacked Momentum Distribution')
ax[0,0].legend()

ax[0,1].hist((new_Q5['eta'], bins3[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_eta, label=labels)
ax[0,1].set_xlabel('Pseudorapidity')
ax[0,1].set_ylabel('Counts')
ax[0,1].set_title('Normalized Stacked Pseudorapidity Distribution')
ax[0,1].legend()

ax[1,0].hist((new_Q5['phi'], bins5[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_phi, label=labels)
ax[1,0].set_xlabel('Angle')
ax[1,0].set_ylabel('Counts')
ax[1,0].set_title('Normalized Stacked Angle Distribution')
ax[1,0].legend()

ax[1,1].hist((new_Q5['ee2'], bins7[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_ee2, label=labels)
ax[1,1].set_xlabel('E2-Correlation')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_title('Normalized Stacked E2-Correlation Distribution')
ax[1,1].legend()

ax[2,0].hist((new_Q5['ee3'], bins9[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_ee3, label=labels)
ax[2,0].set_xlabel('E3-Correlation')
ax[2,0].set_ylabel('Counts')
ax[2,0].set_title('Normalized Stacked E3-Correlation Distribution')
ax[2,0].legend()

ax[2,1].hist((new_Q5['d2'], bins11[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_d2, label=labels)
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('Counts')
ax[2,1].set_title('Normalized Stacked Jet Discrimination Distribution')
ax[2,1].legend()

ax[3,0].hist((new_Q5['angularity'], bins13[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_ang, label=labels)
ax[3,0].set_xlabel('Angularity')
ax[3,0].set_ylabel('Counts')
ax[3,0].set_title('Normalized Stacked Angularity Distribution')
ax[3,0].legend()

ax[3,1].hist((new_Q5['t1'], bins15[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t1, label=labels)
ax[3,1].set_xlabel('1-Sub-jettiness')
ax[3,1].set_ylabel('Counts')
ax[3,1].set_title('Normalized Stacked 1-Sub-jettiness Distribution')
ax[3,1].legend()

ax[4,0].hist((new_Q5['t2'], bins17[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t2, label=labels)
ax[4,0].set_xlabel('2-Sub-jettiness')
ax[4,0].set_ylabel('Counts')
ax[4,0].set_title('Normalized Stacked 2-Sub-jettiness Distribution')
ax[4,0].legend()

ax[4,1].hist((new_Q5['t3'], bins19[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t3, label=labels)
ax[4,1].set_xlabel('3-Sub-jettiness')
ax[4,1].set_ylabel('Counts')
ax[4,1].set_title('Normalized Stacked 3-Sub-jettiness Distribution')
ax[4,1].legend()

ax[5,0].hist((new_Q5['t21'], bins21[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t21, label=labels)
ax[5,0].set_xlabel('21-Sub-jettiness Ratio')
ax[5,0].set_ylabel('Counts')
ax[5,0].set_title('Normalized Stacked 21-Sub-jettiness Ratio Distribution')
ax[5,0].legend()

ax[5,1].hist((new_Q5['t32'], bins23[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t32, label=labels)
ax[5,1].set_xlabel('32-Sub-jettiness Ratio')
ax[5,1].set_ylabel('Counts')
ax[5,1].set_title('Normalized Stacked 32-Sub-jettiness Ratio Distribution')
ax[5,1].legend()

ax[6,0].hist((new_Q5['KtDeltaR'], bins25[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_KtDeltaR, label=labels)
ax[6,0].set_xlabel('Angular Distance')
ax[6,0].set_ylabel('Counts')
ax[6,0].set_title('Normalized Stacked Angular Distance Distribution')
ax[6,0].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# The plots shown above display even more dramatically the relationships we guessed at earlier. We can use these plots to make decisions about where to apply cuts for each variable in order to test which improves the significance by the largest amount.

# 4. Optimize event selections using multiple features (if necessary)
# 
#     *Find a set of feature cuts which achieve high expected significance.
# 
#     *Compare significance (before/after event selection) derived in your pT samples to your lab partner. Describe your findings.

# Below, cuts (based on the most recent group of plots) for each of the promising variables listed previously are tested for the significance that such a cut would result in.

# In[208]:


new_Q6 = np.empty([])
new_Q6 = new_Q5[(new_Q5['ee2'] <= 0.07) & (new_Q5['ee2'] >= 0.02)]
new_H6 = np.empty([])
new_H6 = new_H5[(new_H5['ee2'] <= 0.07) & (new_H5['ee2'] >= 0.02)]

new_Q7 = np.empty([])
new_Q7 = new_Q5[(new_Q5['t2'] <= 0.62) & (new_Q5['t2'] >= 0.12)]
new_H7 = np.empty([])
new_H7 = new_H5[(new_H5['t2'] <= 0.62) & (new_H5['t2'] >= 0.12)]

new_Q8 = np.empty([])
new_Q8 = new_Q5[(new_Q5['t3'] <= 0.45) & (new_Q5['t3'] >= 0.14)]
new_H8 = np.empty([])
new_H8 = new_H5[(new_H5['t3'] <= 0.45) & (new_H5['t3'] >= 0.14)]

new_Q9 = np.empty([])
new_Q9 = new_Q5[(new_Q5['t21'] <= 0.75) & (new_Q5['t21'] >= 0.15)]
new_H9 = np.empty([])
new_H9 = new_H5[(new_H5['t21'] <= 0.75) & (new_H5['t21'] >= 0.15)]

new_Q10 = np.empty([])
new_Q10 = new_Q5[(new_Q5['KtDeltaR'] <= 0.46) & (new_Q5['KtDeltaR'] >= 0)]
new_H10 = np.empty([])
new_H10 = new_H5[(new_H5['KtDeltaR'] <= 0.46) & (new_H5['KtDeltaR'] >= 0)]

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

print(s16)
print(s18)
print(s20)
print(s22)
print(s24)


# As we can see, the cut associates with 2-sub-jettiness results in the largest calculated significance. We thus apply this cut to the datasets, and then test the remaining variables for further possible improvements and optimization of the results of our significance calculations.

# In[209]:


new_Q11 = np.empty([])
new_Q11 = new_Q7[(new_Q7['ee2'] <= 0.07) & (new_Q7['ee2'] >= 0.02)]
new_H11 = np.empty([])
new_H11 = new_H7[(new_H7['ee2'] <= 0.07) & (new_H7['ee2'] >= 0.02)]

new_Q12 = np.empty([])
new_Q12 = new_Q7[(new_Q7['t3'] <= 0.45) & (new_Q7['t3'] >= 0.14)]
new_H12 = np.empty([])
new_H12 = new_H7[(new_H7['t3'] <= 0.45) & (new_H7['t3'] >= 0.14)]

new_Q13 = np.empty([])
new_Q13 = new_Q7[(new_Q7['t21'] <= 0.75) & (new_Q7['t21'] >= 0.15)]
new_H13 = np.empty([])
new_H13 = new_H7[(new_H7['t21'] <= 0.75) & (new_H7['t21'] >= 0.15)]

new_Q14 = np.empty([])
new_Q14 = new_Q7[(new_Q7['KtDeltaR'] <= 0.46) & (new_Q7['KtDeltaR'] >= 0)]
new_H14 = np.empty([])
new_H14 = new_H7[(new_H7['KtDeltaR'] <= 0.46) & (new_H7['KtDeltaR'] >= 0)]

xQ2 = np.array([new_Q11['ee2'].size, new_Q12['t3'].size, new_Q13['t21'].size, new_Q14['KtDeltaR'].size])
xH2 = np.array([new_H11['ee2'].size, new_H12['t3'].size, new_H13['t21'].size, new_H14['KtDeltaR'].size])

NH2 = xH2 * NH_Hnorm
NQ2 = xQ2 * NQ_Hnorm

s25 = stats.poisson.sf(NQ2[0] + NH2[0], NQ2[0], loc=0)
s26 = stats.norm.isf(s25, loc=0, scale=1)

s27 = stats.poisson.sf(NQ2[1] + NH2[1], NQ2[1], loc=0)
s28 = stats.norm.isf(s27, loc=0, scale=1)

s29 = stats.poisson.sf(NQ2[2] + NH2[2], NQ2[2], loc=0)
s30 = stats.norm.isf(s29, loc=0, scale=1)

s31 = stats.poisson.sf(NQ2[3] + NH2[3], NQ2[3], loc=0)
s32 = stats.norm.isf(s31, loc=0, scale=1)

print(s26)
print(s28)
print(s30)
print(s32)


# This round of tests shows that the cuts associated with the e2-correlation make the biggest difference to our significance. We will apply these cuts to the dataset. 
# 
# At this point, having made two separate variable cuts, we should plot the remaining data for the variables which we are interested in to see if we can get an even better idea of how we can cut them in order to maximize the possible significance.

# In[210]:


print(len(new_Q11['ee2']))
print(len(new_H11['ee2']))


# In[211]:


counts19, bins19 = np.histogram(new_H5['t3'], bins=3330)
counts20, bins20 = np.histogram(new_Q5['t3'], bins=3330)

counts21, bins21 = np.histogram(new_H5['t21'], bins=3330)
counts22, bins22 = np.histogram(new_Q5['t21'], bins=3330)

counts23, bins23 = np.histogram(new_H5['t32'], bins=3330)
counts24, bins24 = np.histogram(new_Q5['t32'], bins=3330)

counts25, bins25 = np.histogram(new_H5['KtDeltaR'], bins=3330)
counts26, bins26 = np.histogram(new_Q5['KtDeltaR'], bins=3330)


# In[212]:


counts_t3 = [counts20*NQ_Hnorm, counts19*NH_Hnorm]
counts_t21 = [counts22*NQ_Hnorm, counts21*NH_Hnorm]
counts_t32 = [counts24*NQ_Hnorm, counts23*NH_Hnorm]
counts_KtDeltaR = [counts26*NQ_Hnorm, counts25*NH_Hnorm]


# In[213]:


plt.rcParams["figure.figsize"] = (20,5)
fig, ax = plt.subplots(1, 3)
labels = ['QCD Distribution', 'Higgs Distribution']

ax[0].hist((new_Q11['t3'], bins19[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t3, label=labels)
ax[0].set_xlabel('3-Sub-jettiness')
ax[0].set_ylabel('Counts')
ax[0].set_title('Normalized Stacked 3-Sub-jettiness Distribution')
ax[0].legend()

ax[1].hist((new_Q11['t21'], bins21[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_t21, label=labels)
ax[1].set_xlabel('21-Sub-jettiness Ratio')
ax[1].set_ylabel('Counts')
ax[1].set_title('Normalized Stacked 21-Sub-jettiness Ratio Distribution')
ax[1].legend()

ax[2].hist((new_Q11['KtDeltaR'], bins25[:-1]), 100, density=False, histtype='bar', stacked=True, weights=counts_KtDeltaR, label=labels)
ax[2].set_xlabel('Angular Distance')
ax[2].set_ylabel('Counts')
ax[2].set_title('Normalized Stacked Angular Distance Distribution')
ax[2].legend()

plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)


# Using the abive plots as guides, we can then test out the following variable cuts to see which grant the greatest improvement.

# In[214]:


new_Q15 = np.empty([])
new_Q15 = new_Q11[(new_Q11['t3'] <= 0.45) & (new_Q11['t3'] >= 0.12)]
new_H15 = np.empty([])
new_H15 = new_H11[(new_H11['t3'] <= 0.45) & (new_H11['t3'] >= 0.12)]

new_Q16 = np.empty([])
new_Q16 = new_Q11[(new_Q11['t21'] <= 0.75) & (new_Q11['t21'] >= 0.13)]
new_H16 = np.empty([])
new_H16 = new_H11[(new_H11['t21'] <= 0.75) & (new_H11['t21'] >= 0.13)]

new_Q17 = np.empty([])
new_Q17 = new_Q11[(new_Q11['KtDeltaR'] <= 0.35) & (new_Q11['KtDeltaR'] >= 0.15)]
new_H17 = np.empty([])
new_H17 = new_H11[(new_H11['KtDeltaR'] <= 0.35) & (new_H11['KtDeltaR'] >= 0.15)]

xQ3 = np.array([new_Q15['t3'].size, new_Q16['t21'].size, new_Q17['KtDeltaR'].size])
xH3 = np.array([new_H15['t3'].size, new_H16['t21'].size, new_H17['KtDeltaR'].size])

NH3 = xH3 * NH_Hnorm
NQ3 = xQ3 * NQ_Hnorm

s33 = stats.poisson.sf(NQ3[0] + NH3[0], NQ3[0], loc=0)
s34 = stats.norm.isf(s33, loc=0, scale=1)

s35 = stats.poisson.sf(NQ3[1] + NH3[1], NQ3[1], loc=0)
s36 = stats.norm.isf(s35, loc=0, scale=1)

s37 = stats.poisson.sf(NQ3[2] + NH3[2], NQ3[2], loc=0)
s38 = stats.norm.isf(s37, loc=0, scale=1)

print(s34)
print(s36)
print(s38)


# As the calculations above show, this last cut with respect to the angular distance variable pushes us over the 5 sigma threshold, allowing us to make significant discriminations between events. It is possible that further variable cuts would improve it even more, but this should be enough to be able to confidently tell the difference between a QCD background event and a Higgs signal event.

# Bonus (optional):
# 
# Plot 2-dimensional plots using the top two most discriminative features.
# Can you find a curve or a linear combination in this 2D plane which gives even better sensitivity? Extended reading: Lab 7 is a classificaition problem using multi-dimensional features in supervised machine learning. We can use popular machine learning tools to develop an optimial classifier which can maximize information by using all features. For interested students, you can read https://scikit-learn.org/stable/supervised_learning.html

# In[ ]:




