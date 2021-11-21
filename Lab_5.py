#!/usr/bin/env python
# coding: utf-8

# ## Higgs Classification
# In this project we are going to look at an inclusive search for the standard model Higgs boson in pp collisions at √ s = 13 TeV at the LHC using simulated data [A. Schuy]. The Higgs bosons are produced with large transverse momentum (pT) and decaying to a bottom quark-antiquark pair. The Higgs candidates could be reconstructed as large-radius jets using Calorimeters. Due to large QCD backgorund containmination, the direct 5-sigma observation of this Higgs channel is not accomplished yet[Phys. Rev. Lett. 120, 071802 (2018)]. We are going to use a set of training datasets to optimize event selections in order to enhance the discovery sensitivity. The optimal event selections will be applied to a selected pseudo-experiment data.
# 
# Both of you will use the same training samples for analysis. Each sample contains 14 features: ‘pt', 'eta', 'phi', 'mass', 'ee2', 'ee3', 'd2', 'angularity', 't1', 't2', 't3', 't21', 't32', 'KtDeltaR' [Eur. Phys. J. C 79 (2019) 836]. You can explore different strategies for event selection optimization using training samples. The optimal event selection will be applied to pseudo-experiment data.
# 
# Download the training datasets from one of the two pT-range folders. In each folder, there are 2 files, each containing 100k jets. The signal dataset is labeled as “higgs” and the background dataset is labeled as “qcd.”
# 
# Explore the training data by addressing the following:
# 
# Do all of the features provide discrimination power between signal and background?
# Are there correlations among these features?
# Compute the expected discovery sensitivity (significance of the expected signal) by normalizing each sample appropriately.
# Develop a plan to optimize the discovery sensitivity by applying selections to these features.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py


# In[2]:


# import library
import pickle

# open the file of interest, and use pickle loading
infile_1 = open ("higgs_100000_pt_250_500.pkl",'rb')
infile_2 = open ("higgs_100000_pt_1000_1200.pkl",'rb')
infile_3 = open ("qcd_100000_pt_250_500.pkl",'rb')
infile_4 = open ("qcd_100000_pt_1000_1200.pkl",'rb')

new_dict_1 = pickle.load(infile_1)
new_dict_2 = pickle.load(infile_2)
new_dict_3 = pickle.load(infile_3)
new_dict_4 = pickle.load(infile_4)

# list all keys of the files
print('higgs low pt')
print(new_dict_1.keys())
print('higgs high pt')
print(new_dict_2.keys())
print('qcd low pt')
print(new_dict_3.keys())
print('qcd high pt')
print(new_dict_4.keys())


# In[3]:


# Print two variables, mass and d2, of the first 10 jets
for i in range(10):
 print(new_dict_1['mass'][i],new_dict_1['d2'][i])


# In[4]:


plt.rcParams["figure.figsize"] = (20,10)

fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['mass'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['mass'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['mass'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['mass'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Mass Distribution')
ax[0,1].set_title('High Energy Mass Distribution')
ax[1,0].set_title('Low Energy Mass Distribution (Log)')
ax[1,1].set_title('High Energy Mass Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Mass')
ax[1,1].set_xlabel('Mass')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# In the plots above, we see the mass distributions predicted for a QCD background distribution and for a distribution of Higgs boson events separated by high and low energies and repeated as semilog plots to emphasis the difference between the predictions. In this situation, the mass is the mass of a jet event. As we can see, there is clearly the ability available to discriminate between the two predictions for both low and high energies.

# In[5]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['pt'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['pt'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['pt'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['pt'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['pt'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['pt'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['pt'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['pt'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Momentum Distribution')
ax[0,1].set_title('High Energy Momentum Distribution')
ax[1,0].set_title('Low Energy Momentum Distribution (Log)')
ax[1,1].set_title('High Energy Momentum Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Momentum')
ax[1,1].set_xlabel('Momentum')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# These plot show the predicted distributions of the transverse momentum of a jet event for high and low energies. There is a good deal of overlap between them for both energy levels, so our ability to discriminate between the two may be more difficult for some momenta than others, and may also depend on the precision to which we can measure them.

# In[6]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['eta'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['eta'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['eta'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['eta'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['eta'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['eta'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['eta'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['eta'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Pseudorapidity Distribution')
ax[0,1].set_title('High Energy Pseudorapidity Distribution')
ax[1,0].set_title('Low Energy Pseudorapidity Distribution (Log)')
ax[1,1].set_title('High Energy Pseudorapidity Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Pseudorapidity')
ax[1,1].set_xlabel('Pseudorapidity')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# Pseudorapidity is a geometric quantity which describes a magnitude between $-\infty$ and $\infty$ which is produced as a function of the angle of a jet event trajectory from the beam path. It returns zero when the angle is 90 degrees and is larger as the angle changes in a way that reduces the arclength between the jet trajectory and the beam axis, such that it is $\infty$ when the trajectory is pointing directly along the beam path and $-\infty$ when it is anti-parallel.
# 
# From the plots above, we can see that both predictions share symmetry around zero. There is some ability to discriminate between the two (given we have high enough precision), and this will be easier for higher energies than it will be for low energies.

# In[7]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['phi'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['phi'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['phi'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['phi'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['phi'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['phi'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['phi'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['phi'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Angle Distribution')
ax[0,1].set_title('High Energy Angle Distribution')
ax[1,0].set_title('Low Energy Angle Distribution (Log)')
ax[1,1].set_title('High Energy Angle Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Radiation Angle')
ax[1,1].set_xlabel('Radiation Angle')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The radiation angle is the angle measured as a rotation around the beam axis, and goes from $-\pi$ to $\pi$. The plots show that, in relation to the radiation angle, there is a uniform probability distribution for both predictions and both energy levels. Therefore, we would be unable to tell the difference between one or the other by looking only at the radiation angle distribution.

# In[8]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['ee2'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['ee2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['ee2'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['ee2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['ee2'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['ee2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['ee2'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['ee2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy E2-Correlation Distribution')
ax[0,1].set_title('High Energy E2-Correlation Distribution')
ax[1,0].set_title('Low Energy E2-Correlation Distribution (Log)')
ax[1,1].set_title('High Energy E2-Correlation Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('E2-Correlation')
ax[1,1].set_xlabel('E2-Correlation')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The ee2 variable is the correlation function describing the energies and pair-wise angles of particles in a jet event with a substructure of two prongs and 3-point correlators. As we can see from the plots above, this variable does provide the opportunity for discrimination between the QCD background and a Higgs event, especially for high energy jets.

# In[9]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['ee3'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['ee3'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['ee3'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['ee3'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['ee3'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['ee3'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['ee3'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['ee3'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy E3-Correlation Distribution')
ax[0,1].set_title('High Energy E3-Correlation Distribution')
ax[1,0].set_title('Low Energy E3-Correlation Distribution (Log)')
ax[1,1].set_title('High Energy E3-Correlation Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('E3-Correlation')
ax[1,1].set_xlabel('E3-Correlation')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The ee3 variable is the correlation function describing the energies and pair-wise angles of particles in a jet event with a 3-prong substructure and 4-point correlators. Similarly to the distributions for the e2 energy correlation, this variable provides plenty of opportunity for discrimination between the QCD and Higgs, with an especially dramatic difference for high energy.

# In[10]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['d2'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['d2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['d2'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['d2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['d2'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['d2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['d2'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['d2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Jet Discrimination Distribution')
ax[0,1].set_title('High Energy Jet Discrimination Distribution')
ax[1,0].set_title('Low Energy Jet Discrimination Distribution (Log)')
ax[1,1].set_title('High Energy Jet Discrimination Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Jet Discrimination')
ax[1,1].set_xlabel('Jet Discrimination')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The jet discrimination is a function of the energy correlation of a three-prong jet event (ee3) multiplied by the cube of the ratio of the energy correlation of a one-prong event (ee1) and the energy correlation of a two-prong event (ee2). It is used to help determine the substructure of an event, and does show enough difference between distributions to be able to distinguish between the QCD background and a Higgs distribution.

# In[11]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['angularity'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['angularity'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['angularity'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['angularity'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['angularity'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['angularity'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['angularity'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['angularity'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Angularity Distribution')
ax[0,1].set_title('High Energy Angularity Distribution')
ax[1,0].set_title('Low Energy Angularity Distribution (Log)')
ax[1,1].set_title('High Energy Angularity Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Angularity')
ax[1,1].set_xlabel('Angularity')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# Angularity is a measure of the orientation and symmetry of energy flow inside a jet. The high energy distributions show distinguishability, but though it appears at first glance that we could do the same for the low energy, it might be a risky decision given the tail of lower probabilities for the QCD distribution that occur at higher angularity values.

# In[106]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['t1'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['t1'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['t1'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['t1'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['t1'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['t1'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['t1'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['t1'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy 1-Sub-jettiness Distribution')
ax[0,1].set_title('High Energy 1-Sub-jettiness Distribution')
ax[1,0].set_title('Low Energy 1-Sub-jettiness Distribution (Log)')
ax[1,1].set_title('High Energy 1-Sub-jettiness Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('1-Sub-jettiness')
ax[1,1].set_xlabel('1-Sub-jettiness')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The 1-sub-jettiness of a jet event is a measure of how well the conal distribution of particles in the jet fits a single jet substructure using the transverse momenta of the particles and the angular distance between the particles and the axis of the jet. Because the plots above show entire (or nearly entire) overlap between the Higgs and QCD distributions, this measure is not an appropriate feature to use for distinguishability.

# In[107]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['t2'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['t2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['t2'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['t2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['t2'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['t2'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['t2'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['t2'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy 2-Sub-jettiness Distribution')
ax[0,1].set_title('High Energy 2-Sub-jettiness Distribution')
ax[1,0].set_title('Low Energy 2-Sub-jettiness Distribution (Log)')
ax[1,1].set_title('High Energy 2-Sub-jettiness Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('2-Sub-jettiness')
ax[1,1].set_xlabel('2-Sub-jettiness')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The 2-sub-jettiness is a measure of how well the conal distribution of particles in a jet fits a dual jet substructure. There is a great deal of overlap between the different distributions for both energy levels, so it is unlikely that this would be a good variable to use for discrimination between the two.

# In[108]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['t3'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['t3'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['t3'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['t3'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['t3'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['t3'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['t3'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['t3'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy 3-Sub-jettiness Distribution')
ax[0,1].set_title('High Energy 3-Sub-jettiness Distribution')
ax[1,0].set_title('Low Energy 3-Sub-jettiness Distribution (Log)')
ax[1,1].set_title('High Energy 3-Sub-jettiness Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('3-Sub-jettiness')
ax[1,1].set_xlabel('3-Sub-jettiness')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The 3-sub-jettiness is a measure of how well the conal distribution of particles in a jet fits a triple jet substructure. Though the majority of the two distributions overlap, on the upper end of the tails for both energy levels there is some distinguishability.

# In[109]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['t21'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['t21'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['t21'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['t21'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['t21'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['t21'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['t21'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['t21'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy 21-Sub-jettiness Ratio Distribution')
ax[0,1].set_title('High Energy 21-Sub-jettiness Ratio Distribution')
ax[1,0].set_title('Low Energy 21-Sub-jettiness Ratio Distribution (Log)')
ax[1,1].set_title('High Energy 21-Sub-jettiness Ratio Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('21-Sub-jettiness Ratio')
ax[1,1].set_xlabel('21-Sub-jettiness Ratio')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The 2-1-sub-jettiness ratio is the ratio between the 2-subjettiness and the 1-subjettiness of a jet event. Though the distributions overlap, they may be distinguishable over large enough datasets since the peak probabilities for the Higgs distributions occur at ratio values which are in the lower probability tails of the QCD background distributions.

# In[110]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['t32'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['t32'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['t32'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['t32'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['t32'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['t32'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['t32'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['t32'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy 32-Sub-jettiness Ratio Distribution')
ax[0,1].set_title('High Energy 32-Sub-jettiness Ratio Distribution')
ax[1,0].set_title('Low Energy 32-Sub-jettiness Ratio Distribution (Log)')
ax[1,1].set_title('High Energy 32-Sub-jettiness Ratio Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('32-Sub-jettiness Ratio')
ax[1,1].set_xlabel('32-Sub-jettiness Ratio')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()

ax[1,0].legend()
ax[1,1].legend()


# The 3-2-sub-jettiness ratio is the ratio between the 3-subjettiness and the 2-subjettiness of a jet event. For the lower energies, the distributions entirely overlap, but for the higher energies there may be some distinguishability at the lowest values for the ratio.

# In[163]:


fig, ax = plt.subplots(2, 2)
ax[0,0].hist(new_dict_3['KtDeltaR'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(new_dict_1['KtDeltaR'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(new_dict_4['KtDeltaR'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(new_dict_2['KtDeltaR'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(new_dict_3['KtDeltaR'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(new_dict_1['KtDeltaR'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(new_dict_4['KtDeltaR'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(new_dict_2['KtDeltaR'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Low Energy Angular Distance Distribution')
ax[0,1].set_title('High Energy Angular Distance Distribution')
ax[1,0].set_title('Low Energy Angular Distance Distribution (Log)')
ax[1,1].set_title('High Energy Angular Distance Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('KtDeltaR')
ax[1,1].set_xlabel('KtDeltaR')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# Angular distance is the measure of distance between two subjets in a jet event with a large radial distance. From these plots, this looks like a variable with decent distinguishability between the distributions at larger angular distances.

# Below, we have multiple collections of plots in order to look for correlation between different variables. We can expect certain quantities to be correlated, such as a particular sub-jettiness, and its ratio with another sub-jettiness, but for other variables there may be more interesting relationships to find.

# In[133]:


plt.rcParams["figure.figsize"] = (20,20)
fig, ax = plt.subplots(6, 4)
ax[0,0].hist2d(new_dict_3['mass'], new_dict_3['pt'], bins=30)
f = ax[0,1].hist2d(new_dict_4['mass'], new_dict_4['pt'], bins=30)
ax[0,0].set_title('Mass vs. Momentum (Low Energy)')
ax[0,0].set_ylabel('Momentum')
ax[0,0].set_xlabel('Mass')
ax[0,1].set_ylabel('Momentum')
ax[0,1].set_xlabel('Mass')
ax[0,1].set_title('Mass vs. Momentum (High Energy)')

ax[0,2].hist2d(new_dict_3['mass'], new_dict_3['ee2'], bins=30)
ax[0,3].hist2d(new_dict_4['mass'], new_dict_4['ee2'], bins=30)
ax[0,2].set_title('Mass vs. 2-jet Energy Correlation (Low Energy)')
ax[0,2].set_ylabel('2-jet Energy Correlation')
ax[0,2].set_xlabel('Mass')
ax[0,3].set_ylabel('2-jet Energy Correlation')
ax[0,3].set_xlabel('Mass')
ax[0,3].set_title('Mass vs. 2-jet Energy Correlation (High Energy)')

ax[1,0].hist2d(new_dict_3['mass'], new_dict_3['ee3'], bins=30)
ax[1,1].hist2d(new_dict_4['mass'], new_dict_4['ee3'], bins=30)
ax[1,0].set_title('Mass vs. 3-jet Energy Correlation (Low Energy)')
ax[1,0].set_ylabel('3-jet Energy Correlation')
ax[1,0].set_xlabel('Mass')
ax[1,1].set_ylabel('3-jet Energy Correlation')
ax[1,1].set_xlabel('Mass')
ax[1,1].set_title('Mass vs. 3-jet Energy Correlation (High Energy)')

ax[1,2].hist2d(new_dict_3['mass'], new_dict_3['d2'], bins=30)
ax[1,3].hist2d(new_dict_4['mass'], new_dict_4['d2'], bins=30)
ax[1,2].set_title('Mass vs. Jet Discrimination (Low Energy)')
ax[1,2].set_ylabel('Jet Discrimination')
ax[1,2].set_xlabel('Mass')
ax[1,3].set_ylabel('Jet Discrimination')
ax[1,3].set_xlabel('Mass')
ax[1,3].set_title('Mass vs. Jet Discrimination (High Energy)')

ax[2,0].hist2d(new_dict_3['mass'], new_dict_3['angularity'], bins=30)
ax[2,1].hist2d(new_dict_4['mass'], new_dict_4['angularity'], bins=30)
ax[2,0].set_title('Mass vs. Angularity (Low Energy)')
ax[2,0].set_ylabel('Momentum')
ax[2,0].set_xlabel('Angularity')
ax[2,1].set_ylabel('Momentum')
ax[2,1].set_xlabel('Angularity')
ax[2,1].set_title('Mass vs. Angularity (High Energy)')

ax[2,2].hist2d(new_dict_3['mass'], new_dict_3['t1'], bins=30)
ax[2,3].hist2d(new_dict_4['mass'], new_dict_4['t1'], bins=30)
ax[2,2].set_title('Mass vs. 1-Sub-jettiness (Low Energy)')
ax[2,2].set_ylabel('1-Sub-jettiness')
ax[2,2].set_xlabel('Mass')
ax[2,3].set_ylabel('1-Sub-jettiness')
ax[2,3].set_xlabel('Mass')
ax[2,3].set_title('Mass vs. 1-Sub-jettiness (High Energy)')

ax[3,0].hist2d(new_dict_3['mass'], new_dict_3['t2'], bins=30)
ax[3,1].hist2d(new_dict_4['mass'], new_dict_4['t2'], bins=30)
ax[3,0].set_title('Mass vs. 2-Sub-jettiness (Low Energy)')
ax[3,0].set_ylabel('2-Sub-jettiness')
ax[3,0].set_xlabel('Mass')
ax[3,1].set_ylabel('2-Sub-jettiness')
ax[3,1].set_xlabel('Mass')
ax[3,1].set_title('Mass vs. 2-Sub-jettiness (High Energy)')

ax[3,2].hist2d(new_dict_3['mass'], new_dict_3['t3'], bins=30)
ax[3,3].hist2d(new_dict_4['mass'], new_dict_4['t3'], bins=30)
ax[3,2].set_title('Mass vs. 3-Sub-jettiness (Low Energy)')
ax[3,2].set_ylabel('3-Sub-jettiness')
ax[3,2].set_xlabel('Mass')
ax[3,3].set_ylabel('3-Sub-jettiness')
ax[3,3].set_xlabel('Mass')
ax[3,3].set_title('Mass vs. 3-Sub-jettiness (High Energy)')

ax[4,0].hist2d(new_dict_3['mass'], new_dict_3['t21'], bins=30)
ax[4,1].hist2d(new_dict_4['mass'], new_dict_4['t21'], bins=30)
ax[4,0].set_title('Mass vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[4,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[4,0].set_xlabel('Mass')
ax[4,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[4,1].set_xlabel('Mass')
ax[4,1].set_title('Mass vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[4,2].hist2d(new_dict_3['mass'], new_dict_3['t32'], bins=30)
ax[4,3].hist2d(new_dict_4['mass'], new_dict_4['t32'], bins=30)
ax[4,2].set_title('Mass vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[4,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[4,2].set_xlabel('Mass')
ax[4,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[4,3].set_xlabel('Mass')
ax[4,3].set_title('Mass vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[5,0].hist2d(new_dict_3['mass'], new_dict_3['KtDeltaR'], bins=30)
ax[5,1].hist2d(new_dict_4['mass'], new_dict_4['KtDeltaR'], bins=30)
ax[5,0].set_title('Mass vs. Angular Distance (Low Energy)')
ax[5,0].set_ylabel('Angular Distance')
ax[5,0].set_xlabel('Mass')
ax[5,1].set_ylabel('Angular Distance')
ax[5,1].set_xlabel('Mass')
ax[5,1].set_title('Mass vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)

#left  = the left side of the subplots of the figure
#right = the right side of the subplots of the figure
#bottom = the bottom of the subplots of the figure
#top = the top of the subplots of the figure
#wspace = the amount of width reserved for blank space between subplots
#hspace = the amount of height reserved for white space between subplots
plt.show()


# Above are the probabilty density distributions between mass and all other variables, and we can see that there is some positive correlation between mass and 2-jet energy correlation, mass and angularity, and mass and angular distance. There appears to be some negative correlation between mass and jet discrimination, mass and 2-sub-jettiness, mass and 3-sub-jettiness, and mass and both sub-jettiness ratios.

# In[134]:


plt.rcParams["figure.figsize"] = (20,20)
fig, ax = plt.subplots(6, 4)
ax[0,0].hist2d(new_dict_3['pt'], new_dict_3['eta'], bins=30)
f = ax[0,1].hist2d(new_dict_4['pt'], new_dict_4['eta'], bins=30)
ax[0,0].set_title('Momentum vs. Pseudorapidity (Low Energy)')
ax[0,0].set_ylabel('Pseudorapidity')
ax[0,0].set_xlabel('Momentum')
ax[0,1].set_ylabel('Pseudorapidity')
ax[0,1].set_xlabel('Momentum')
ax[0,1].set_title('Momentum vs. Pseudorapidity (High Energy)')

ax[0,2].hist2d(new_dict_3['pt'], new_dict_3['ee2'], bins=30)
ax[0,3].hist2d(new_dict_4['pt'], new_dict_4['ee2'], bins=30)
ax[0,2].set_title('Momentum vs. 2-jet Energy Correlation (Low Energy)')
ax[0,2].set_ylabel('2-jet Energy Correlation')
ax[0,2].set_xlabel('Momentum')
ax[0,3].set_ylabel('2-jet Energy Correlation')
ax[0,3].set_xlabel('Momentum')
ax[0,3].set_title('Momentum vs. 2-jet Energy Correlation (High Energy)')

ax[1,0].hist2d(new_dict_3['pt'], new_dict_3['ee3'], bins=30)
ax[1,1].hist2d(new_dict_4['pt'], new_dict_4['ee3'], bins=30)
ax[1,0].set_title('Momentum vs. 3-jet Energy Correlation (Low Energy)')
ax[1,0].set_ylabel('3-jet Energy Correlation')
ax[1,0].set_xlabel('Momentum')
ax[1,1].set_ylabel('3-jet Energy Correlation')
ax[1,1].set_xlabel('Momentum')
ax[1,1].set_title('Momentum vs. 3-jet Energy Correlation (High Energy)')

ax[1,2].hist2d(new_dict_3['pt'], new_dict_3['d2'], bins=30)
ax[1,3].hist2d(new_dict_4['pt'], new_dict_4['d2'], bins=30)
ax[1,2].set_title('Momentum vs. Jet Discrimination (Low Energy)')
ax[1,2].set_ylabel('Jet Discrimination')
ax[1,2].set_xlabel('Momentum')
ax[1,3].set_ylabel('Jet Discrimination')
ax[1,3].set_xlabel('Momentum')
ax[1,3].set_title('Momentum vs. Jet Discrimination (High Energy)')

ax[2,0].hist2d(new_dict_3['pt'], new_dict_3['angularity'], bins=30)
ax[2,1].hist2d(new_dict_4['pt'], new_dict_4['angularity'], bins=30)
ax[2,0].set_title('Momentum vs. Angularity (Low Energy)')
ax[2,0].set_ylabel('Angularity')
ax[2,0].set_xlabel('Momentum')
ax[2,1].set_ylabel('Angularity')
ax[2,1].set_xlabel('Momentum')
ax[2,1].set_title('Momentum vs. Angularity (High Energy)')

ax[2,2].hist2d(new_dict_3['pt'], new_dict_3['t1'], bins=30)
ax[2,3].hist2d(new_dict_4['pt'], new_dict_4['t1'], bins=30)
ax[2,2].set_title('Momentum vs. 1-Sub-jettiness (Low Energy)')
ax[2,2].set_ylabel('1-Sub-jettiness')
ax[2,2].set_xlabel('Momentum')
ax[2,3].set_ylabel('1-Sub-jettiness')
ax[2,3].set_xlabel('Momentum')
ax[2,3].set_title('Momentum vs. 1-Sub-jettiness (High Energy)')

ax[3,0].hist2d(new_dict_3['pt'], new_dict_3['t2'], bins=30)
ax[3,1].hist2d(new_dict_4['pt'], new_dict_4['t2'], bins=30)
ax[3,0].set_title('Momentum vs. 2-Sub-jettiness (Low Energy)')
ax[3,0].set_ylabel('2-Sub-jettiness')
ax[3,0].set_xlabel('Momentum')
ax[3,1].set_ylabel('2-Sub-jettiness')
ax[3,1].set_xlabel('Momentum')
ax[3,1].set_title('Momentum vs. 2-Sub-jettiness (High Energy)')

ax[3,2].hist2d(new_dict_3['pt'], new_dict_3['t3'], bins=30)
ax[3,3].hist2d(new_dict_4['pt'], new_dict_4['t3'], bins=30)
ax[3,2].set_title('Momentum vs. 3-Sub-jettiness (Low Energy)')
ax[3,2].set_ylabel('3-Sub-jettiness')
ax[3,2].set_xlabel('Momentum')
ax[3,3].set_ylabel('3-Sub-jettiness')
ax[3,3].set_xlabel('Momentum')
ax[3,3].set_title('Momentum vs. 3-Sub-jettiness (High Energy)')

ax[4,0].hist2d(new_dict_3['pt'], new_dict_3['t21'], bins=30)
ax[4,1].hist2d(new_dict_4['pt'], new_dict_4['t21'], bins=30)
ax[4,0].set_title('Momentum vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[4,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[4,0].set_xlabel('Momentum')
ax[4,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[4,1].set_xlabel('Momentum')
ax[4,1].set_title('Momentum vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[4,2].hist2d(new_dict_3['pt'], new_dict_3['t32'], bins=30)
ax[4,3].hist2d(new_dict_4['pt'], new_dict_4['t32'], bins=30)
ax[4,2].set_title('Momentum vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[4,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[4,2].set_xlabel('Momentum')
ax[4,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[4,3].set_xlabel('Momentum')
ax[4,3].set_title('Momentum vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[5,0].hist2d(new_dict_3['pt'], new_dict_3['KtDeltaR'], bins=30)
ax[5,1].hist2d(new_dict_4['pt'], new_dict_4['KtDeltaR'], bins=30)
ax[5,0].set_title('Momentum vs. Angular Distance (Low Energy)')
ax[5,0].set_ylabel('Angular Distance')
ax[5,0].set_xlabel('Momentum')
ax[5,1].set_ylabel('Angular Distance')
ax[5,1].set_xlabel('Momentum')
ax[5,1].set_title('Momentum vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# Throughout all of the probability density distributions of momentum with other variables, there don't appear to be any correlation patterns.

# In[135]:


fig, ax = plt.subplots(5, 4)

ax[0,0].hist2d(new_dict_3['ee2'], new_dict_3['ee3'], bins=30)
ax[0,1].hist2d(new_dict_4['ee2'], new_dict_4['ee3'], bins=30)
ax[0,0].set_title('2-jet Energy Correlation vs. 3-jet Energy Correlation (Low Energy)')
ax[0,0].set_ylabel('3-jet Energy Correlation')
ax[0,0].set_xlabel('2-jet Energy Correlation')
ax[0,1].set_ylabel('3-jet Energy Correlation')
ax[0,1].set_xlabel('2-jet Energy Correlation')
ax[0,1].set_title('2-jet Energy Correlation vs. 3-jet Energy Correlation (High Energy)')

ax[0,2].hist2d(new_dict_3['ee2'], new_dict_3['d2'], bins=30)
ax[0,3].hist2d(new_dict_4['ee2'], new_dict_4['d2'], bins=30)
ax[0,2].set_title('2-jet Energy Correlation vs. Jet Discrimination (Low Energy)')
ax[0,2].set_ylabel('Jet Discrimination')
ax[0,2].set_xlabel('2-jet Energy Correlation')
ax[0,3].set_ylabel('Jet Discrimination')
ax[0,3].set_xlabel('2-jet Energy Correlation')
ax[0,3].set_title('2-jet Energy Correlation vs. Jet Discrimination (High Energy)')

ax[1,0].hist2d(new_dict_3['ee2'], new_dict_3['angularity'], bins=30)
ax[1,1].hist2d(new_dict_4['ee2'], new_dict_4['angularity'], bins=30)
ax[1,0].set_title('2-jet Energy Correlation vs. Angularity (Low Energy)')
ax[1,0].set_ylabel('Angularity')
ax[1,0].set_xlabel('2-jet Energy Correlation')
ax[1,1].set_ylabel('Angularity')
ax[1,1].set_xlabel('2-jet Energy Correlation')
ax[1,1].set_title('2-jet Energy Correlation vs. Angularity (High Energy)')

ax[1,2].hist2d(new_dict_3['ee2'], new_dict_3['t1'], bins=30)
ax[1,3].hist2d(new_dict_4['ee2'], new_dict_4['t1'], bins=30)
ax[1,2].set_title('2-jet Energy Correlation vs. 1-Sub-jettiness (Low Energy)')
ax[1,2].set_ylabel('1-Sub-jettiness')
ax[1,2].set_xlabel('2-jet Energy Correlation')
ax[1,3].set_ylabel('1-Sub-jettiness')
ax[1,3].set_xlabel('2-jet Energy Correlation')
ax[1,3].set_title('2-jet Energy Correlation vs. 1-Sub-jettiness (High Energy)')

ax[2,0].hist2d(new_dict_3['ee2'], new_dict_3['t2'], bins=30)
ax[2,1].hist2d(new_dict_4['ee2'], new_dict_4['t2'], bins=30)
ax[2,0].set_title('2-jet Energy Correlation vs. 2-Sub-jettiness (Low Energy)')
ax[2,0].set_ylabel('2-Sub-jettiness')
ax[2,0].set_xlabel('2-jet Energy Correlation')
ax[2,1].set_ylabel('2-Sub-jettiness')
ax[2,1].set_xlabel('2-jet Energy Correlation')
ax[2,1].set_title('2-jet Energy Correlation vs. 2-Sub-jettiness (High Energy)')

ax[2,2].hist2d(new_dict_3['ee2'], new_dict_3['t3'], bins=30)
ax[2,3].hist2d(new_dict_4['ee2'], new_dict_4['t3'], bins=30)
ax[2,2].set_title('2-jet Energy Correlation vs. 3-Sub-jettiness (Low Energy)')
ax[2,2].set_ylabel('3-Sub-jettiness')
ax[2,2].set_xlabel('2-jet Energy Correlation')
ax[2,3].set_ylabel('3-Sub-jettiness')
ax[2,3].set_xlabel('2-jet Energy Correlation')
ax[2,3].set_title('2-jet Energy Correlation vs. 3-Sub-jettiness (High Energy)')

ax[3,0].hist2d(new_dict_3['ee2'], new_dict_3['t21'], bins=30)
ax[3,1].hist2d(new_dict_4['ee2'], new_dict_4['t21'], bins=30)
ax[3,0].set_title('2-jet Energy Correlation vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[3,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[3,0].set_xlabel('2-jet Energy Correlation')
ax[3,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[3,1].set_xlabel('2-jet Energy Correlation')
ax[3,1].set_title('2-jet Energy Correlation vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[3,2].hist2d(new_dict_3['ee2'], new_dict_3['t32'], bins=30)
ax[3,3].hist2d(new_dict_4['ee2'], new_dict_4['t32'], bins=30)
ax[3,2].set_title('2-jet Energy Correlation vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[3,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[3,2].set_xlabel('2-jet Energy Correlation')
ax[3,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[3,3].set_xlabel('2-jet Energy Correlation')
ax[3,3].set_title('2-jet Energy Correlation vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[4,0].hist2d(new_dict_3['ee2'], new_dict_3['KtDeltaR'], bins=30)
ax[4,1].hist2d(new_dict_4['ee2'], new_dict_4['KtDeltaR'], bins=30)
ax[4,0].set_title('2-jet Energy Correlation vs. Angular Distance (Low Energy)')
ax[4,0].set_ylabel('Angular Distance')
ax[4,0].set_xlabel('2-jet Energy Correlation')
ax[4,1].set_ylabel('Angular Distance')
ax[4,1].set_xlabel('2-jet Energy Correlation')
ax[4,1].set_title('2-jet Energy Correlation vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# For the probability density distributions involving 2-jet energy correlation, it appears there is a possible positive correlation between it and angular distance, as well as negative correlation between it and the 2- and 3-sub-jettiness, as well as both sub-jettiness ratios.

# In[148]:


plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(4, 4)

ax[0,2].hist2d(new_dict_3['ee3'], new_dict_3['d2'], bins=30)
ax[0,3].hist2d(new_dict_4['ee3'], new_dict_4['d2'], bins=30)
ax[0,2].set_title('3-jet Energy vs. Jet Discrimination (Low Energy)')
ax[0,2].set_ylabel('Jet Discrimination')
ax[0,2].set_xlabel('3-jet Energy Correlation')
ax[0,3].set_ylabel('Jet Discrimination')
ax[0,3].set_xlabel('3-jet Energy Correlation')
ax[0,3].set_title('3-jet Energy vs. Jet Discrimination (High Energy)')

ax[1,0].hist2d(new_dict_3['ee3'], new_dict_3['angularity'], bins=30)
ax[1,1].hist2d(new_dict_4['ee3'], new_dict_4['angularity'], bins=30)
ax[1,0].set_title('3-jet Energy vs. Angularity (Low Energy)')
ax[1,0].set_ylabel('Angularity')
ax[1,0].set_xlabel('3-jet Energy Correlation')
ax[1,1].set_ylabel('Angularity')
ax[1,1].set_xlabel('3-jet Energy Correlation')
ax[1,1].set_title('3-jet Energy vs. Angularity (High Energy)')

ax[1,2].hist2d(new_dict_3['ee3'], new_dict_3['t1'], bins=30)
ax[1,3].hist2d(new_dict_4['ee3'], new_dict_4['t1'], bins=30)
ax[1,2].set_title('3-jet Energy vs. 1-Sub-jettiness (Low Energy)')
ax[1,2].set_ylabel('1-Sub-jettiness')
ax[1,2].set_xlabel('3-jet Energy Correlation')
ax[1,3].set_ylabel('1-Sub-jettiness')
ax[1,3].set_xlabel('3-jet Energy Correlation')
ax[1,3].set_title('3-jet Energy vs. 1-Sub-jettiness (High Energy)')

ax[2,0].hist2d(new_dict_3['ee3'], new_dict_3['t2'], bins=30)
ax[2,1].hist2d(new_dict_4['ee3'], new_dict_4['t2'], bins=30)
ax[2,0].set_title('3-jet Energy vs. 2-Sub-jettiness (Low Energy)')
ax[2,0].set_ylabel('2-Sub-jettiness')
ax[2,0].set_xlabel('3-jet Energy Correlation')
ax[2,1].set_ylabel('2-Sub-jettiness')
ax[2,1].set_xlabel('3-jet Energy Correlation')
ax[2,1].set_title('3-jet Energy vs. 2-Sub-jettiness (High Energy)')

ax[2,2].hist2d(new_dict_3['ee3'], new_dict_3['t3'], bins=30)
ax[2,3].hist2d(new_dict_4['ee3'], new_dict_4['t3'], bins=30)
ax[2,2].set_title('3-jet Energy vs. 3-Sub-jettiness (Low Energy)')
ax[2,2].set_ylabel('3-Sub-jettiness')
ax[2,2].set_xlabel('3-jet Energy Correlation')
ax[2,3].set_ylabel('3-Sub-jettiness')
ax[2,3].set_xlabel('3-jet Energy Correlation')
ax[2,3].set_title('3-jet Energy vs. 3-Sub-jettiness (High Energy)')

ax[3,0].hist2d(new_dict_3['ee3'], new_dict_3['t21'], bins=30)
ax[3,1].hist2d(new_dict_4['ee3'], new_dict_4['t21'], bins=30)
ax[3,0].set_title('3-jet Energy vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[3,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[3,0].set_xlabel('3-jet Energy Correlation')
ax[3,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[3,1].set_xlabel('3-jet Energy Correlation')
ax[3,1].set_title('3-jet Energy vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[3,2].hist2d(new_dict_3['ee3'], new_dict_3['t32'], bins=30)
ax[3,3].hist2d(new_dict_4['ee3'], new_dict_4['t32'], bins=30)
ax[3,2].set_title('3-jet Energy vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[3,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[3,2].set_xlabel('3-jet Energy Correlation')
ax[3,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[3,3].set_xlabel('3-jet Energy Correlation')
ax[3,3].set_title('3-jet Energy vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[0,0].hist2d(new_dict_3['ee3'], new_dict_3['KtDeltaR'], bins=30)
ax[0,1].hist2d(new_dict_4['ee3'], new_dict_4['KtDeltaR'], bins=30)
ax[0,0].set_title('3-jet Energy vs. Angular Distance (Low Energy)')
ax[0,0].set_ylabel('Angular Distance')
ax[0,0].set_xlabel('3-jet Energy Correlation')
ax[0,1].set_ylabel('Angular Distance')
ax[0,1].set_xlabel('3-jet Energy Correlation')
ax[0,1].set_title('3-jet Energy vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# For 3-jet energy correlation, there do not appear to be any variables with which it has positive or negative correlations.

# In[149]:


plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(4, 4)

ax[0,0].hist2d(new_dict_3['d2'], new_dict_3['angularity'], bins=30)
ax[0,1].hist2d(new_dict_4['d2'], new_dict_4['angularity'], bins=30)
ax[0,0].set_title('Jet Discrimination vs. Angularity (Low Energy)')
ax[0,0].set_ylabel('Angularity')
ax[0,0].set_xlabel('Jet Discrimination')
ax[0,1].set_ylabel('Angularity')
ax[0,1].set_xlabel('Jet Discrimination')
ax[0,1].set_title('Jet Discrimination vs. Angularity (High Energy)')

ax[0,2].hist2d(new_dict_3['d2'], new_dict_3['t1'], bins=30)
ax[0,3].hist2d(new_dict_4['d2'], new_dict_4['t1'], bins=30)
ax[0,2].set_title('Jet Discrimination vs. 1-Sub-jettiness (Low Energy)')
ax[0,2].set_ylabel('1-Sub-jettiness')
ax[0,2].set_xlabel('Jet Discrimination')
ax[0,3].set_ylabel('1-Sub-jettiness')
ax[0,3].set_xlabel('Jet Discrimination')
ax[0,3].set_title('Jet Discrimination vs. 1-Sub-jettiness (High Energy)')

ax[1,0].hist2d(new_dict_3['d2'], new_dict_3['t2'], bins=30)
ax[1,1].hist2d(new_dict_4['d2'], new_dict_4['t2'], bins=30)
ax[1,0].set_title('Jet Discrimination vs. 2-Sub-jettiness (Low Energy)')
ax[1,0].set_ylabel('2-Sub-jettiness')
ax[1,0].set_xlabel('Jet Discrimination')
ax[1,1].set_ylabel('2-Sub-jettiness')
ax[1,1].set_xlabel('Jet Discrimination')
ax[1,1].set_title('Jet Discrimination vs. 2-Sub-jettiness (High Energy)')

ax[1,2].hist2d(new_dict_3['d2'], new_dict_3['t3'], bins=30)
ax[1,3].hist2d(new_dict_4['d2'], new_dict_4['t3'], bins=30)
ax[1,2].set_title('Jet Discrimination vs. 3-Sub-jettiness (Low Energy)')
ax[1,2].set_ylabel('3-Sub-jettiness')
ax[1,2].set_xlabel('Jet Discrimination')
ax[1,3].set_ylabel('3-Sub-jettiness')
ax[1,3].set_xlabel('Jet Discrimination')
ax[1,3].set_title('Jet Discrimination vs. 3-Sub-jettiness (High Energy)')

ax[2,0].hist2d(new_dict_3['d2'], new_dict_3['t21'], bins=30)
ax[2,1].hist2d(new_dict_4['d2'], new_dict_4['t21'], bins=30)
ax[2,0].set_title('Jet Discrimination vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[2,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[2,0].set_xlabel('Jet Discrimination')
ax[2,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[2,1].set_xlabel('Jet Discrimination')
ax[2,1].set_title('Jet Discrimination vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[2,2].hist2d(new_dict_3['d2'], new_dict_3['t32'], bins=30)
ax[2,3].hist2d(new_dict_4['d2'], new_dict_4['t32'], bins=30)
ax[2,2].set_title('Jet Discrimination vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[2,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[2,2].set_xlabel('Jet Discrimination')
ax[2,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[2,3].set_xlabel('Jet Discrimination')
ax[2,3].set_title('Jet Discrimination vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[3,0].hist2d(new_dict_3['d2'], new_dict_3['KtDeltaR'], bins=30)
ax[3,1].hist2d(new_dict_4['d2'], new_dict_4['KtDeltaR'], bins=30)
ax[3,0].set_title('Jet Discrimination vs. Angular Distance (Low Energy)')
ax[3,0].set_ylabel('Angular Distance')
ax[3,0].set_xlabel('Jet Discrimination')
ax[3,1].set_ylabel('Angular Distance')
ax[3,1].set_xlabel('Jet Discrimination')
ax[3,1].set_title('Jet Discrimination vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# For jet discrimination, it is a little tricky to tell, but there may be positive correlation between it and the sub-jettiness variables. There is also possibly a negative correlation occuring between it and angular distance, especially for low energies.

# In[151]:


plt.rcParams["figure.figsize"] = (20,12)
fig, ax = plt.subplots(3, 4)

ax[0,2].hist2d(new_dict_3['angularity'], new_dict_3['t1'], bins=30)
ax[0,3].hist2d(new_dict_4['angularity'], new_dict_4['t1'], bins=30)
ax[0,2].set_title('Angularity vs. 1-Sub-jettiness (Low Energy)')
ax[0,2].set_ylabel('1-Sub-jettiness')
ax[0,2].set_xlabel('Angularity')
ax[0,3].set_ylabel('1-Sub-jettiness')
ax[0,3].set_xlabel('Angularity')
ax[0,3].set_title('Angularity vs. 1-Sub-jettiness (High Energy)')

ax[1,0].hist2d(new_dict_3['angularity'], new_dict_3['t2'], bins=30)
ax[1,1].hist2d(new_dict_4['angularity'], new_dict_4['t2'], bins=30)
ax[1,0].set_title('Angularity vs. 2-Sub-jettiness (Low Energy)')
ax[1,0].set_ylabel('2-Sub-jettiness')
ax[1,0].set_xlabel('Angularity')
ax[1,1].set_ylabel('2-Sub-jettiness')
ax[1,1].set_xlabel('Angularity')
ax[1,1].set_title('Angularity vs. 2-Sub-jettiness (High Energy)')

ax[1,2].hist2d(new_dict_3['angularity'], new_dict_3['t3'], bins=30)
ax[1,3].hist2d(new_dict_4['angularity'], new_dict_4['t3'], bins=30)
ax[1,2].set_title('Angularity vs. 3-Sub-jettiness (Low Energy)')
ax[1,2].set_ylabel('3-Sub-jettiness')
ax[1,2].set_xlabel('Angularity')
ax[1,3].set_ylabel('3-Sub-jettiness')
ax[1,3].set_xlabel('Angularity')
ax[1,3].set_title('Angularity vs. 3-Sub-jettiness (High Energy)')

ax[2,0].hist2d(new_dict_3['angularity'], new_dict_3['t21'], bins=30)
ax[2,1].hist2d(new_dict_4['angularity'], new_dict_4['t21'], bins=30)
ax[2,0].set_title('Angularity vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[2,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[2,0].set_xlabel('Angularity')
ax[2,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[2,1].set_xlabel('Angularity')
ax[2,1].set_title('Angularity vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[2,2].hist2d(new_dict_3['angularity'], new_dict_3['t32'], bins=30)
ax[2,3].hist2d(new_dict_4['angularity'], new_dict_4['t32'], bins=30)
ax[2,2].set_title('Angularity vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[2,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[2,2].set_xlabel('Angularity')
ax[2,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[2,3].set_xlabel('Angularity')
ax[2,3].set_title('Angularity vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[0,0].hist2d(new_dict_3['angularity'], new_dict_3['KtDeltaR'], bins=30)
ax[0,1].hist2d(new_dict_4['angularity'], new_dict_4['KtDeltaR'], bins=30)
ax[0,0].set_title('Angularity vs. Angular Distance (Low Energy)')
ax[0,0].set_ylabel('Angular Distance')
ax[0,0].set_xlabel('Angularity')
ax[0,1].set_ylabel('Angular Distance')
ax[0,1].set_xlabel('Angularity')
ax[0,1].set_title('Angularity vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# As far as we are able to see from the plots above, there doesn't appear to be any correlation between angularity and these variables.

# In[164]:


plt.rcParams["figure.figsize"] = (20,12)
fig, ax = plt.subplots(3, 4)

ax[0,0].hist2d(new_dict_3['t1'], new_dict_3['t2'], bins=30)
ax[0,1].hist2d(new_dict_4['t1'], new_dict_4['t2'], bins=30)
ax[0,0].set_title('1-Sub-jettiness vs. 2-Sub-jettiness (Low Energy)')
ax[0,0].set_ylabel('2-Sub-jettiness')
ax[0,0].set_xlabel('1-Sub-jettiness')
ax[0,1].set_ylabel('2-Sub-jettiness')
ax[0,1].set_xlabel('1-Sub-jettiness')
ax[0,1].set_title('1-Sub-jettiness vs. 2-Sub-jettiness (High Energy)')

ax[0,2].hist2d(new_dict_3['t1'], new_dict_3['t3'], bins=30)
ax[0,3].hist2d(new_dict_4['t1'], new_dict_4['t3'], bins=30)
ax[0,2].set_title('1-Sub-jettiness vs. 3-Sub-jettiness (Low Energy)')
ax[0,2].set_ylabel('3-Sub-jettiness')
ax[0,2].set_xlabel('1-Sub-jettiness')
ax[0,3].set_ylabel('3-Sub-jettiness')
ax[0,3].set_xlabel('1-Sub-jettiness')
ax[0,3].set_title('1-Sub-jettiness vs. 3-Sub-jettiness (High Energy)')

ax[1,0].hist2d(new_dict_3['t1'], new_dict_3['t21'], bins=30)
ax[1,1].hist2d(new_dict_4['t1'], new_dict_4['t21'], bins=30)
ax[1,0].set_title('1-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[1,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[1,0].set_xlabel('1-Sub-jettiness')
ax[1,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[1,1].set_xlabel('1-Sub-jettiness')
ax[1,1].set_title('1-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[1,2].hist2d(new_dict_3['t1'], new_dict_3['t32'], bins=30)
ax[1,3].hist2d(new_dict_4['t1'], new_dict_4['t32'], bins=30)
ax[1,2].set_title('1-Sub-jettiness vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[1,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[1,2].set_xlabel('1-Sub-jettiness')
ax[1,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[1,3].set_xlabel('1-Sub-jettiness')
ax[1,3].set_title('1-Sub-jettiness vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[2,0].hist2d(new_dict_3['t1'], new_dict_3['KtDeltaR'], bins=30)
ax[2,1].hist2d(new_dict_4['t1'], new_dict_4['KtDeltaR'], bins=30)
ax[2,0].set_title('1-Sub-jettiness vs. Angular Distance (Low Energy)')
ax[2,0].set_ylabel('Angular Distance')
ax[2,0].set_xlabel('1-Sub-jettiness')
ax[2,1].set_ylabel('Angular Distance')
ax[2,1].set_xlabel('1-Sub-jettiness')
ax[2,1].set_title('1-Sub-jettiness vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# These plots show a positive correlation between 1-sub-jettiness and 2-, 3-, and 21-sub-jettiness. Though this isn't particularly helpful, it can easily be imagined that if a narrow jet cone is observed, all sub-jettiness values will be smaller, whereas a broader cone could return larger sub-jettiness values for all if the particle density were fairly uniform. There is also what appears to be a negative correlation with angular distance, which makes sense, since the angular distance is a measure between 2 subjets.

# In[154]:


plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(2, 4)

ax[0,0].hist2d(new_dict_3['t2'], new_dict_3['t3'], bins=30)
ax[0,1].hist2d(new_dict_4['t2'], new_dict_4['t3'], bins=30)
ax[0,0].set_title('2-Sub-jettiness vs. 3-Sub-jettiness (Low Energy)')
ax[0,0].set_ylabel('3-Sub-jettiness')
ax[0,0].set_xlabel('2-Sub-jettiness')
ax[0,1].set_ylabel('3-Sub-jettiness')
ax[0,1].set_xlabel('2-Sub-jettiness')
ax[0,1].set_title('2-Sub-jettiness vs. 3-Sub-jettiness (High Energy)')

ax[0,2].hist2d(new_dict_3['t2'], new_dict_3['t21'], bins=30)
ax[0,3].hist2d(new_dict_4['t2'], new_dict_4['t21'], bins=30)
ax[0,2].set_title('2-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[0,2].set_ylabel('2-1 Sub-jettiness Ratio')
ax[0,2].set_xlabel('2-Sub-jettiness')
ax[0,3].set_ylabel('2-1 Sub-jettiness Ratio')
ax[0,3].set_xlabel('2-Sub-jettiness')
ax[0,3].set_title('2-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[1,0].hist2d(new_dict_3['t2'], new_dict_3['t32'], bins=30)
ax[1,1].hist2d(new_dict_4['t2'], new_dict_4['t32'], bins=30)
ax[1,0].set_title('2-Sub-jettiness vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[1,0].set_ylabel('3-2 Sub-jettiness Ratio')
ax[1,0].set_xlabel('2-Sub-jettiness')
ax[1,1].set_ylabel('3-2 Sub-jettiness Ratio')
ax[1,1].set_xlabel('2-Sub-jettiness')
ax[1,1].set_title('2-Sub-jettiness vs. 3-2 Sub-jettiness Ratio  (High Energy)')

ax[1,2].hist2d(new_dict_3['t2'], new_dict_3['KtDeltaR'], bins=30)
ax[1,3].hist2d(new_dict_4['t2'], new_dict_4['KtDeltaR'], bins=30)
ax[1,2].set_title('2-Sub-jettiness vs. Angular Distance (Low Energy)')
ax[1,2].set_ylabel('Angular Distance')
ax[1,2].set_xlabel('2-Sub-jettiness')
ax[1,3].set_ylabel('Angular Distance')
ax[1,3].set_xlabel('2-Sub-jettiness')
ax[1,3].set_title('2-Sub-jettiness vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# As before, we see positive correlation between the sub-jettiness variables and negative correlation with angular distance.

# In[155]:


plt.rcParams["figure.figsize"] = (20,8)
fig, ax = plt.subplots(2, 4)

ax[0,0].hist2d(new_dict_3['t3'], new_dict_3['t21'], bins=30)
ax[0,1].hist2d(new_dict_4['t3'], new_dict_4['t21'], bins=30)
ax[0,0].set_title('3-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (Low Energy)')
ax[0,0].set_ylabel('2-1 Sub-jettiness Ratio')
ax[0,0].set_xlabel('3-Sub-jettiness')
ax[0,1].set_ylabel('2-1 Sub-jettiness Ratio')
ax[0,1].set_xlabel('3-Sub-jettiness')
ax[0,1].set_title('3-Sub-jettiness vs. 2-1 Sub-jettiness Ratio (High Energy)')

ax[0,2].hist2d(new_dict_3['t3'], new_dict_3['t32'], bins=30)
ax[0,3].hist2d(new_dict_4['t3'], new_dict_4['t32'], bins=30)
ax[0,2].set_title('3-Sub-jettiness vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[0,2].set_ylabel('3-2 Sub-jettiness Ratio')
ax[0,2].set_xlabel('3-Sub-jettiness')
ax[0,3].set_ylabel('3-2 Sub-jettiness Ratio')
ax[0,3].set_xlabel('3-Sub-jettiness')
ax[0,3].set_title('3-Sub-jettiness vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[1,0].hist2d(new_dict_3['t3'], new_dict_3['KtDeltaR'], bins=30)
ax[1,1].hist2d(new_dict_4['t3'], new_dict_4['KtDeltaR'], bins=30)
ax[1,0].set_title('3-Sub-jettiness vs. Angular Distance (Low Energy)')
ax[1,0].set_ylabel('Angular Distance')
ax[1,0].set_xlabel('3-Sub-jettiness')
ax[1,1].set_ylabel('Angular Distance')
ax[1,1].set_xlabel('3-Sub-jettiness')
ax[1,1].set_title('3-Sub-jettiness vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# These plots show positive correlation with the ratios (which we expect) as well as negative with angular distance (which we also expect).

# In[158]:


plt.rcParams["figure.figsize"] = (20,3)
fig, ax = plt.subplots(1, 4)

ax[0].hist2d(new_dict_3['t21'], new_dict_3['t32'], bins=30)
ax[1].hist2d(new_dict_4['t21'], new_dict_4['t32'], bins=30)
ax[0].set_title('2-1 Sub-jettiness Ratio vs. 3-2 Sub-jettiness Ratio (Low Energy)')
ax[0].set_ylabel('3-2 Sub-jettiness Ratio')
ax[0].set_xlabel('2-1 Sub-jettiness Ratio')
ax[1].set_ylabel('3-2 Sub-jettiness Ratio')
ax[1].set_xlabel('2-1 Sub-jettiness Ratio')
ax[1].set_title('2-1 Sub-jettiness Ratio vs. 3-2 Sub-jettiness Ratio (High Energy)')

ax[2].hist2d(new_dict_3['t21'], new_dict_3['KtDeltaR'], bins=30)
ax[3].hist2d(new_dict_4['t21'], new_dict_4['KtDeltaR'], bins=30)
ax[2].set_title('2-1 Sub-jettiness Ratio vs. Angular Distance (Low Energy)')
ax[2].set_ylabel('Angular Distance')
ax[2].set_xlabel('2-1 Sub-jettiness Ratio')
ax[3].set_ylabel('Angular Distance')
ax[3].set_xlabel('2-1 Sub-jettiness Ratio')
ax[3].set_title('2-1 Sub-jettiness Ratio vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# This shows possible negative correlation between the 2-1-sub-jettiness ratio and angular distance, but it is not particularly strong.

# In[14]:


plt.rcParams["figure.figsize"] = (20,8)
fig, ax = plt.subplots(2,4)

ax[0,0].hist2d(new_dict_3['t32'], new_dict_3['KtDeltaR'], bins=30)
ax[0,1].hist2d(new_dict_4['t32'], new_dict_4['KtDeltaR'], bins=30)
ax[0,0].set_title('3-2 Sub-jettiness Ratio vs. Angular Distance (Low Energy)')
ax[0,0].set_ylabel('Angular Distance')
ax[0,0].set_xlabel('3-2 Sub-jettiness Ratio')
ax[0,1].set_ylabel('Angular Distance')
ax[0,1].set_xlabel('3-2 Sub-jettiness Ratio')
ax[0,1].set_title('3-2 Sub-jettiness Ratio vs. Angular Distance (High Energy)')

ax[0,2].hist2d(new_dict_3['eta'], new_dict_3['KtDeltaR'], bins=30)
ax[0,3].hist2d(new_dict_4['eta'], new_dict_4['KtDeltaR'], bins=30)
ax[0,2].set_title('Pseudorapidity vs. Angular Distance (Low Energy)')
ax[0,2].set_ylabel('Angular Distance')
ax[0,2].set_xlabel('Pseudorapidity')
ax[0,3].set_ylabel('Angular Distance')
ax[0,3].set_xlabel('Pseudorapidity')
ax[0,3].set_title('Pseudorapidity vs. Angular Distance (High Energy)')

ax[1,0].hist2d(new_dict_3['phi'], new_dict_3['KtDeltaR'], bins=30)
ax[1,1].hist2d(new_dict_4['phi'], new_dict_4['KtDeltaR'], bins=30)
ax[1,0].set_title('Radiation Angle vs. Angular Distance (Low Energy)')
ax[1,0].set_ylabel('Angular Distance')
ax[1,0].set_xlabel('Radiation Angle')
ax[1,1].set_ylabel('Angular Distance')
ax[1,1].set_xlabel('Radiation Angle')
ax[1,1].set_title('Radiation Angle vs. Angular Distance (High Energy)')

plt.tick_params(labelsize = 12)
plt.subplots_adjust(left=0, bottom=0.05, right=0.9, top=0.9, wspace=None, hspace=0.35)
plt.show()


# None of the above variables show correlation with angular distance.

# In order to normalize these features, we need to calculate four different normalization constants (one for each distribution for both energy levels) by using the ratio of what we expect to measure and what we have.

# In[15]:


NH_low = 100/100000
NH_high = 50/100000
NQ_low = 20000/100000
NQ_high = 2000/100000


# Now we need to normalize each dataset.

# In[17]:


HL_norm = NH_low * new_dict_1
HH_norm = NH_high * new_dict_2
QL_norm = NQ_low * new_dict_3
QH_norm = NQ_high * new_dict_4


# In[19]:


plt.rcParams["figure.figsize"] = (20,10)

fig, ax = plt.subplots(2, 2)
ax[0,0].hist(QL_norm['mass'], 30, density=True, label='Higgs distribution')
ax[0,0].hist(HL_norm['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[0,1].hist(QH_norm['mass'], 30, density=True, label='Higgs distribution')
ax[0,1].hist(HH_norm['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[1,0].hist(QL_norm['mass'], 30, density=True, label='Higgs distribution')
ax[1,0].hist(HL_norm['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[1,1].hist(QH_norm['mass'], 30, density=True, label='Higgs distribution')
ax[1,1].hist(HH_norm['mass'], 30, density=True, fill=False, label='QCD distribution')
ax[0,0].set_title('Normalized Low Energy Mass Distribution')
ax[0,1].set_title('Normalized High Energy Mass Distribution')
ax[1,0].set_title('Normalized Low Energy Mass Distribution (Log)')
ax[1,1].set_title('Normalized High Energy Mass Distribution (Log)')
ax[0,0].set_ylabel('Probability')
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('Mass')
ax[1,1].set_xlabel('Mass')
ax[1,0].semilogy()
ax[1,1].semilogy()
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()


# The plots of the newly normalized masses show quite a large difference between the distributions. This shows that by normalizing our variables, we can increase the discovery sensitivity per variable. We can further increase this sensitivity for each variable by excluding values over which there is no disrmination possible between the two distributions. By paying attention to only those ranges of values in which we will be able to tell the difference between the two, the sensitivity is increased.

# In[ ]:




