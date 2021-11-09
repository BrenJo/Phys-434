#!/usr/bin/env python
# coding: utf-8

# ## Lab 4: Working with 'real' data
# 
# ### Introduction
# In this lab we are going to work on how to estimate the background from 'real' data. Real is in air quotes because the data is actually from simplified simulations to make the problems manageable in a single lab. But the data will have some features that resemble that of real data sets.

# In[416]:


#%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py

plt.rcParams["figure.figsize"] = (10,5)


# We are working with datasets that have been produced by simulations in order to include some features that may be present in a real-life dataset which would cause them to have a distribution that differs from the different models we have been looking at. In the first part, we will look at dataset simulated to resemble some things we might see if we were measuring gamma rays with a low Earth orbit satellite. The second part will involve analyzing several images worth of simulated data in order to identify signals that look like faint stars.

# In[417]:


hf = h5py.File('gammaray_lab4.h5', 'r')


# In[418]:


hf.keys()


# In[419]:


data = np.array(hf.get('data'))
data[:,0]


# In[420]:


hf.close()


# ### Problem 1
# In this problem we are looking at the data from a gamma-ray satellite orbiting in low Earth orbit. It takes a reading of the number of particles detected every 100 milliseconds, and is in an approximately 90 minute orbit. While it is looking for gamma-ray bursts, virtually all of the particles detected are background cosmic rays.
# 
# 1) Down load the data from the course website (gammaray_lab4.h5), and import it into your working environment. The data has 4 columns and more than 25 million rows. The columns are time (in gps seconds), Solar phase (deg) showing the position of the sun relative to the orbit, Earth longitude (deg) giving the position of the spacecraft relative to the ground, and particle counts. Make a few plots, generally exploring your data and making sure you understand it. Give a high level description of the data features you see. Specifically comment on whether you see signal contamination in your data, and how you plan to build a background pdf().
# 
# 2) The background is not consistent across the dataset. Find and describe as accurately as you can how the background changes.
# 
# 3) Create a model for the background that includes time dependence, and explicitly compare your model to the data. How good is your model of the background?
# 
# 4) Because the background varies, your discovery sensitivity threshold (how many particles you would need to see) also varies. What is the '5-sigma' threshold for a 100 millisecond GRB at different times?
# 
# Optional: while this is simulated data, it is based on a real effect seen by low Earth orbit satellites. Can you identify the cause of the variable background and propose a physical model?

# Our first problem is set up to study a hypothetical dataset produced by gamma-ray detection from a satellite in low-Earth orbit which travels at a rate of approximately 90 minutes per orbit and records measurements of the number of particles detected at rate of once every 100 milliseconds. Presumably, we are searching for gamma ray bursts, but the real object of this exercize is to analyze the background distribution in order to find and describe the features that cause it to differ from a standard distribution.
# 
# The four measured variables are time in GPS seconds, solar phase in degrees, Earth longitude in degrees, and counts of gamma ray detection.

# In[421]:


t = data[0,:]
sol = data[1,:]
lon = data[2,:]
n = data[3,:]
print(n.size)
print(t.size)
print(sol.size)
print(lon.size)
print(np.max(n))
print(n.size*100/(1000*60*60*24))


# As we can see from above, we have a total of 25,920,001 measurements, with our maximum reading at 30 particles. Using the rate of measurement and the number of measurements, we can also see that the data was collected over the course of a 30-day month. Therefore, before we get started analyzing the data, we can say with relative confidence that we are unlikely to see strong effects due to changes in the axial orientation of the Earth with respect to the Sun.
# 
# Since we are dealing with dicrete events, we will use Poisson distributions as models. First, we can find the average count over the entire dataset, then plot the count density distribution of our data and compare it to a Poisson distribution centered around the average.

# In[422]:


x = np.arange(0,30)
mean = np.mean(n)
exp = stats.poisson.pmf(x, mean, loc=0)

fig, ax = plt.subplots(1, 2)
ax[0].hist(n, 30, density=True, label='Count distribution')
ax[0].step(x, exp, label='Average Poisson distribution')
ax[0].set_title('Count vs. Poisson Distributions')
ax[0].set_ylabel('Probability')
ax[0].set_xlabel('Counts')
ax[1].hist(n, 30, density=True, label='Count distribution')
ax[1].step(x, exp, label='Average Poisson distribution')
ax[1].semilogy()
#ax[1].set_ylabel('Probability')
ax[1].set_xlabel('Counts')
ax[1].set_title('Count vs. Poisson Distributions (log)')
plt.tick_params(labelsize = 12)
plt.xlim([0,35])
plt.legend()
plt.show()
fig.tight_layout()


# As we can see from the plots above, there is a slight offset between the actual background and the background we expect, as well as an effect that has produced a larger tail on the right-hand side for the actual distribution. This suggests that something is increasing the number of counts measured from what we would expect to see.
# 
# In order to see what might be causing this difference, we can use the metadata in our dataset (the solar phase and the Earth longitude) to check and see if that difference is related to another measured circumstance. 
# 
# First, we will analyze the solar phase with respect to time and the count density with respect to solar phase using two-dimensional density distributions. The first ana;ysis will tell us whether or not the period of the solar phase is consistent, and the second will tell us whether or not the solar phase affects the distribution of gamma rays we detect.

# In[423]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(sol, t, bins=50)
f = ax[1].hist2d(sol, n, bins=30)
ax[0].set_title('Solar Phase vs. Time')
ax[0].set_ylabel('Time')
ax[0].set_xlabel('Solar Phase')
ax[1].set_ylabel('Counts')
ax[1].set_xlabel('Solar Phase')
ax[1].set_title('Count Distribution According to Solar Phase')
plt.tick_params(labelsize = 12)
plt.colorbar(f[3], ax=ax)
plt.show()
#fig.tight_layout()


# Looking at our plots above, we can see from the first that although there is some minor variation of the solar phase over time, the count distribution is still relatively consistent. From the second plot, we can see that the background measured is very consistent across solar phases, and it is unlikely that solar phase has any effect on the difference in our distribution from what we would expect. Likewise, we have further evidence of a right-hand tail, as the gradient between 0 and the maximum density is steeper than that between the maximum density and 15 or 20.
# 
# Since we can rule out solar phase as the culprit for our skewed distribution, we can move on to the relationships of longitude over time and count density with respect to Earth longitude.

# In[424]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(lon, t, bins=50)
g = ax[1].hist2d(lon, n, bins=30)
ax[0].set_title('Earth Longitude vs. Time')
ax[0].set_ylabel('Time')
ax[0].set_xlabel('Longitude')
ax[1].set_ylabel('Counts')
ax[1].set_xlabel('Longitude')
ax[1].set_title('Count Distribution According to Longitude')
plt.tick_params(labelsize = 10)
plt.colorbar(g[3], ax=ax)
plt.show()


# These plots show that there is definitely some count density dependence on longitude. While the first plot shows that the period of longitude over time is pretty consistent, the second plot shows that there is a discrepancy between the count densities that occur from longitudes of approximately 180 degrees to 310 degrees and those that occur elsewhere, with the former being relatively steady and the rest showing a significant increase in the densities at higher counts that follows a fairly dramatic curve. It also appears that the distribution produced in the mid-range is significantly less skewed than that produced in the lower and higher ranges. Therefore, we can say that there is a dependence of the count density distribution on the orbit's positional relation to the Earth.
# 
# In order to further analyze this relationship, we can calculate the average counts based on what longitude they occur in correspondence with.

# In[425]:


hist, xedges, yedges, bins = plt.hist2d(lon, n, bins = 30)
xedges


# Looking closer, we can see that there is an ubrupt increase in the width and average of the distribution somewhere around 320 degrees. We know this is a periodic relationship, so we can expect this increase to occur repeatedly with every turning over of the longitudes that occurs as the satellite orbits the Earth multiple times.

# It is a good idea to examine the contrast of the data distribution with a function that is the sum of the expected sinusoidal distributions of the periodic relationships the satellite experiences with regards to the Sun and Earth.

# In[426]:


print(hist[0])
print(lon[40])


# In[427]:


print(lon[0],lon[-1])
print(sol[0],sol[-1])
print(sol[40])


# In[428]:


g = (n/4)*np.cos((2*np.pi/360)*lon*t) + (n/4)*np.sin((2*np.pi/360)*sol*t) + mean
plt.plot(t[0:800], g[0:800], 'b', label='Periodic Function')
plt.plot(t[0:800], n[0:800], 'r', label='Counts over time')
plt.title('Counts Over Time Compared with Periodic Function')
plt.xlabel('Counts')
plt.ylabel('Time')
plt.legend()
plt.show()


# This isn't an excellent approximation, but it does show some periodicity in the data. In order to see if this changes, we can plot the same quantities for different sections of the dataset.

# In[429]:


plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(2, 2)
ax[0,0].plot(t[0:800], g[0:800], 'b')
ax[0,0].plot(t[0:800], n[0:800], 'r')
ax[0,1].plot(t[6486000:6486800], g[6486000:6486800], 'b')
ax[0,1].plot(t[6486000:6486800], n[6486000:6486800], 'r')
ax[1,0].plot(t[19446000:19446800], g[19446000:19446800], 'b')
ax[1,0].plot(t[19446000:19446800], n[19446000:19446800], 'r')
ax[1,1].plot(t[25919200:25920000], g[25919200:25920000], 'b')
ax[1,1].plot(t[25919200:25920000], n[25919200:25920000], 'r')
ax[0,0].set_title('Counts Over Time Compared with Periodic Function')
ax[0,1].set_title('Counts Over Time Compared with Periodic Function')
ax[1,0].set_title('Counts Over Time Compared with Periodic Function')
ax[1,1].set_title('Counts Over Time Compared with Periodic Function')
ax[0,0].set_xlabel('Counts')
ax[0,0].set_ylabel('Time')
ax[0,1].set_xlabel('Counts')
ax[0,1].set_ylabel('Time')
ax[1,0].set_xlabel('Counts')
ax[1,0].set_ylabel('Time')
ax[1,1].set_xlabel('Counts')
ax[1,1].set_ylabel('Time')
plt.show()
fig.tight_layout()


# While these plots do show the periodicity of the count density, it is not a very good match for the data distribution, though those plots taken from the middle ranges of the data set look better that those at the extremeties.
# 
# Instead, let's examine the average counts according to the longitude they correspond with by cutting the count and longitude data into 30 bins and taking the average count in each bin, and then plotting those average values with respect to longitude.

# In[430]:


plt.rcParams["figure.figsize"] = (10,5)
avg = np.empty([1])

for i in hist:
    counts = 0
    for j in range(len(i)):
            counts = counts + j*i[j]
    avg = np.append(avg, counts/(len(i)*7e4))

plt.plot(xedges[1:], avg[1:], 'b', label='Average')
plt.xlabel('Longitude')
plt.ylabel('Average Counts')
plt.legend()
plt.title('Average Counts According to Longitude')

print(hist.shape)
print(np.min(avg))
print(np.where(avg == np.min(avg)))
print(np.where(avg <= 2.4))
print(xedges[26])
print(xedges[0])
print(np.max(avg))
print(np.where(avg == np.max(avg)))
print(xedges[28]-xedges[26])
print(avg)


# This plot shows that there is a definite relationship occuring that looks similar to an exponential decay. It also shows that the increase is abrupt and fairly steep. This could be a trick of the bin-size, or it could be that the count average really does jump from around 2.4 to 4 at approximately 312 degrees. 
# 
# The following plot shows an exponential decay function with parameters that have been chosen based on looking at several different values and choosing those that fit the decay of the average best.

# In[431]:


x = np.linspace(0,360)
mu = xedges[28]-xedges[26]
#plt.plot(xedges[26:]-xedges[26], avg[25:], 'b', label='Average')
plt.plot(xedges[27:]-xedges[26], avg[27:], 'b', label='Average')
plt.plot(xedges[1:26]+xedges[-1]-xedges[26], avg[1:26], 'b')
#plt.plot(x, 0.18*np.log(0.55*x+0.01)+3.75, 'r')
plt.plot(x, 2.6*np.exp(-0.0114*x)+2.35, 'k', label='Exponential Decay')
plt.xlabel('Longitude')
plt.ylabel('Average Counts')
plt.legend()
plt.title('Average Counts According to Longitude')
plt.show()


# This shows that there is a contribution to the overall data distribution that corresponds to an exponential decay.
# 
# Using the averages, we can also see how the the background modeled as a Poisson function changes with longitude.

# In[432]:


print(avg[3], avg[9], avg[15], avg[26], avg[30], avg[27])
print(xedges[3], xedges[9], xedges[15], xedges[26], xedges[30])


# Looking at the averages printed above, we can see that they differ. However, because a Poisson function needs discrete values, if we were to round them to the nearest integer we would be unable to see the difference. In order to make that difference visible, we can multiply each average by a factor of 10 before rounding; graphing distributions based on these new values does not give us a quantitative comparison, but it does show the relationship between how the distributions change with respect to each other according to longitudinal position.

# In[433]:


x = np.arange(0,300)
#plt.hist(n, 30, density=True)
#plt.step(x, exp, 'm', label='Average distribution over 360' r'$\deg$')
b1 = stats.poisson.pmf(x, np.around(avg[3]*10), loc=0)
b2 = stats.poisson.pmf(x, np.around(avg[9]*10), loc=0)
b3 = stats.poisson.pmf(x, np.around(avg[15]*10), loc=0)
b4 = stats.poisson.pmf(x, np.around(avg[26]*10), loc=0)
b5 = stats.poisson.pmf(x, np.around(avg[-1]*10), loc=0)
plt.xlim([0,80])
plt.step(x, b1, 'b', label='36'r'$\degree$')
plt.step(x, b2, 'g', label='108'r'$\degree$')
plt.step(x, b3, 'y', label='180'r'$\degree$')
plt.step(x, b4, 'orange', label='312'r'$\degree$')
plt.step(x, b5, 'r', label='360'r'$\degree$')
plt.ylabel('Probability')
plt.xlabel('Counts')
plt.title('Background Distributions at Different Longitudinal Positions')
plt.legend()
plt.show()


# The plot above shows how both the background distribution width and average changes according to the longitude at which we consider it. Those corresponding to middle longitudes are narrower, and symmetrical about a lower value, whereas those corresponding to the lowest and highest longitudes are wider and distributed around a higher measurement count.
# 
# Using these distributions, we can determine the number of counts necessary for a 5 sigma signal measurement when the satellite is positioned relative to these different Earth longitudes.

# In[434]:


P = stats.norm.sf(5, loc=0, scale=1)

s1 = stats.poisson.isf(P, np.around(avg[3]*10), loc=0)
s2 = stats.poisson.isf(P, np.around(avg[9]*10), loc=0)
s3 = stats.poisson.isf(P, np.around(avg[15]*10), loc=0)
s4 = stats.poisson.isf(P, np.around(avg[26]*10), loc=0)
s5 = stats.poisson.isf(P, np.around(avg[-1]*10), loc=0)

print('Longitude | 5-sigma signal')
print('--------------------------')
print('   36     |   ', s1)
print('   108    |   ', s2)
print('   180    |   ', s3)
print('   312    |   ', s4)
print('   360    |   ', s5)


# As we can see from the values presented above, we need a much higher count for a measurement to be considered a signal when we are at the highest and lowest longitudinal positions.
# 
# One reason this may occur is if the latitude of the orbit were not constant. Because it is in a low-Earth orbit, the measurements the satellite makes are likely to be affected by the Earth's magnetic field, which would reduce the counts measured at lower latitudes by shielding it from cosmic particles that could interact with other matter and produce gamma rays. As the orbit tilts closer to the poles, however, there will be more cosmic particles that get through, producing more gamma rays to be measured. Though this would seem to indicate two dramatic count increases reather than one (as the satellite goes near both the south and north poles), if the season were such that one of the poles was at a tilt receiving maximum solar radiation (such as the north pole at summer solstice), the increase could be as we saw it above.

# ### Problem 2
# In this problem we are going to look at a stack of telescope images (again simulated). We have 10 images, but you and your lab partner will be looking for different signals. One of you will be looking for the faintest stars, while the other will be looking for a transient (something like a super novae that only appears in one image). Flip a coin to determine which of you is pursuing which question.
# 
# 1) Dowload the data from images.h5. This is a stack of 10 square images, each 200 pixels on a side.
# 
# 2) Explore the data. Is there signal contamination? Is the background time dependent? Is it consistent spatially? Develop a plan to calculate your background pdf().
# 
# 3) Using your background distribution, hunt for your signal (either faint stars, or a transient). Describe what you find.
# 
# 4) You and your lab partner had different pdf(), but were using the same data. Explore why this is.

# In[435]:


im = h5py.File('images.h5', 'r')


# In[436]:


im.keys()


# In[437]:


image1 = np.array(im.get('image1'))
imstack = np.array(im.get('imagestack'))


# In[438]:


im.close()


# In[439]:


image1.shape


# In[440]:


imstack.shape


# For this problem, we will be looking for faint stars in the given images by using the image data to determine the background distribution, and then finding the threshold at which we can confidently consider a measurement to be a signal indicating a faint star.
# 
# In order to find the background, first we need to view the images to see if there are any obvious anomalies and signal contamination. Below is first the original image, followed by each of the individuak ten images taken over time.

# In[441]:


fig, ax = plt.subplots(1, 1)
plt.imshow(image1)
plt.tick_params(labelsize = 12)
plt.show()


# In[442]:


plt.rcParams["figure.figsize"] = (20,10)

fig, ax = plt.subplots(2, 5)
ax[0,0].imshow(imstack[:,:,0])
ax[0,1].imshow(imstack[:,:,1])
ax[0,2].imshow(imstack[:,:,2])
ax[0,3].imshow(imstack[:,:,3])
ax[0,4].imshow(imstack[:,:,4])
ax[1,0].imshow(imstack[:,:,5])
ax[1,1].imshow(imstack[:,:,6])
ax[1,2].imshow(imstack[:,:,7])
ax[1,3].imshow(imstack[:,:,8])
ax[1,4].imshow(imstack[:,:,9])
plt.tick_params(labelsize = 14)
plt.show()
fig.tight_layout()


# From viewing the different images above, there does not appear to be any contamination present. Each image looks relatively similar, and they do not appear to have any odd bright patches, streaks, or gradations.
# 
# Since the ten images above were taken at different times, it is important to check to see if there is any variance between images that could be time dependent. We can do that by examining the distributions of each image with respect to the others.

# In[443]:


print(np.max(imstack))


# In[444]:


images = np.empty([10,40000])
for i in range(0,10):
    images[i,:] = imstack[:,:,i].flatten()
print(images.shape)
print(np.max(images))


# In[445]:


#images = np.empty([1,40200])

#for m in range(0,10):
#    new_image = np.empty([200])
#    for n in range(0,200):
#        for p in range(0,200):
#            new_image = np.append(new_image, [imstack[n,p,m]], axis=0)
#    images = np.append(images, [new_image], axis=0)
#print(images.shape)
#print(images)
#print(np.max(images))


# In[446]:


x = np.arange(-5, 10, 1000)
#plt.xlim([-5, np.max(images)])
plt.hist(images[0,:], 100, fill=True, label='Image 1')
plt.hist(images[1,:], 100, fill=True, label='Image 2')
plt.hist(images[2,:], 100, fill=True, label='Image 3')
plt.hist(images[3,:], 100, fill=True, label='Image 4')
plt.hist(images[4,:], 100, fill=True, label='Image 5')
plt.hist(images[5,:], 100, fill=True, label='Image 6')
plt.hist(images[6,:], 100, fill=True, label='Image 7')
plt.hist(images[7,:], 100, fill=True, label='Image 8')
plt.hist(images[8,:], 100, fill=True, label='Image 9')
plt.hist(images[9,:], 100, fill=True, label='Image 10')
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.ylabel('Counts')
plt.semilogy()
plt.xlabel('Magnitude')
plt.title('Background Distribution Over Time (Log Scale)')
plt.legend()
plt.show()


# As we can see from the plot above, there is some minor variation in what we would consider the background (on the left) between images, but it is not large enough to consider the background to be time-dependent in a way that will affect what we are looking for. However, where the background ends and where the signal tail starts, there is certainly a good deal of time-dependence, as the distribution changes quite a bit from image to image. These are the measurements we would be looking at for transients and stars, and this is where we will find the faint stars.
# 
# Before we do that, we need to determine the cut-off threshold for the background that will give us a measurement with low enough probability that we can call it a signal. Additionally, we can remove the background from the data we are looking at in order to reduce the size of the datasets we are using and comparing.
# 
# We can model the distribution by using the minimum value measured for any pixel in any image and reflect it over the vertical axis, creating possibly a broader distribution than the background we would see in any given image, but hopefully encompassing all backgrounds of the images in the process.

# In[447]:


print(np.min(images))


# In[448]:


x = np.arange(-5, 10, 1000)
plt.xlim([-5,5])
plt.hist(images[0,:], 100, fill=True, label='Image 1')
plt.hist(images[1,:], 100, fill=True, label='Image 2')
plt.hist(images[2,:], 100, fill=True, label='Image 3')
plt.hist(images[3,:], 100, fill=True, label='Image 4')
plt.hist(images[4,:], 100, fill=True, label='Image 5')
plt.hist(images[5,:], 100, fill=True, label='Image 6')
plt.hist(images[6,:], 100, fill=True, label='Image 7')
plt.hist(images[7,:], 100, fill=True, label='Image 8')
plt.hist(images[8,:], 100, fill=True, label='Image 9')
plt.hist(images[9,:], 100, fill=True, label='Image 10')
plt.axvline(x=np.min(images), label='Lower threshold')
plt.axvline(x=np.abs(np.min(images)), label='Upper threshold')
plt.ylabel('Counts')
plt.semilogy()
plt.xlabel('Magnitude')
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.title('Background Distribution Over Time (Log Scale)')
plt.legend()
plt.show()


# In[449]:


bkgd1 = images[0,:].copy()
bkgd2 = images[1,:].copy()
bkgd3 = images[2,:].copy()
bkgd4 = images[3,:].copy()
bkgd5 = images[4,:].copy()
bkgd6 = images[5,:].copy()
bkgd7 = images[6,:].copy()
bkgd8 = images[7,:].copy()
bkgd9 = images[8,:].copy()
bkgd10 = images[9,:].copy()

for j in range(0,40000):
    if bkgd1[j]>=2.71:
            bkgd1[j] = 0
    if bkgd2[j]>=2.71:
            bkgd2[j] = 0
    if bkgd3[j]>=2.71:
            bkgd3[j] = 0
    if bkgd4[j]>=2.71:
            bkgd4[j] = 0
    if bkgd5[j]>=2.71:
            bkgd5[j] = 0
    if bkgd6[j]>=2.71:
            bkgd6[j] = 0
    if bkgd7[j]>=2.71:
            bkgd7[j] = 0
    if bkgd8[j]>=2.71:
            bkgd8[j] = 0
    if bkgd9[j]>=2.71:
            bkgd9[j] = 0
    if bkgd10[j]>=2.71:
            bkgd10[j] = 0


# In[450]:


x = np.arange(-5, 10, 1000)
#plt.xlim([-5, np.max(images)])
plt.hist(bkgd1, 100, fill=True, label='Image 1')
plt.hist(bkgd2, 100, fill=True, label='Image 2')
plt.hist(bkgd3, 100, fill=True, label='Image 3')
plt.hist(bkgd4, 100, fill=True, label='Image 4')
plt.hist(bkgd5, 100, fill=True, label='Image 5')
plt.hist(bkgd6, 100, fill=True, label='Image 6')
plt.hist(bkgd7, 100, fill=True, label='Image 7')
plt.hist(bkgd8, 100, fill=True, label='Image 8')
plt.hist(bkgd9, 100, fill=True, label='Image 9')
plt.hist(bkgd10, 100, fill=True, label='Image 10')
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.ylabel('Counts')
plt.semilogy()
plt.xlabel('Magnitude')
plt.title('Background Distribution Over Time (Log Scale)')
plt.legend()
plt.show()


# The above graph shows a pretty good summarization of the background overall. Since we are looking for faint stars, what we want to do is remove the background from the data set and find those measurements which are likely to be stars. 
# 
# Below are some ways to manipulate the data in ways that change the appearance of the background. As we can see, convolution causes the background to be more prominent, while summing the ten images together and dividing by the number of images sharpens the appearance of stars and other objects not a part of the background.

# In[453]:


from scipy.signal import convolve2d

image1_1 = image1/np.sum(image1)
image1_avg = scipy.signal.convolve2d(image1_1,image1)
image2_avg = scipy.signal.convolve2d(image1,image1)


# In[454]:


image_avg = (imstack[:,:,0] + imstack[:,:,1] + imstack[:,:,2] + imstack[:,:,3] + imstack[:,:,4] + imstack[:,:,5] + imstack[:,:,6] + imstack[:,:,7]
             + imstack[:,:,8] + imstack[:,:,9])/10
fig, ax = plt.subplots(1, 4)
ax[0].imshow(image1)
ax[1].imshow(image1_avg)
ax[2].imshow(image2_avg)
ax[3].imshow(image_avg)
ax[0].set_title('Original Image')
ax[1].set_title('Average (Convolution) of Original')
ax[2].set_title('Sum (Convolution) of Original')
ax[3].set_title('Average of All Images')
plt.tick_params(labelsize = 12)
plt.show()


# The image on the far right is what we need: we need a sharper image by which we can distinguish the faint stars from the background. Therefore, we can remove the background we found earlier, and concentrate on those measurements which do not fit neatly into the background.

# In[455]:


sig1 = images[0,:].copy()
sig2 = images[1,:].copy()
sig3 = images[2,:].copy()
sig4 = images[3,:].copy()
sig5 = images[4,:].copy()
sig6 = images[5,:].copy()
sig7 = images[6,:].copy()
sig8 = images[7,:].copy()
sig9 = images[8,:].copy()
sig10 = images[9,:].copy()

for j in range(0,40000):
    if sig1[j]<=2.71:
            sig1[j] = 0
    if sig2[j]<=2.71:
            sig2[j] = 0
    if sig3[j]<=2.71:
            sig3[j] = 0
    if sig4[j]<=2.71:
            sig4[j] = 0
    if sig5[j]<=2.71:
            sig5[j] = 0
    if sig6[j]<=2.71:
            sig6[j] = 0
    if sig7[j]<=2.71:
            sig7[j] = 0
    if sig8[j]<=2.71:
            sig8[j] = 0
    if sig9[j]<=2.71:
            sig9[j] = 0
    if sig10[j]<=2.71:
            sig10[j] = 0


# In[456]:


np.max(images)


# In[465]:


#x = np.arange(-5, 10, 1000)
plt.xlim([3, 50])
plt.hist(sig1, 100, fill=True, label='Image 1')
plt.hist(sig2, 100, fill=True, label='Image 2')
plt.hist(sig3, 100, fill=True, label='Image 3')
plt.hist(sig4, 100, fill=True, label='Image 4')
plt.hist(sig5, 100, fill=True, label='Image 5')
plt.hist(sig6, 100, fill=True, label='Image 6')
plt.hist(sig7, 100, fill=True, label='Image 7')
plt.hist(sig8, 100, fill=True, label='Image 8')
plt.hist(sig9, 100, fill=True, label='Image 9')
plt.hist(sig10, 100, fill=True, label='Image 10')
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.ylabel('Counts')
plt.semilogy()
plt.xlabel('Magnitude')
plt.title('Signal Distribution Over Time (Log Scale)')
plt.legend()
plt.show()


# As the plot above shows, what is left when we remove the background is still quite messy, and it changes with time. In order to get a better idea of possible candidates, we need to do some averaging over time and see if we can get some signals which are not on the high end (these are the bright stars) and are not transients (which will look like their own distributions).

# In[466]:


average = (sig1+sig2+sig3+sig4+sig5+sig6+sig7+sig8+sig9+sig10)/10

plt.hist(average, 100, fill=True, label='Average')

plt.ylabel('Counts')
plt.semilogy()
plt.xlabel('Magnitude')
plt.title('Averaged Signal Distribution (Log Scale)')
plt.legend()
plt.show()


# Our averaged signal above shows a clearer distribution, which is exactly what we were hoping for. We have five or six clear candidates for faint stars between magnitudes fifteen and thirty. They show up as clear bars, because they are not moving.

# This is a different case from what we would want to use if we were looking for transients. If that were the case, rather than using a Poisson distribution, we would want to use a Rayleigh distribution in order to judge whether or not movement is taking place.

# In[ ]:




