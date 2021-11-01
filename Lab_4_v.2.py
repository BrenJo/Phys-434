#!/usr/bin/env python
# coding: utf-8

# In[153]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py

plt.rcParams["figure.figsize"] = (10,5)


# In[154]:


hf = h5py.File('gammaray_lab4.h5', 'r')


# In[155]:


hf.keys()


# In[156]:


data = np.array(hf.get('data'))
data[:,0]


# In[157]:


hf.close()


# In[158]:


t = data[0,:]
sol = data[1,:]
lon = data[2,:]
n = data[3,:]
print(n.size)
print(np.max(n))


# In[160]:


x = np.arange(0,30)
mean = np.mean(n)
exp = stats.poisson.pmf(x, mean, loc=0)

fig, ax = plt.subplots(1, 2)
ax[0].hist(n, 30, density=True)
ax[0].step(x, exp)
ax[1].hist(n, 30, density=True)
ax[1].step(x, exp)
ax[1].semilogy()
plt.tick_params(labelsize = 14)
plt.xlim([0,35])
plt.show()
fig.tight_layout()


# In[161]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(sol, t, bins=30)
ax[1].hist2d(sol, n, bins=30)
plt.tick_params(labelsize = 14)
plt.show()
fig.tight_layout()


# In[162]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(lon, t, bins=45)
ax[1].hist2d(lon, n, bins=30)
plt.tick_params(labelsize = 14)
plt.show()
fig.tight_layout()


# In[163]:


hist, xedges, yedges, bins = plt.hist2d(lon, n, bins = 30)
xedges


# In[ ]:





# In[164]:


print(hist[0])
print(lon[40])


# In[171]:


print(lon[0],lon[-1])
print(sol[0],sol[-1])
print(sol[40])


# In[172]:


g = (n/4)*np.cos((2*np.pi/360)*lon*t) + (n/4)*np.sin((2*np.pi/360)*sol*t) + mean
plt.plot(t[0:800], g[0:800], 'b')
plt.plot(t[0:800], n[0:800], 'r')
plt.show()


# In[173]:


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
plt.show()
fig.tight_layout()


# In[174]:


plt.rcParams["figure.figsize"] = (10,5)
avg = np.empty([1])

for i in hist:
    counts = 0
    for j in range(len(i)):
            counts = counts + j*i[j]
    avg = np.append(avg, counts/(len(i)*7e4))
print(xedges[0])
plt.plot(xedges[1:], avg[1:], 'b', label='Average')
print(hist.shape)
print(np.min(avg))
print(np.where(avg == np.min(avg)))
print(xedges[26])
print(np.max(avg))
print(np.where(avg == np.max(avg)))
print(xedges[28]-xedges[26])


# In[175]:


x = np.linspace(0,360)
mu = xedges[28]-xedges[26]
plt.plot(xedges[26:]-xedges[26], ave[25:], 'b', label='Average')
plt.plot(xedges[1:26]+xedges[-1]-xedges[26], ave[0:25], 'b')
plt.plot(x, 1.4*np.log(6*x)-3, 'r')
plt.plot(x, 2.6*np.exp(-0.0114*x)+2.35, 'k')
plt.plot(x, 200*stats.lognorm.pdf(x, 1, loc = 0, scale = np.exp(mu)), 'g')
plt.show()


# In[101]:


T_lon = 5400  # in GPS seconds
r = 0
for i in range(0,lon.size):
    mod = lon[i]%45.0
    
    if mod<=0.1199999:
        r = r + 1
    else:
        r = r
print(r)
    


# In[ ]:





# In[ ]:


T_lon = 5400
t.size/T_lon


# In[83]:


im = h5py.File('images.h5', 'r')


# In[84]:


im.keys()


# In[90]:


image1 = np.array(im.get('image1'))
imstack = np.array(im.get('imagestack'))


# In[91]:


im.close()


# In[92]:


image1.shape


# In[93]:


imstack.shape


# In[ ]:


fig, ax = plt.subplots(1, 1)
plt.imshow(image1)
plt.tick_params(labelsize = 24)
plt.show()


# In[ ]:




