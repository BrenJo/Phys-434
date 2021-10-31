#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import h5py

plt.rcParams["figure.figsize"] = (20,10)


# In[3]:


hf = h5py.File('gammaray_lab4.h5', 'r')


# In[4]:


hf.keys()


# In[5]:


data = np.array(hf.get('data'))
data[:,0]


# In[6]:


hf.close()


# In[7]:


t = data[0,:]
sol = data[1,:]
lon = data[2,:]
n = data[3,:]
print(n.size)
print(np.max(n))


# In[8]:


fig, ax = plt.subplots(1, 2)
ax[0].hist(n, 30, density=True)
ax[1].hist(n, 30, density=True)
ax[1].semilogy()
plt.tick_params(labelsize = 24)
plt.xlim([0,35])
plt.show()
fig.tight_layout()


# In[11]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(sol, t, bins=30)
ax[1].hist2d(sol, n, bins=30)
plt.tick_params(labelsize = 24)
plt.show()
fig.tight_layout()


# In[17]:


fig, ax = plt.subplots(1, 2)
ax[0].hist2d(lon, t, bins=45)
ax[1].hist2d(lon, n, bins=30)
plt.tick_params(labelsize = 24)
plt.show()
fig.tight_layout()


# In[36]:


hist, xedges, yedges, bins = plt.hist2d(lon, n, bins = 30)
xedges


# In[45]:


print(hist[0])


# In[44]:


avg = np.empty([1])
counts = 0
for i in range(0,30):
    counts = np.sum(hist[i])
    avg_bin = np.array(counts/hist[i].size)
    avg = np.append(avg, [avg_bin], axis=0)
print(avg.shape)
print(avg)


# In[26]:


f = np.where(lon == np.min(np.abs(lon - 360)))
print(np.min(np.abs(lon - 360)))
print(f)


# In[ ]:


print()


# In[20]:


lon_f = np.empty([1,54000])
n_f = np.empty([1,54000])

for i in range(0,n.size):
    mod = lon[i]%360
    
    if mod<=0.002:
        new_row_n = n[i:(i+54000)]
        new_row_lon = lon[i:(i+54000)]
        n_f = np.append(n_f,[new_row_n],axis=0)
        lon_f = np.append(lon_f,[new_row_lon],axis=0)
    else:
        n_f = n_f
        lon_f = lon_f

print(n_f.shape)
print(lon_f.shape)


# In[19]:


print(lon[0],lon[-1])
print(lon.size)
print(n.size)
print(lon.size/360)


# In[17]:


num = 0
for i in range(0,n.size):
    if lon[i]>=0 and lon[i]<=12:
        num = num + 1
    else:
        num = num
print(num)


# In[26]:


bin_1 = np.empty([1,864481])
bin_2 = np.empty([1,864481])
bin_3 = np.empty([1,864481])
bin_4 = np.empty([1,864481])
bin_5 = np.empty([1,864481])
bin_6 = np.empty([1,864481])
bin_7 = np.empty([1,864481])
bin_8 = np.empty([1,864481])
bin_9 = np.empty([1,864481])
bin_10 = np.empty([1,864481])
bin_11 = np.empty([1,864481])
bin_12 = np.empty([1,864481])
bin_13 = np.empty([1,864481])
bin_14 = np.empty([1,864481])
bin_15 = np.empty([1,864481])
bin_16 = np.empty([1,864481])
bins = np.empty([1,864481])
bins = np.empty([1,864481])


# In[25]:


#counts = np.empty([1])
bins = np.empty([1,864481])
degree = 0

for j in range(0,30):
    counts = np.empty([1])
    bins = bins
    degree = degree
    for k in range(0,n.size):
        if lon[k]>=degree and lon[k]<=(degree + 12):
            counts = np.append(counts, [n[k]], axis=0)
        else:
            counts = counts
        if degree>360:
            degree = 0
        else:
            degree = degree
        if counts.size==864481:
            counts = counts
        else:
            counts = np.append(counts, [0], axis=0)
    bins = np.append(bins, [counts], axis=0)
print(bins.shape)


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
    


# In[80]:


T_lon = 5400  # in GPS seconds

lon_f = np.empty([1,T_lon])
n_f = np.empty([1,T_lon])

for i in range(0,t.size):
    mod = t[i]%T_lon
    
    if mod<=0:
        new_row_n = n[i:(i+T_lon)]
        new_row_lon = lon[i:(i+T_lon)]
        n_f = np.append(n_f,[new_row_n],axis=0)
        bkgd_f = np.append(lon_f,[new_row_lon],axis=0)
    else:
        n_f = n_f
        lon_f = lon_f

print(n_f.shape)
print(lon_f.shape)


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




