# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:04:20 2020

@author: sstucker
"""

import numpy as np
from math import acos, asin
import matplotlib.pyplot as plt

MAX_ALINE_RATE = 76000  # Hz. 
exposure_samples = 2  # The samples set to high per camera trigger. The width of this pulse defines the exposure time
shutter_samples = 2  # The samples set to low per camera trigger
B_FRAC = 0.23856334924933  # The fraction of a figure 8 half cycle used for the exposure such that aline width is unitary for unit amplitude pattern

alines_per_b = 100
aline_width = 0.05  # mm

def xfunc(t):
    return aline_width*np.cos(t)

def yfunc(t):
    return aline_width*np.cos(t)*np.sin(t)

# Start and end of B-line in radian time units
bstart = np.pi / 2 - (np.pi * B_FRAC) / 2
bend = np.pi / 2 + (np.pi * B_FRAC) / 2

# The distance along the B-line
bdist = np.sqrt((xfunc(bstart) - xfunc(bend))**2 + (yfunc(bstart) - yfunc(bend))**2)  # mm

period_samples = exposure_samples + shutter_samples
max_fs = MAX_ALINE_RATE * period_samples
print('DAC sample rate:', max_fs)

# Build B-line trigger
bline = np.tile(np.append(np.ones(exposure_samples), np.zeros(shutter_samples)), alines_per_b)
bpad = np.zeros(np.ceil((len(bline) - len(bline) * B_FRAC) / (2 * B_FRAC)).astype(int))
trigger = np.concatenate([bpad, bline, bpad, bpad, bline, bpad])

b1idx = np.concatenate([np.zeros(len(bpad)), np.ones(len(bline)), np.zeros(3 * len(bpad) +len(bline))]).astype(bool)
b2idx = np.concatenate([np.zeros(3 * len(bpad) +len(bline)), np.ones(len(bline)), np.zeros(len(bpad))]).astype(bool)

b1x = np.linspace(xfunc(bstart), xfunc(bend), len(bline))
b1y = np.linspace(yfunc(bstart), yfunc(bend), len(bline))
b2x = np.linspace(xfunc(bstart + np.pi), xfunc(bend + np.pi), len(bline))
b2y = np.linspace(yfunc(bstart + np.pi), yfunc(bend + np.pi), len(bline))

t = np.linspace(0, 2*np.pi, len(trigger))
x = xfunc(t)
y = yfunc(t)
x[b1idx] = b1x
y[b1idx] = b1y
x[b2idx] = b2x
y[b2idx] = b2y

plt.figure(1)
plt.plot(trigger)
plt.plot(x)
plt.plot(y)


plt.figure(2)
ax2 = plt.axes()
ax2.set_aspect('equal')
ax2.plot(x, y)
ax2.scatter(x[trigger.astype(bool)], y[trigger.astype(bool)])
