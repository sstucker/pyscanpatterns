# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020

@author: sstucker
"""

import numpy as np
import matplotlib.pyplot as plt


class ScanPattern:

    def __init__(self, geometry, axial=False, rate=5, order=3, ss=10000):
        """
        :param geometry: The type of scan pattern to generate. Options include:
            'raster': Conventional raster scan (C-scan)
            'circle': Circular scan
            'lemniscate': A figure-8 with two perpendicular B-scans
            'rose': Odd petaled rhodonea function with n-perpendicular B-scans
        :param axial: Whether or not to generate a Z-coordinate signal in the 3rd axis
        :param rate: Rate in hz to complete a C scan
        :param order: for use with 'rose' patterns, number of B-scans [3, 5, 7, 9]
        :param ss: Sample rate of generated drive signals in samples per second
        """

        self.rate = rate
        self.ss = ss

        self._x = np.array([])
        self._y = np.array([])
        self._trigger = np.array([])
        if axial:
            self._z = np.array([])

        if geometry is 'raster':
            self._generate_raster_scan
        elif geometry is 'circe':
            pass
        elif geometry is 'lemniscate':
            pass
        elif geometry is 'rose':
            pass
        else:
            print('Geometry mode ', geometry, ' not understood. Initializing in Raster mode.')

    def _generate_raster_scan(self, alines, blines, flyback_duty=0.05, fov=None, spacing=None):
        """
        Generates raster pattern. If no fov or spacing is passed, constrains FOV to -2, 2 voltage units
        :param alines: Number of A-lines in raster scan
        :param blines: Number of B-lines in raster scan
        :param flyback_duty: Percentage of B-line scan time to be used for flyback signal. Default 5%
        :param fov: [a, b] 2D field of view
        :param spacing: [a, b] 2D spacing. Overrides fov definition if both are passed
        :return: -1 if error, 0 if successful
        """

        t = np.linspace()

        new_x = []
        for bline in range(blines):
            new_x.append(np.linspace(-1, 1, fs))

myPat = ScanPattern('raster')


