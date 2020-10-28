# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020
@author: sstucker
"""
import time
import numpy as np
from scipy import signal


def rotfunc(xs, ys, rotation_rad):
    rot = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                    [np.sin(rotation_rad), np.cos(rotation_rad)]])
    xy = np.column_stack([xs, ys])
    prod = np.matmul(xy, rot)
    return prod[:, 0], prod[:, 1]


class LineScanPattern:
    """
    Template for all scan patterns for the generation of 2D or 3D images acquired with A-line scans. Generates
    normalized signals for the steering of galvos and the triggering of a line camera in order to acquire a given
    geometry.
    """

    def get_sample_rate(self):
        """
        :return: The approximate sample rate in hz that samples should be generated such that the pattern is
        completed at the rate specified
        """
        raise NotImplementedError()

    def get_pattern_rate(self):
        """
        :return: The approximate rate at which the pattern is completed by the scanner. This is defined by user, but
        not always directly: usually, the maximum rate of line camera exposure drives this value
        """
        raise NotImplementedError()

    def get_line_trig(self):
        """
        :return: A continuous waveform representing the digital (1 and 0) triggering of the line camera's exposure.
        """
        raise NotImplementedError()

    def get_frame_trig(self):
        """
        :return: A continuous waveform representing the digital (1 and 0) triggering of the beginning of a new
        repetition of the pattern. This is used to trigger the frame grabber
        """
        raise NotImplementedError()

    def get_x(self):
        """
        :return: A continuous waveform representing the analog signal for driving the first dimension of the scanner.
        In voltage units that must be converted to physical units by the user.
        """
        raise NotImplementedError()

    def get_y(self):
        """
        :return: A continuous waveform representing the analog signal for driving the second dimension of the scanner.
        In voltage units that must be converted to physical units by the user.
        """
        raise NotImplementedError()

    def get_dimensions(self):
        """
        :return: An array of pattern dimensions which orient the A-scans with respect to one another. i.e. a raster
        pattern's first and second dimension represent the height and width of the pattern in A-lines.
        """
        raise NotImplementedError()

    def get_total_number_of_alines(self):
        """
        :return: The total number of line acquisitions per pattern
        """
        raise NotImplementedError()

    def generate(self):
        """
        Generate the pattern from some number of parameters. The pattern object is unuseable until this is called
        :return: 0 on success
        """
        raise NotImplementedError()


class RasterScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=76000):
        """
        :param max_trigger_rate: The maximum rate that the camera can be triggered in Hz. The scan pattern and its
        rate will be determined to achieve but not exceed this rate. Default < 76 kHz.
        """

        self._max_rate = max_trigger_rate

        self._pattern_rate = None

        self._x = np.array([])
        self._y = np.array([])
        self._line_trig = np.array([])
        self._frame_trig = np.array([])
        self._fs = None
        self._alines = None
        self._blines = None

        self.generate()

    def get_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines]
        """
        return [self._alines, self._blines]

    def get_total_number_of_alines(self):
        return int(self._alines * self._blines)

    def get_sample_rate(self):
        return self._fs

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_signals(self):
        return [self._line_trig, self._frame_trig, self._x, self._y]

    def get_line_trig(self):
        return self._line_trig

    def get_frame_trig(self):
        return self._frame_trig

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def generate(self, alines=64, blines=1, flyback_duty=0.2, exposure_width=0.8,
                 fov=None, samples_on=2, samples_off=None, rotation_rad=0, slow_axis_step=False, trigger_blines=False):
        """
        Generates raster pattern. If no fov or spacing is passed, constrains FOV to -1, 1 voltage units
        :param alines: Number of A-lines in raster scan
        :param blines: Number of B-lines in raster scan
        :param flyback_duty: Percentage of B-line scan time to be used for flyback signal. Default 15%
        :param exposure_width: Percentage of entire fast-axis length to trigger exposures along. The ends of the fast-
            axis are prone to distortions as the galvos change direction. Default 70%
        :param fov: [FOV width (fast axis), FOV height (slow axis)] 2D field of view in voltage-spatial units
        :param samples_on: Line exposure time in samples. Must exceed 0. Default None: same as samples off (50% duty)
        :param samples_off: The number of low samples to generate between exposure trigger pulses. Limited by dac rate.
            Default 2.
        :param rotation_rad: Angle in radians by which to rotate scan pattern
        :return: -1 if error, 0 if successful
        """

        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self._alines = int(alines)
        self._blines = int(blines)

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        # Build B-line trigger
        bline_pulses = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._alines)
        bline_pad = np.zeros(int(len(bline_pulses) * (1 - exposure_width) / 2))
        bline_trig_padded = np.concatenate([bline_pad, bline_pulses, bline_pad])
        flyback_pad = np.zeros(int((flyback_duty * len(bline_trig_padded)) / 2))

        bline_trig = np.concatenate([flyback_pad, bline_trig_padded, flyback_pad])
        self._line_trig = np.tile(bline_trig, self._blines)

        # Build frame trigger
        if trigger_blines:  # Frame trigger for each B-line
            frame_trig_per_b = np.zeros(len(bline_trig))
            frame_trig_per_b[0:samples_on] = 1
            self._frame_trig = np.tile(frame_trig_per_b, self._blines)
        else:  # If one trigger for entire volume
            self._frame_trig = np.zeros(len(self._line_trig))
            self._frame_trig[0:samples_on] = 1

        # Fast axis generation
        x = np.concatenate([
            np.linspace(0, fov[0] / 2, len(flyback_pad)),
            np.linspace(fov[0] / 2, -fov[0] / 2, len(bline_trig_padded)),
            np.linspace(-fov[0] / 2, 0, len(flyback_pad))
        ])
        self._x = np.tile(x, self._blines)

        # Slow axis generation
        if slow_axis_step:  # Step function
            pos = np.linspace(-fov[1] / 2, fov[1] / 2, self._blines)
            self._y = np.array([])
            for i in range(0, len(pos)):
                flyto = np.linspace(pos[i - 1], pos[i], int(2 * len(flyback_pad)))
                hold = pos[i] * np.ones(len(bline_trig_padded))
                self._y = np.concatenate([self._y, flyto, hold])
            self._y = np.roll(self._y, -len(flyback_pad))
        else:  # Slow axis is a sawtooth
            self._y = np.concatenate([
                np.linspace(0, fov[1] / 2, len(flyback_pad)),
                np.linspace(fov[1] / 2, -fov[1] / 2, self._blines * len(bline_trig_padded) + 2 * (self._blines - 1) * len(flyback_pad)),
                np.linspace(-fov[1] / 2, 0, len(flyback_pad)),
            ])

        self._pattern_rate = 1 / ((1 / self._fs) * len(self._y))

        return 0


class Figure8ScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=75000):

        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._line = np.array([])
        self._frame = np.array([])
        self._pattern_rate = None
        self._fs = None
        self._aline_width = None
        self._a_per_b = None

        self.generate()

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_dimensions(self):
        return [self._a_per_b, 2]

    def get_total_number_of_alines(self):
        return int(self._a_per_b * 2)

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._line, self._frame, self._x, self._y]

    def get_line_trig(self):
        return self._line

    def get_frame_trig(self):
        return self._frame

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def generate(self, aline_width=0.05, aline_per_b=64, samples_on=1, samples_off=None, rotation_rad=0):

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self._aline_width = aline_width
        self._a_per_b = int(aline_per_b)

        # The fraction of a figure 8 half cycle used for the exposure such that aline width is 1
        b_frac = 0.23856334924933

        def xfunc(ts):
            return aline_width * np.cos(ts)

        def yfunc(ts):
            return aline_width * np.cos(ts) * np.sin(ts)

        # Start and end of B-line in radian time units
        bstart = np.pi / 2 - (np.pi * b_frac) / 2
        bend = np.pi / 2 + (np.pi * b_frac) / 2

        # The distance along the B-line
        bdist = np.sqrt((xfunc(bstart) - xfunc(bend)) ** 2 + (yfunc(bstart) - yfunc(bend)) ** 2)  # mm

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        # Build B-line trigger
        bline = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._a_per_b)

        bpad = np.zeros(np.ceil((len(bline) - len(bline) * b_frac) / (2 * b_frac)).astype(int))
        line_trigger = np.concatenate([bpad, bline, bpad, bpad, bline, bpad])

        frame_start = np.zeros(len(line_trigger))
        frame_start[len(bpad) - 8:len(bpad) - 8 + samples_on] = 1

        b1idx = np.concatenate([np.zeros(len(bpad)), np.ones(len(bline)), np.zeros(3 * len(bpad) + len(bline))]).astype(
            bool)
        b2idx = np.concatenate([np.zeros(3 * len(bpad) + len(bline)), np.ones(len(bline)), np.zeros(len(bpad))]).astype(
            bool)

        # Straighten B-lines
        b1x = np.linspace(xfunc(bstart), xfunc(bend), len(bline))
        b1y = np.linspace(yfunc(bstart), yfunc(bend), len(bline))
        b2x = np.linspace(xfunc(bstart + np.pi), xfunc(bend + np.pi), len(bline))
        b2y = np.linspace(yfunc(bstart + np.pi), yfunc(bend + np.pi), len(bline))

        t = np.linspace(0, 2 * np.pi, len(line_trigger))
        x = xfunc(t)
        y = yfunc(t)
        x[b1idx] = b1x
        y[b1idx] = b1y
        x[b2idx] = b2x
        y[b2idx] = b2y

        x, y = rotfunc(x, y, rotation_rad)

        self._x = x
        self._y = y
        self._line = line_trigger
        self._frame = frame_start

        self._pattern_rate = 1 / ((1 / self._fs) * len(line_trigger))


class RoseScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=75950):

        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._line = np.array([])
        self._frame = np.array([])
        self._pattern_rate = None
        self._fs = None

        # Defaults
        self._k = 3
        self._aline_width = 0.06
        self._a_per_b = 36

        self.generate(self._k, self._aline_width, self._a_per_b)

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_dimensions(self):
        return [self._a_per_b, 2]

    def get_total_number_of_alines(self):
        return int(self._a_per_b * 2)

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._line, self._frame, self._x, self._y]

    def get_line_trig(self):
        return self._line

    def get_frame_trig(self):
        return self._frame

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def generate(self, k, aline_width, aline_per_b, samples_on=1, samples_off=None, rotation_rad=0):

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self._aline_width = aline_width
        self._a_per_b = int(aline_per_b)

        b_frac = 0.3

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        # Build B-line trigger
        bline = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._a_per_b)
        bpad = np.zeros(np.ceil((len(bline) - len(bline) * b_frac) / (2 * b_frac)).astype(int))
        kth_b = np.concatenate([bpad, bline, bpad])

        line_trigger = np.tile(kth_b, k)

        frame_start = np.zeros(len(line_trigger))
        frame_start[0:samples_on] = 1

        def xfunc(ts):
            return aline_width * np.cos(k * ts) * np.cos(ts)

        def yfunc(ts):
            return aline_width * np.cos(k * ts) * np.sin(ts)

        t = np.linspace(0, np.pi, len(line_trigger))
        x = xfunc(t)
        y = yfunc(t)

        x_onsets = []
        y_onsets = []
        for i in range(len(line_trigger)):
            if line_trigger[i] == 1:
                x_onsets.append(x[i])
                y_onsets.append(y[i])

        # plt.figure(1)
        # plt.plot(x, y)
        # plt.scatter(x_onsets, y_onsets)
        # plt.title('k = '+str(k))
        # plt.show()
        #
        # plt.figure(2)
        # plt.plot(x)
        # plt.plot(y)
        # plt.plot(line_trigger)

        x, y = rotfunc(x, y, rotation_rad)

        self._x = x
        self._y = y
        self._line = line_trigger
        self._frame = frame_start

        self._pattern_rate = 1 / ((1 / self._fs) * len(line_trigger))


if __name__ == "__main__":

    raster = RasterScanPattern()
    raster.generate(10, 1)

    x = raster.get_x()
    y = raster.get_y()
    l = raster.get_line_trig()
    f = raster.get_frame_trig()

    print("Raster pattern rate", raster.get_pattern_rate(), 'hz')
    print("DAC sample rate", raster.get_sample_rate(), 'hz')

    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.plot(y)
    plt.plot(l)
    plt.plot(f)
    plt.show()
