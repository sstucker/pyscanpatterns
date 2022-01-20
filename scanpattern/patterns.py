import numpy as np


def _sigmoidspace(start, stop, n, b=1):
    """Interpolate n points of a sigmoidal function between start and stop.

    Like numpy.linspace, but sigmoidally eased.

    Args:
        start (float): Start value
        stop (float): Stop value
        n (int): The number of points to interpolate
        b (int): start + (stop - start) / (1 + np.exp(-b * np.linspace(-10, 10, n)))
    """
    return start + (stop - start) / (1 + np.exp(-b * np.linspace(-10, 10, n)))


def _rotfunc(xs, ys, rotation_rad):
    """Rotate the coordinates xs and ys by the given angle in radians

    Args:
        xs (numpy.ndarray): X coordinates
        ys (numpy.ndarray): Y coordinates
        rotation_rad (float): Angle by which to rotate the points in radians
    """
    rot = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                    [np.sin(rotation_rad), np.cos(rotation_rad)]])
    xy = np.column_stack([xs, ys])
    prod = np.matmul(xy, rot)
    return prod[:, 0], prod[:, 1]


class LineScanPattern:
    """
    Template class for all scan patterns for the generation of 2D or 3D images acquired with A-line scans. Generates
    normalized signals for the steering of galvos and the triggering of a line camera in order to acquire a given
    geometry.
    """

    def __init__(self):
        self._line_trigger = np.array([])
        self._frame_trigger = np.array([])
        self._x = np.array([])
        self._y = np.array([])
        self._sample_rate = 0
        self._pattern_rate = 0

    @property
    def line_trigger(self):
        return self._line_trigger

    @property
    def frame_trigger(self):
        return self._frame_trigger

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def sample_rate(self):
        """The approximate samples per second that should be generated by the DAC such that the pattern is completed at the rate specified."""
        return self._sample_rate

    @property
    def pattern_rate(self):
        """The approximate rate at which the pattern is specified to be completed by the scanner."""
        return self._pattern_rate
    
    @property
    def dimensions(self):
        """An array of dimensions which orient the A-scans with respect to one another. i.e. a raster pattern's first and second dimension represent the height and width of the pattern in A-lines."""
        raise NotImplementedError()
    
    @property
    def total_number_of_alines(self):
        """The total number of point acquisitions per pattern."""
        raise NotImplementedError()

    def generate(self):
        """Generate the pattern using some number of parameters. The pattern object is unuseable until this is called."""
        raise NotImplementedError()


class Figure8ScanPattern(LineScanPattern):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.aline_width = None
        self.aline_per_b = None
        self.max_trigger_rate = None
        self.samples_on = None
        self.samples_off = None
        self.rotation_rad = None
        self.h = None

        self.aline_per_b = None

        if len(args) > 0:
            self.generate(*args, **kwargs)

    @property
    def dimensions(self):
        return [self.aline_per_b, 2]

    @property
    def total_number_of_alines(self):
        return int(self.aline_per_b * 2)

    def generate(self, aline_width: float, aline_per_b: int, max_trigger_rate: float,
                 samples_on: int = 1, samples_off: int = None, rotation_rad: float = 0, h: float = 5):
        """Generate a Figure-8 pattern: a pattern consisting of two perpendicular B-scans.

        Args:
            aline_width (float): The width of the A-line in units.
            aline_per_b (int): The number of A-lines in each B-line.
            max_trigger_rate (float): The maximum rate in Hz at which to toggle the line trigger signal. This is the primary constraint on imaging rate.
            samples_on (int): Optional. The number of trigger samples to drive high for each exposure.
            samples_off (int): Optional. The number of trigger samples to drive low after each exposure. By default, equivalent to `samples_on`.
            rotation_rad (float): Optional. Rotates the scan. Default 0.
            h (float): Optional. 1 / H is the proportion of the B-line length to exclude from the scan for flyback. Default 5.
        """

        self.aline_width = aline_width
        self.aline_per_b = aline_per_b
        self.max_trigger_rate = max_trigger_rate
        self.samples_on = samples_on
        self.samples_off = samples_off
        self.rotation_rad = rotation_rad
        self.h = h

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self.aline_width = aline_width
        self.aline_per_b = int(aline_per_b)

        # Build B-line trigger
        btrig = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self.aline_per_b)

        # TODO parameterize these, especially H
        G = np.pi / 2.5  # Cylical point at which B-scans begin in the figure
        A = 1 / 1.6289944852363252  # Factor such that B-scans given a G are normalized in length
        AY = 1.223251  # Stretch factor along second axis to achieve perpendicular B-scans
        H = 5  # 1 / H is the proportion of the B-line length used for flyback in samples

        def fig8(t):
            x = A * np.cos(t)
            y = A * np.cos(t) * np.sin(t) * AY
            return [x, y]

        t_fb0 = _sigmoidspace(-G, G, int(1 / H * len(btrig)), b=0.2)
        [x_fb0, y_fb0] = fig8(t_fb0)

        t_fb1 = _sigmoidspace(-G + np.pi, G + np.pi, int(1 / H * len(btrig)), b=0.2)
        [x_fb1, y_fb1] = fig8(t_fb1)

        x_b1 = np.linspace(x_fb0[-1], x_fb1[0], len(btrig))
        y_b1 = np.linspace(y_fb1[-1], y_fb0[0], len(btrig))

        x_b0 = np.linspace(x_fb1[-1], x_fb0[0], len(btrig))
        x_b0 = np.linspace(x_fb1[-1], x_fb0[0], len(btrig))
        y_b0 = np.linspace(y_fb1[-1], y_fb0[0], len(btrig))

        x = np.concatenate([x_b0[0:-1], x_fb0[0:-1], x_b1[0:-1], x_fb1[0:-1]]) * self.aline_width
        y = np.concatenate([y_b0[0:-1], y_fb0[0:-1], y_b1[0:-1], y_fb1[0:-1]]) * self.aline_width

        x, y = _rotfunc(x, y, rotation_rad)

        fb = np.zeros(len(x_fb0))
        line_trigger = np.concatenate([btrig, fb, btrig, fb])
        frame_start = np.zeros(len(line_trigger))
        frame_start[0:samples_on] = 1

        self._x = x
        self._y = y
        self._line_trigger = line_trigger
        self._frame_trigger = frame_start

        period_samples = samples_on + samples_off
        self._sample_rate = self.max_trigger_rate * period_samples

        self._pattern_rate = 1 / ((1 / self._sample_rate) * len(line_trigger))


class RoseScanPattern(LineScanPattern):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.p = None
        self.aline_width = None
        self.aline_per_b = None
        self.max_trigger_rate = None
        self.samples_on = None
        self.samples_off = None
        self.rotation_rad = None
        self.h = None

        if len(args) > 0:
            self.generate(*args, **kwargs)

    @property
    def dimensions(self):
        return [self.aline_per_b, self._p]

    @property
    def total_number_of_alines(self):
        return int(self.aline_per_b * self._p)

    def generate(self, p: int, aline_width: float, aline_per_b: int, max_trigger_rate: float,
                 samples_on: int = 1, samples_off: int = None, rotation_rad: float = 0, h: float = 5):
        """Generate a Rose pattern: a pattern consisting of any number of orthogonal B-scans.

        For a rose pattern with 2 B-scans, use a `Figure8ScanPattern`.

        Args:
            p (int): The number of B-scans the pattern should have.
            aline_width (float): The width of the A-line in units.
            aline_per_b (int): The number of A-lines in each B-line.
            max_trigger_rate (float): The maximum rate in Hz at which to toggle the line trigger signal. This is the constraint on imaging rate.
            samples_on (int): Optional. The number of trigger samples to drive high for each exposure. Default 2.
            samples_off (int): Optional. The number of trigger samples to drive low after each exposure. By default, equivalent to `samples_on`.
            rotation_rad (float): Optional. Rotates the scan. Default 0.
            h (float): Optional. 1 / H is the proportion of the B-line length to exclude from the scan for flyback. Default 5.
        """

        self.p = p
        self.aline_width = aline_width
        self.max_trigger_rate = max_trigger_rate
        self.samples_on = samples_on
        self.samples_off = samples_off
        self.rotation_rad = rotation_rad
        self.h = h

        p = int(p)
        if p % 2 == 0:
            raise ValueError("'p' must be an odd integer")
        else:
            k = p
            p0 = 0
            period = np.pi
        
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self.aline_width = aline_width
        self.aline_per_b = int(aline_per_b)
        self._p = int(p)

        b_frac = 0.2
        H = 5  # 1 / H is the proportion of the B-line length used for flyback in samples

        period_samples = samples_on + samples_off
        self._sample_rate = self.max_trigger_rate * period_samples

        b_trig = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self.aline_per_b)

        t = np.array([])
        lt = np.array([])

        period_p = period / k

        period_b = period_p * b_frac
        period_fb = period_p * (1 - b_frac)

        for i in range(p):  # Generate each petal

            p_start = p0 + i * period_p

            t_fb = _sigmoidspace(p_start, p_start + period_fb, int(1 / H * len(b_trig)), b=1)
            t_b = np.linspace(p_start + period_fb, p_start + period_fb + period_b, len(b_trig))

            t_p = np.concatenate([t_fb, t_b])

            trig_p = np.concatenate([np.zeros(len(t_fb)), b_trig])

            def xfunc(ts):
                return aline_width * np.cos(p * ts) * np.cos(ts)

            def yfunc(ts):
                return aline_width * np.cos(p * ts) * np.sin(ts)

            t = np.append(t, t_p)
            lt = np.append(lt, trig_p)

        t -= period_fb / 2

        x = xfunc(t)
        y = yfunc(t)

        x, y = _rotfunc(x, y, rotation_rad)

        self._x = x
        self._y = y
        self._line_trigger = lt

        ft = np.zeros(len(lt))
        ft[0:samples_on] = 1
        self._frame_trigger = ft
        self._pattern_rate = 1 / ((1 / self._sample_rate) * len(x))


class BidirectionalRasterScanPattern(LineScanPattern):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.alines = None
        self.blines = None
        self.max_trigger_rate = None
        self.exposure_fraction = None
        self.flyback_duty = None
        self.fov = None
        self.samples_on = None
        self.samples_off = None

        if len(args) > 0:
            self.generate(*args, **kwargs)

    @property
    def dimensions(self):
        """Returns [number of a-lines, number of b-lines]."""
        return [self.alines, self.blines]

    @property
    def total_number_of_alines(self):
        return int(self.alines * self.blines)

    def generate(self, alines: int, blines: int, max_trigger_rate: float, exposure_fraction: float = 0.9,
                 flyback_duty: float = 0.1, fov: list = None, samples_on: int = 2, samples_off: int = None):
        """Generate a bi-directionally scanned raster pattern.

        Args:
            alines (int): The number of A-lines in each B-line.
            blines (int): The number of B-lines.
            max_trigger_rate (float): The maximum rate at which to toggle the line trigger signal. This is the constraint on imaging rate.
            exposure_fraction (float): Optional. The fraction of the scan sweep to use for exposure. Default 0.9 90%.
            flyback_duty (float): Optional. The fraction of the scan sweep duration to dedicate to flyback. Default 0.1 10%.
            fov (list): Optional. If provided, scales the height and width of the pattern by these values.
            samples_on (int): Optional. The number of trigger samples to drive high for each exposure. Default 2.
            samples_off (int): Optional. The number of trigger samples to drive low after each exposure. By default, equivalent to `samples_on`.
        """

        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self.alines = int(alines)
        self.blines = int(blines)
        self.max_trigger_rate = max_trigger_rate
        self.exposure_fraction = exposure_fraction
        self.flyback_duty = flyback_duty
        self.fov = fov
        self.samples_on = samples_on
        self.samples_off = samples_off

        period_samples = samples_on + samples_off
        self._sample_rate = self.max_trigger_rate * period_samples

        bline_trig = np.tile(np.concatenate([np.ones(samples_on), np.zeros(samples_off)]), self.alines)
        bline_pad_len = int((1 - exposure_fraction) / 2 * len(bline_trig))
        bline_trig = np.concatenate([np.zeros(bline_pad_len), np.zeros(bline_pad_len), bline_trig])

        fast_axis_scan = np.array([])
        slow_axis_scan = np.array([])

        fast_range = (fov[0] + fov[0] * (1 - exposure_fraction)) / 2  # +/-
        slow_range = fov[1] / 2

        bpos = np.linspace(-slow_range, slow_range, self.blines)
        for i in range(blines):
            if i % 2 == 0:
                xs = np.linspace(fast_range, -fast_range, len(bline_trig) + 1)[1::]
            else:
                xs = np.linspace(-fast_range, fast_range, len(bline_trig) + 1)[1::]
            fast_axis_scan = np.append(fast_axis_scan, xs)

        for i in range(len(bpos) - 1):
            slow_axis_scan = np.append(slow_axis_scan,
                                       np.concatenate([np.ones(len(bline_trig) - bline_pad_len) * bpos[i],
                                                       np.linspace(bpos[i], bpos[i + 1], bline_pad_len + 1)[1::]]))
        slow_axis_scan = np.append(slow_axis_scan, np.ones(len(bline_trig) - bline_pad_len) * bpos[-1])

        slow_fb_len = int(len(fast_axis_scan) * flyback_duty)

        # flyback_slow = _sigmoidspace(slow_axis_scan[-1], slow_axis_scan[0], bline_pad_len)
        flyback_slow = np.linspace(slow_axis_scan[-1], slow_axis_scan[0], bline_pad_len + slow_fb_len)

        self._x = fast_axis_scan
        self._x = np.concatenate([self._x, np.ones(slow_fb_len) * self._x[-1]])

        self._y = np.concatenate([slow_axis_scan, flyback_slow])

        self._line_trigger = np.concatenate([np.tile(bline_trig, self.blines), np.zeros(slow_fb_len)])

        self._frame_trigger = np.zeros(len(self._line_trigger))
        self._frame_trigger[0:samples_on] = 1

        self._pattern_rate = 1 / (len(self._x) * (1 / self._sample_rate))


class RasterScanPattern(LineScanPattern):

    def __init__(self, *args, **kwargs):

        super().__init__()

        self.alines = None
        self.blines = None
        self.max_trigger_rate = None
        self.exposure_fraction = None
        self.flyback_duty = None
        self.fov = None
        self.samples_on = None
        self.samples_off = None
        self.samples_park = None
        self.samples_step = None
        self.rotation_rad = None
        self.fast_axis_step = None
        self.slow_axis_step = None
        self.aline_repeat = None
        self.trigger_blines = None

        if len(args) > 0:
            self.generate(*args, **kwargs)

    @property
    def dimensions(self):
        """Returns [number of a-lines, number of b-lines]."""
        return [self.alines, self.blines]

    @property
    def total_number_of_alines(self):
        return int((self.alines * self.aline_repeat) * self.blines)

    def generate(self, alines: int, blines: int, max_trigger_rate: float, flyback_duty: float = 0.2,
                 exposure_fraction: float = 0.8, fov: list = None, samples_on: int = 2, samples_off: int = None,
                 samples_park: int = 1, samples_step: int = 1, rotation_rad: float = 0, fast_axis_step: bool = False,
                 slow_axis_step: bool = False, aline_repeat: int = 1, trigger_blines: bool = False):
        """Generate a raster pattern.

        This raster pattern implementation supports stepped slow and fast axis scanning as well as repeated A-lines.

        Args:
            alines (int): The number of A-lines in each B-line.
            blines (int): The number of B-lines.
            max_trigger_rate (float): The maximum rate at which to toggle the line trigger signal. This is the constraint on imaging rate.
            exposure_fraction (float): Optional. The fraction of the scan sweep to use for exposure. Default 0.9 90%.
            flyback_duty (float): Optional. The fraction of the scan sweep duration to dedicate to flyback. Default 0.1 10%.
            fov (list): Optional. If provided, scales the height and width of the pattern by these values.
            samples_on (int): Optional. The number of trigger samples to drive high for each exposure. Default 2.
            samples_off (int): Optional. The number of trigger samples to drive low after each exposure. By default, equivalent to `samples_on`.
            samples_park (int): Optional. The number of samples to wait after holding galvos constant before triggering a repeated A-line. Default 1.
            samples_step (int): Optional. The number of samples to interpolate between step positions. Default 1.
            rotation_rad (float): Optional. Rotates the scan. Default 0.
            fast_axis_step (bool): Optional. If True, the fast axis is stepped to approximate a square FOV. Default False.
            slow_axis_step (bool): Optional. If True, the slow axis is stepped to approximate a square FOV. Default False.
            aline_repeat (int): Optional. If > 1, the number of A-lines to scan at each position. Default 1.
            trigger_blines (bool): Optional. If True, the frame trigger signal goes high for each B-line.
        """

        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self.alines = int(alines)
        self.blines = int(blines)
        self.max_trigger_rate = max_trigger_rate
        self.exposure_fraction = exposure_fraction
        self.flyback_duty = flyback_duty
        self.fov = fov
        self.samples_on = samples_on
        self.samples_off = samples_off
        self.samples_park = samples_park
        self.samples_step = samples_step
        self.rotation_rad = rotation_rad
        self.fast_axis_step = fast_axis_step
        self.slow_axis_step = slow_axis_step
        self.aline_repeat = aline_repeat
        self.trigger_blines = trigger_blines

        period_samples = samples_on + samples_off
        self._sample_rate = self.max_trigger_rate * period_samples

        if aline_repeat < 2:
            aline_repeat = 1
        else:
            fast_axis_step = True

        aline_trig = np.concatenate([np.ones(samples_on), np.zeros(samples_off)])
        if fast_axis_step is True:
            aline_trig = np.concatenate(
                [np.zeros(samples_park), np.tile(aline_trig, aline_repeat), np.zeros(samples_step)])

        bline_trig = np.tile(aline_trig, self.alines)
        bline_pad = np.zeros(int((len(bline_trig) * (1 - exposure_fraction)) / (2 * exposure_fraction)))
        bline_padded = np.concatenate([bline_pad, bline_trig, bline_pad])
        flyback_pad = np.zeros(int((len(bline_padded) * flyback_duty) / (1 - flyback_duty)))
        bline_trig = np.concatenate([flyback_pad, bline_padded])

        fast_axis_flyback = np.linspace(-fov[0] / 2, fov[0] / 2, len(flyback_pad) + 2)[1:-1]

        if fast_axis_step is False:
            fast_axis_scan = np.linspace(fov[0] / 2, -fov[0] / 2, len(bline_padded))
        else:
            fast_axis_scan = np.array([])
            positions = np.linspace(fov[0] / 2, -fov[0] / 2, self.alines)
            fast_axis_scan = np.append(fast_axis_scan, positions[0] * np.ones(len(bline_pad)))
            for i in range(len(positions) - 1):
                fast_axis_scan = np.append(fast_axis_scan, positions[i] * np.ones(len(aline_trig) - samples_step))
                fast_axis_scan = np.append(fast_axis_scan,
                                           np.linspace(positions[i], positions[i + 1], samples_step + 2)[1:-1])
            fast_axis_scan = np.append(fast_axis_scan, positions[-1] * np.ones(len(aline_trig) + len(bline_pad)))
        bline_scan = np.concatenate([fast_axis_flyback, fast_axis_scan])

        line_trig = np.tile(bline_trig, self.blines)

        if self.alines > 1:
            x = np.tile(bline_scan, self.blines)
        else:
            x = np.zeros(len(line_trig))

        if trigger_blines:
            f = np.zeros(len(bline_trig))
            f[0:samples_on] = 1
            frame_trig = np.tile(f, self.blines)
        else:
            frame_trig = np.zeros(len(line_trig))
            frame_trig[0:samples_on] = 1

        if self.blines > 1:

            slow_axis_flyback = fast_axis_flyback

            if slow_axis_step is False:
                slow_axis_scan = np.linspace(-fov[0] / 2, fov[0] / 2, len(x) - len(fast_axis_flyback) + 2)[1:-1]
                y = np.concatenate([slow_axis_scan, slow_axis_flyback[::-1]])
            else:
                slow_axis_scan = np.array([])
                positions = np.linspace(fov[0] / 2, -fov[0] / 2, self.alines)
                for i in range(len(positions) - 1):
                    slow_axis_scan = np.append(slow_axis_scan, positions[i] * np.ones(len(fast_axis_scan)))
                    slow_axis_scan = np.append(slow_axis_scan,
                                               np.linspace(positions[i], positions[i + 1], len(fast_axis_flyback) + 2)[
                                               1:-1])
                slow_axis_scan = np.append(slow_axis_scan, positions[-1] * np.ones(len(fast_axis_scan)))
                y = np.concatenate([slow_axis_scan, slow_axis_flyback])
        else:
            y = np.zeros(len(x))

        self._x = x
        self._y = y
        self._line_trigger = line_trig
        self._frame_trigger = frame_trig

        self._x, self._y = _rotfunc(self._x, self._y, rotation_rad)

        self._pattern_rate = 1 / (len(self._x) * (1 / self._sample_rate))


class BlineRepeatedRasterScan(LineScanPattern):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.alines = None
        self.blines = None
        self.bline_repeat = None
        self.max_trigger_rate = None
        self.exposure_fraction = None
        self.flyback_duty = None
        self.fov = None
        self.samples_on = None
        self.samples_off = None
        self.samples_park = None
        self.samples_step = None
        self.rotation_rad = None
        self.slow_axis_step = None

        if len(args) > 0:
            self.generate(*args, **kwargs)

    @property
    def dimensions(self):
        """Returns [number of a-lines, number of b-lines]."""
        return [self.alines, self.blines]

    @property
    def total_number_of_alines(self):
        return int(self.alines * (self.blines * self.bline_repeat))
    
    def generate(self, alines: int, blines: int, max_trigger_rate: float, bline_repeat: int = 2,
                 flyback_duty: float = 0.2, exposure_fraction: float = 0.8, fov: list = None,
                 samples_on: int = 2, samples_off: int = None, samples_park: int = 1, samples_step: int = 1,
                 rotation_rad: float = 0, slow_axis_step: bool = False):
        """Generate a raster pattern with repeated B-lines.

        This raster pattern implementation supports stepped slow axis scanning.

        Args:
            alines (int): The number of A-lines in each B-line.
            blines (int): The number of B-lines.
            max_trigger_rate (float): The maximum rate at which to toggle the line trigger signal. This is the constraint on imaging rate.
            bline_repeat (int): The number of B-lines to scan at each position.
            exposure_fraction (float): Optional. The fraction of the scan sweep to use for exposure. Default 0.9 90%.
            flyback_duty (float): Optional. The fraction of the scan sweep duration to dedicate to flyback. Default 0.1 10%.
            fov (list): Optional. If provided, scales the height and width of the pattern by these values.
            samples_on (int): Optional. The number of trigger samples to drive high for each exposure. Default 2.
            samples_off (int): Optional. The number of trigger samples to drive low after each exposure. By default, equivalent to `samples_on`.
            samples_park (int): Optional. The number of samples to wait after holding galvos constant before triggering a repeated A-line. Default 1.
            samples_step (int): Optional. The number of samples to interpolate between step positions. Default 1.
            rotation_rad (float): Optional. Rotates the scan. Default 0.
            slow_axis_step (bool): Optional. If True, the slow axis is stepped to approximate a square FOV. Default False.
        """
        self.bline_repeat = bline_repeat

        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self.alines = int(alines)
        self.blines = int(blines)
        self.max_trigger_rate = max_trigger_rate
        self.exposure_fraction = exposure_fraction
        self.flyback_duty = flyback_duty
        self.fov = fov
        self.samples_on = samples_on
        self.samples_off = samples_off
        self.samples_park = samples_park
        self.samples_step = samples_step
        self.rotation_rad = rotation_rad
        self.slow_axis_step = slow_axis_step

        period_samples = samples_on + samples_off
        self._sample_rate = self.max_trigger_rate * period_samples

        aline_trig = np.concatenate([np.ones(samples_on), np.zeros(samples_off)])
    
        bline_trig = np.tile(aline_trig, self.alines)
        bline_pad = np.zeros(int((len(bline_trig) * (1 - exposure_fraction)) / (2 * exposure_fraction)))
        bline_padded = np.concatenate([bline_pad, bline_trig, bline_pad])
        flyback_pad = np.zeros(int((len(bline_padded) * flyback_duty) / (1 - flyback_duty)))
        bline_trig = np.concatenate([bline_padded, flyback_pad])
        
        bline_scan = np.linspace(fov[0] / 2, -fov[0] / 2, len(bline_padded))
        bline_flyback = np.linspace(-fov[0] / 2, fov[0] / 2, len(flyback_pad) + 2)[1:-1]
        
        bline = np.concatenate([bline_scan, bline_flyback])
        
        bline_trig_rpt = np.tile(bline_trig, bline_repeat)
        bline_scan_rpt = np.tile(bline, bline_repeat)
        
        fast_axis_scan = np.tile(bline_scan_rpt, self.alines)
        line_trig = np.tile(bline_trig_rpt, self.alines)

        if self.blines > 1:
            slow_axis_positions = np.linspace(fov[0] / 2, -fov[0] / 2, self.blines)
            slow_fb_len = int(len(flyback_pad) + len(bline_pad) / 2)
            slow_axis_scan = np.array([])
            for i, pos in enumerate(slow_axis_positions[:-1]):
                slow_axis_scan = np.append(slow_axis_scan, np.ones(len(bline_trig_rpt) - slow_fb_len) * pos)
                slow_axis_scan = np.append(slow_axis_scan, np.linspace(pos, slow_axis_positions[i + 1], slow_fb_len + 2)[1:-1])
            slow_axis_scan = np.append(slow_axis_scan, np.ones(len(bline_trig_rpt) - slow_fb_len) * slow_axis_positions[-1])
            slow_axis_scan = np.append(slow_axis_scan, np.linspace(slow_axis_positions[-1], slow_axis_positions[0], slow_fb_len + 2)[1:-1])
        else:
            slow_axis_scan = np.zeros(len(fast_axis_scan))
        
        frame_trig = np.zeros(len(line_trig))
        frame_trig[0:samples_on] = 1
        
        self._x = fast_axis_scan
        self._y = slow_axis_scan
        self._line_trigger = line_trig
        self._frame_trigger = frame_trig
        self._pattern_rate = 1 / ((1 / self._sample_rate) * len(self._x))

        self._x, self._y = _rotfunc(self._x, self._y, rotation_rad)
