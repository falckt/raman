# Copyright (c) 2020 alchem0x2A
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copied from
# https://github.com/alchem0x2A/py-wdf-reader/blob/4090b71d1ac7d5cf9628f1808df866b7c552be98/renishawWiRE/types.py

# Declaration of DATA types
from enum import IntEnum


class MeasurementType(IntEnum):
    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3


class ScanType(IntEnum):
    Unspecified = 0
    Static = 1
    Continuous = 2
    StepRepeat = 3
    FilterScan = 4
    FilterImage = 5
    StreamLine = 6
    StreamLineHR = 7
    PointDetector = 8


class UnitType(IntEnum):
    Arbitrary = 0
    RamanShift = 1
    Wavenumber = 2
    Nanometre = 3
    ElectronVolt = 4
    Micron = 5
    Counts = 6
    Electrons = 7
    Millimetres = 8
    Metres = 9
    Kelvin = 10
    Pascal = 11
    Seconds = 12
    Milliseconds = 13
    Hours = 14
    Days = 15
    Pixels = 16
    Intensity = 17
    RelativeIntensity = 18
    Degrees = 19
    Radians = 20
    Celsius = 21
    Fahrenheit = 22
    KelvinPerMinute = 23
    FileTime = 24


class DataType(IntEnum):
    Arbitrary = 0
    Frequency = 1
    Intensity = 2
    X = 3
    Y = 4
    Z = 5
    R = 6
    Theta = 7
    Phi = 8
    Temperature = 9
    Pressure = 10
    Time = 11
    Derived = 12
    Polarization = 13
    FocusTrack = 14
    RampRate = 15
    Checksum = 16
    Flags = 17
    ElapsedTime = 18
