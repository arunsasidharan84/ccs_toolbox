# -*- coding: utf-8 -*-
"""
#############################################################################
Functions to read files from EMBLA or REMLOGIC PSG data format

#############################################################################
DISCLAIMER:         The codes as part of eegBidsCreator, 
                    Copyright (c) 2018-2019, University of Liège
                    GNU General Public License

Original author:    Nikita Beliy, Liege University https://www.uliege.be 
                    Email: Nikita.Beliy@uliege.be
                    
Modified author:    Jun 2024; Arun Sasidharan, CCS, NIMHANS

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE.
#############################################################################
"""

# Import libraries
import math
import struct
import io
from datetime import datetime
from datetime import timedelta
import logging

Logger = logging.getLogger("EmblaChannel")




class EbmEvent(object):
    """Structure for Event-type data"""
    __slots__ = ["LocationIdx", 
                "AuxDataID", 
                "GroupTypeIdx", 
                "StartTime", 
                "TimeSpan", 
                "ScoreID", 
                "CreatorID",
                "EventID"]

    def __str__(self):
        return self.EventID+" {} {} {} {} {} {} {}".format(self.LocationIdx, self.AuxDataID,
            self.GroupTypeIdx, self.StartTime, self.TimeSpan, self.ScoreID, self.CreatorID)

    def __repr__(self):
        return self.EventID

    def __init__(self, data):
        if len(data) != 32+78+2:
            raise Exception("Event data size is not 78")
        # [0:2] Ushort(H) location index
        # [2:4] Ushort(H) Aux. data
        # [4:8] Uint(I)   GroupType
        # [8:16]double(d) StartTime
        # [16:32]double(d)TimeSpan
        # [32:36]Uint(I)  ScoreID
        # [36:37]char(c)  CreatorID 
        # [37:40] (x)     Unused
        # [32:110] utf_16_le EventID
        # [110:112](x)    Unused 
        parced = struct.unpack("<HHIddIbxxx", data[0:32])
        self.EventID = data[32:32+78].decode('utf_16_le')
        self.LocationIdx = parced[0]
        self.AuxDataID   = parced[1]
        self.GroupTypeIdx= parced[2]
        self.StartTime   = parced[3]
        self.TimeSpan    = parced[4]
        self.ScoreID     = parced[5]
        self.CreatorID   = parced[6]


def ReadEvents(data):
    """Reads and extracts the list of events from data, returns the list of Event objects"""
    if len(data)%112 != 0:
        raise Exception("Data size is not multiple of 112, events record is corrupted")
    array = []
    for pos in range(0, len(data), 112):
        array.append(EbmEvent(data[pos:pos+112]))
    return array

def ReadEventsStartTime(data):
    """Reads and extracts list of starttime from data, return the list of datetime objects"""
    if len(data)%12 != 0:
        raise Exception("Data size is not multiple of 112, events record is corrupted")
    array = []
    for pos in range(0, len(data), 12):
        y,m,d,h,minute,sec,usec = struct.unpack("<HBBBBBxI", data[pos:pos+12])
        array.append(datetime(y,m,d,h,minute, sec, usec))
    return array
    


Types = {0: "any", 1:"rectangle", 2:"point", 3:"string", 4:"bool", 5:"byte",
         6:"word", 7:"dword", 8:"long", 9:"float", 10:"double", 11:"adouble",
         12:"reference", 13: "parcel", 14:"time", 15:"timespan", 16:"void",
         17:"action", 18:"specifier", 19:"aword", 20:"abyte",
         1000:"resend", 2000:"events", 2001:"evsttime"}


class Parcel(object):
    """Generic ontainer for a set of data"""
    __slots__ = ["__stream",   # Stream containing data
                 "__size",     # Total size of container in bits
                 "__type",     # Type of the container
                 "__version",  # Version of container
                 "__entries",  # List of contents of container
                 "__start",    # position of the first bit of container
                 "__name",     # Name of the parcel, default est '/'
                 "__parent"    # Parent parcel
                 ] 

    def __str__(self):
        return "Parcel <{0}>, starting at {1}, of size {2},"\
               "containing {3} objects"\
               .format(self.__name, hex(self.__start),
                       hex(self.__size), len(self.__entries))

    def __repr__(self):
        return "{0}: \n{1}".format(self.pwd(), self.__entries)

    def __init__(self, Stream, Name=None, Start=None, Parent=None):
        self.__stream = Stream  # How to test if stream is readable
        if Start is None:
            self.__start = Stream.tell()
        else:
            self.__start = Start
            Stream.seek(Start)
        if Name is None:
            self.__name = "//"
        else:
            self.__name = Name
        self.__parent = Parent

        # [0:2] Ushort(H) version
        # [2:6] Uinit(I)  size
        # [6:8] Ushort(H) type
        head = Stream.read(8)
        self.__version, self.__size, self.__type = struct.unpack("<HIH",head)
        self.__entries = []

        while Stream.tell() < self.__size + self.__start:
            self.__entries.append(Entry(Stream,Parent=self))

        if Stream.tell() != self.__size + self.__start:
            raise Exception("Declared size {0} mismatch "
                            "number of readed bytes {1}"
                            .format(hex(self.__size),
                                    hex(Stream.tell() - self.__start))
                            )

    def pwd(self):
        """Returns the path to this container"""
        string = self.__name
        p = self.__parent
        while p is not None:
            string = p.__name + "/" + string
            p = p.parent()
        return string

    def ls(self, title=""):
        """Returns a list of wrappers (entries) in this container,
        matching the given title, if title is '', then full list 
        of wrappers is returned."""
        res = []
        for en in self.__entries:
            if title == ""\
               or en.name() == title\
               or en.name() == (title + '\0'):
                res.append(en)
        return res

    def get(self, title, index=0):
        """Return data from a wrapper given its name and index"""
        count = 0
        for en in self.__entries:
            if en.name() == title or en.name() == (title + '\0'):
                if index == count: return en.read()
                count = count + 1
        raise Exception("Index {}/{} out of range "
                        "for container {}".format(title,count, self.__name))

    def getlist(self, title=""):
        """Return a list of data from wrappers matching the given title"""
        res = []
        for en in self.__entries:
            if title == ""\
               or en.name() == title\
               or en.name() == (title + '\0'):
                res.append(en.read())
        return res

    def parent(self):
        """Return the parent of this parcel"""
        return self.__parent

    def ls_r(self, level=0):
        """Iteratively printout the contents of this Parcel
        and its sub-parcels."""
        offset = ""
        marker = '\t'
        for i in range(0, level):
            offset = offset + marker
        print(offset + str(self))

        offset = offset + marker
        for c in self.__entries:
            if c.type() == 13:
                c.read().ls_r(level + 1)
            else:
                print(offset + str(c) + "<" + str(c.read()) + ">")



class Entry(object):
    """A wrapper of a generic data"""
    __slots__ = ["__size", "__dsize", "__type", "__stype", "__readed",
                 "__data", "__start", "__name", "__parent", "__stream"]

    def __str__(self):
        string = "{0}({1})".format(self.__stype, self.__type)
        if not self.__readed:
            string = string + '*'
        return string + "<{0}>".format(self.__name)

    def __repr__(self):
        return "{0}, starting at {1}, of size {2}({3})"\
                .format(str(self), hex(self.__start),
                        hex(self.__size), hex(self.__dsize))

    def __init__(self, Stream, Parent, Start=None):
        if Start is None:
            self.__start = Stream.tell()
        else:
            self.__start = Start
            Stream.seek(Start)
        self.__stream = Stream
        self.__parent = Parent
        self.__readed = False

        # [0:4]     int(i)      size
        # [4:8]     int(i)      data size
        # [8:10]    Ushort(H)   type
        # [10:12]   short(h)    unused
        head = Stream.read(10)
        Stream.seek(2,1)
        self.__size, self.__dsize, self.__type = struct.unpack("<iiH", head)
        if self.__type not in Types:
            self.__type = 0
        self.__stype = Types[self.__type]

        self.__data = None
        Stream.seek(self.__dsize,1)
        self.__name = Stream.read(self.__size - self.__dsize - 12)\
                            .decode("ascii").strip('\0')

    def read(self):
        """Read and returns data, formatted folowing the type"""
        if not self.__readed:
            Stream = self.__stream
            Stream.seek(self.__start + 12, 0)
            if self.__type == 4:
                data = (Stream.read(self.__dsize) != 0) 
            elif self.__type == 3:
                data = Stream.read(self.__dsize).decode('1252').strip('\0')
            elif self.__type == 6:
                data = struct.unpack("<H",Stream.read(self.__dsize))[0]
            elif self.__type == 7:
                data = struct.unpack("<I",Stream.read(self.__dsize))[0]
            elif self.__type == 8:
                data = struct.unpack("<l",Stream.read(self.__dsize))[0]
            elif self.__type == 13:
                data = Parcel(Stream, Name=self.__name,
                              Start=Stream.tell(), Parent=self.__parent)
            elif self.__type == 15:
                # Readed data contains 16 bits, if 8 are seconds,
                # 4 are miliseconds, the remaining 2 are always 
                # composed of \xbbT\x06t
                # If timespan contains only seconds, lenth is 8(long long) 
                d = Stream.read(self.__dsize)
                if (len(d) == 16):
                    sec, mils = struct.unpack("<qi", d[0:12])
                    data = float(sec) + float(mils) / 1000
                elif (len(d) == 8):
                    sec = struct.unpack("<q", d[0:8])
                    data = float(sec) 
                else:
                    raise Exception("Unable to parce timespan from string {}"
                                    .format(d))
            elif self.__type == 2000:
                data = ReadEvents(Stream.read(self.__dsize))
            elif self.__type == 2001:
                data = ReadEventsStartTime(Stream.read(self.__dsize))
            else:
                data = Stream.read(self.__dsize)
        else :
            self._readed = True       
        return data

    def type(self): return self.__type

    def name(self): return self.__name




def ReplaceInField(In_string, Void="", ToReplace=None):
    """Find and replace strings in ToReplace tuple 
    in input string. If input string is empty, returns 
    Void string"""

    if not isinstance(In_string, str) or not isinstance(Void, str):
        raise TypeError("ReplaceInField: In_string and Void must be a string")

    if ToReplace is not None:
        if not isinstance(ToReplace, tuple)\
           or len(ToReplace) != 2\
           or not isinstance(ToReplace[0], str)\
           or not isinstance(ToReplace[1], str):
            raise TypeError("ReplaceInField: "
                            "ToReplace must be either None or (str,str)")

    if In_string == "" :
        return Void

    if ToReplace is not None:
        return In_string.replace(ToReplace[0], ToReplace[1])
    return In_string



class GenChannel():
    """An intendent virtual class serving as parent to other,
    format specific channel classes."""

    __base_slots__ = [
        "_scale", "_offset",
        "_unit", "_magnitude",
        "_physMin", "_physMax",
        "_digMin", "_digMax",
        "_seqStartTime",
        "_seqSize",
        "_frequency",
        "_name",
        "_type",
        "_description",
        "_reference",
        "_id",

        "_startTime",
        "_frMultiplier",
        "_baseChannel",
        "BIDSvalues"]


    __slots__ = __base_slots__

    def __copy__(self, source):
        if not isinstance(source, GenChannel):
            raise TypeError(": Source object must be a daughter of "
                            + self.__class__.__name__)
        for f in self.__base_slots__:
            setattr(self, f, getattr(source, f))
        self._baseChannel = source

    "Min and max values for an signed short integer"
    _MAXSHORT = 32767
    _MINSHORT = -32768

    """Dictionary of standard SI prefixes, as defined in BIDS"""
    _SIprefixes = {24:'Y', 21:'Z', 18:'E', 15:'P', 12:'T', 9:'G',
                   6:'M', 3:'k', 2:'h', 1:'da', 0:'', -1:'d', 
                   -2:'c', -3:'m', -6:'µ', -9:'n', -12:'p', 
                   -15:'f', -18:'a', -21:'z', -24:'y'}

    """Inverted dictionary of standard SI prefixes, as defined in BIDS"""
    _SIorders = {'Y':24, 'Z':21, 'E':18, 'P':15, 'T':12,'G':9, 
                 'M': 6,'k': 3,'h': 2,'da': 1, 0:'', 'd':-1, 
                 'c':-2, 'm':-3, 'µ':-6, 'n':-9, 'p':-12, 
                 'f':-15, 'a':-18, 'z':21, 'y':-24}

    _BIDStypes = ["AUDIO", "EEG", "HEOG", "VEOG", "EOG", "ECG", "EKG",
                  "EMG", "EYEGAZE", "GSR", "PUPIL", "REF", "RESP", 
                  "SYSCLOCK", "TEMP", "TRIG", "MISC"]

    def __init__(self):
        super(GenChannel, self).__init__()
        self._scale = 1.
        self._offset = 0.
        self._unit = ""
        self._magnitude = 0
        self._physMin = self._MINSHORT
        self._physMax = self._MAXSHORT
        self._digMin = self._MINSHORT
        self._digMax = self._MAXSHORT

        self._frequency = 1
        self._name = ""
        self._type = ""
        self._description = ""
        self._reference = ""

        self._seqStartTime = []
        self._seqSize = []

        self._startTime = datetime.min
        self._frMultiplier = 1

        self._baseChannel = self
        self._id = -1
        self.BIDSvalues = dict()

    def GetId(self): return self._id

    def SetId(self, Id): self._id = Id

    def GetScale(self): return self._scale

    def GetOffset(self): return self._offset

    def GetPhysMax(self): return self._physMax

    def GetPhysMin(self): return self._physMin

    def GetDigMax(self): return self._digMax

    def GetDigMin(self): return self._digMin

    def SetScale(self, scale, offset=0):
        """Defining new scale and offset. Physical minimum and maximum
        are recalculated accordingly."""
        if not (isinstance(scale, int) or isinstance(scale, float)):
            raise TypeError("Scale must be integer or float value")
        if not (isinstance(offset, int) or isinstance(offset, float)):
            raise TypeError("Offset must be integer or float value")

        self._scale = scale
        self._offset = offset
        self._physMin = self._fromRaw(self._digMin)
        self._physMax = self._fromRaw(self._digMax)

    def SetPhysicalRange(self, minimum, maximum):
        """Defining new physical extrema.
        The scale and offset are recalculated."""

        if not (isinstance(minimum, int) or isinstance(minimum, float)):
            raise TypeError("Physical mimimum must be "
                            "integer or float value")

        if not (isinstance(maximum, int) or isinstance(maximum, float)):
            raise TypeError("Physical maximum must be "
                            "integer or float value")

        if minimum >= maximum:
            raise ValueError("Physical minimum must be "
                             "lower than maximum")
        self._physMin = minimum
        self._physMax = maximum
        self._calculateScale()

    def SetDigitalRange(self, minimum, maximum):
        """Defining new digital extrema.
        The scale and offset are recalculated."""
        if not (isinstance(minimum, int)):
            raise TypeError("Digital mimimum must be integer value")
        if not (isinstance(maximum, int)):
            raise TypeError("Digital maximum must be integer value")
        if minimum >= maximum:
            raise ValueError("Digital minimum must be lower than maximum")
        if minimum < self._MINSHORT:
            raise ValueError("Digital minimum must be "
                             "greater than minimum short value")
        if maximum > self._MAXSHORT:
            raise ValueError("Digital maximum must be "
                             "greater than maximum short value")

        self._digMin = minimum
        self._digMax = maximum
        self._calculateScale()

    def _calculateScale(self):
        """Recalculates scale and offset 
        according to physical and digital extrema."""
        self._scale = (self._physMax - self._physMin)\
            / (self._digMax - self._digMin)
        self._offset = self._physMin - self._scale * self._digMin

    def FromRaw(self, value):
        """Transform raw short integer value to the measured one.
        Input must be integer and in Digital range."""
        if not (isinstance(value, int)):
            raise TypeError(self.__class__ + ": Value must be an integer")
        if value > self._digMax or value < self._digMin:
            raise Exception(self.__class__
                            + ": value " + str(value) + " out of the range ["
                            + str(self._digMin) + ", "
                            + str(self._digMax) + "]")
        return self._fromRaw(value)

    def _fromRaw(self, value):
        """Transform raw short integer value to the measured one.
        No checks in value performed."""
        return value * self._scale + self._offset

    def ToRaw(self, value):
        """Transform measured value to raw short integer.
        Input must be float and in Physical range."""
        if not (isinstance(value, int) or isinstance(value, float)):
            raise TypeError(self.__class__
                            + ": Value must be an integer or float")
        if value > self._physMax or value < self._physMin:
            raise Exception(self.__class__
                            + ": value " + str(value) + " out of the range ["
                            + str(self._physMin) + ", "
                            + str(self._physMax) + "]")
        return self._toRaw(value)

    def _toRaw(self, value):
        """Transform measured value to raw short integer one.
        No checks in value performed."""
        return int((value - self._offset) / self._scale + 0.5)  

    def SetName(self, name):
        if not (isinstance(name, str)):
            raise TypeError(self.__class__
                            + ": Name must be a string")
        self._name = name

    def GetName(self, Void="", ToReplace=None):
        return ReplaceInField(self._name, Void, ToReplace)

    def SetType(self, name):
        if not (isinstance(name, str)):
            raise TypeError(self.__class__ + ": Type must be a string")
        self._type = name

    def GetType(self, Void="", ToReplace=None):
        return ReplaceInField(self._type, Void, ToReplace)

    def BidsifyType(self):
        """Replace the type of channel by a BIDS supported type.
        Matching is performed by searching string from _BIDStypes
        in original type. If not found, a MISC type is attributed.
        """
        if self._type == "EKG":
            self._type = "ECG"
        if self._type in self._BIDStypes:
            # Type already BIDS complient
            return

        bids_type = "MISC"
        for t in self._BIDStypes:
            if t in self._type:
                bids_type = t
                break
        if bids_type == "EKG":
            bids_type = "ECG"
        Logger.debug("{}:Changing type from {} to {}".format(
                     self._name, self._type, t))
        self._type = t

    def SetDescription(self, name):
        if not (isinstance(name, str)):
            raise TypeError(self.__class__ + " : Description must be a string")
        self._description = name

    def GetDescription(self, Void="", ToReplace=None):
        return ReplaceInField(self._description, Void, ToReplace)

    def SetReference(self, name):
        if not (isinstance(name, str)):
            raise TypeError(self.__class__ + ": Reference must be a string")
        self._reference = name

    def GetReference(self, Void="", ToReplace=None):
        return ReplaceInField(self._reference, Void, ToReplace)

    def SetUnit(self, unit):
        if not (isinstance(unit, str)):
            raise TypeError(self.__class__ + ": Unit must be a string")
        self._unit = unit

    def GetUnit(self, wMagnitude=True, Void=""):
        if wMagnitude:
            if self._unit == "":
                if self._magnitude == 0:
                    return Void
                else:
                    return "x10^" + str(self._magnitude)

            if self._magnitude in self._SIprefixes:
                return self._SIprefixes[self._magnitude] + self._unit
            else:
                magn = min(self._SIprefixes.keys(),
                           key=lambda k: abs(k - self._magnitude))
                return "x10^" + str(self._magnitude - magn)\
                    + " " + self._SIprefixes[magn] + self._unit
        else:
            if self._unit == "": return Void
            else: return self._unit    

    def SetMagnitude(self, magn):
        """Setting the magnitude to the measured value.
        This affects scale, offset and physical range."""
        if not (isinstance(magn, int)):
            raise TypeError(self.__class__ + ": magnitude must be an integer")
        self._scale /= 10**(magn + self._magnitude)
        self._offset /= 10**(magn + self._magnitude)
        self._physMin /= 10**(magn + self._magnitude)
        self._physMax /= 10**(magn + self._magnitude)
        self._magnitude = magn

    def OptimizeMagnitude(self):
        magn = math.log10(self._scale) + self._magnitude
        if magn < 0 : 
            magn = int(math.floor(magn) / 3 - 0.5 + 1) * 3
        else :
            magn = int(math.ceil(magn) / 3 + 0.5 - 1) * 3
        self.SetMagnitude(magn)

    def GetFrequency(self):
        return self._frequency

    def SetFrequency(self, freq):
        if not isinstance(freq, int):
            raise TypeError("Frequency must be an integer representing Hz")
        self._frequency = freq

    def GetMagnitude(self):
        return self._magnitude 

    """Functions related to the sequences, i.e.
    unenturupted periods of data-taking."""

    def GetNsequences(self):
        """Returns number of interupted sequences"""
        return len(self._seqStartTime)

    def GetSequenceStart(self, seq=0):
        """Returns the start time of the ith sequence"""
        return self._seqStartTime[seq]

    def GetSequenceEnd(self, seq=0):
        return self._seqStartTime[seq]\
            + timedelta(seconds=self.GetSequenceDuration(seq))

    def GetSequenceSize(self, seq=0):
        """Returns the size (number of measurements) in given sequence"""
        return self._seqSize[seq]

    def GetSequenceDuration(self, seq=0):
        """Returns the time span (in seconds) of given sequence"""
        return self._seqSize[seq] / self._frequency

    def SetStartTime(self, start):
        if not isinstance(start, datetime):
            raise TypeError("StartTime must be a datetime object")
        self._startTime = start

    def GetStartTime(self):
        return self._startTime

    def SetFrequencyMultiplyer(self, frMult):
        if not isinstance(frMult,int):
            raise TypeError("Frequency multiplyer must be a positif integer")
        if frMult <= 0:
            raise ValueError("Frequency multiplyer must be positif")
        self._frMultiplier = frMult

    def GetFrequencyMultiplyer(self):
        return self._frMultiplier

    """
    Functions related to the index of a partiular data points.
    Each point can be indexed by global index, common to all channels,
    given the common time origin, and common frequency, or by local index
    defined its position in its sequence.
    """
    def GetGlobalIndex(self, point, sequence,
                       StartTime=None, freqMultiplier=None):
        """
        Converts local index from a sequence to a global one
        Do not check if given point is actually exists in sequence

        It uses round to get the index if StartTime is not synchronized
        with sequence time

        Parameters
        ----------
        point : int
            local index of data point in sequence
        sequence : int
            index of a sequence
        StartTime : datetime, optional
            the time from which the global index should be calculated
            if not defined, channel's start time is used
        freqMultiplier : int, optional
            frequency multiplier used to convert from local channel
            frequency to a common  one. If not set, channel defined
            multiplier is used

        Returns
        -------
        int
            the global index

        Raises
        ------
        IndexError
            if sequence is out of range
        """
        if StartTime is None:
            StartTime = self._startTime
        if freqMultiplier is None:
            freqMultiplier = self._frMultiplier
        if not isinstance(StartTime, datetime):
            raise TypeError("StartTime must be datetime object")
        if not (isinstance(freqMultiplier,int) or freqMultiplier > 0):
            raise TypeError("freqMultiplier must be a positive integer") 
        if not isinstance(sequence, int) or not isinstance(point, int):
            raise TypeError("sequence and point must be integer")
        if sequence < 0 or sequence >= len(self._seqStartTime):
            raise IndexError("sequence (" + str(sequence) 
                             + ")is out of the range")

        time = self._getTime(point, self._seqStartTime[sequence], 1) 
        index = (time - StartTime).total_seconds()\
            / (self._frequency * freqMultiplier)
        return round(index)

    def GetLocalindex(self, point, StartTime=None, freqMultiplier=None):
        """
        Converts global index to a local one. returns 
        a tuple (index, sequence).
        If point happens before start of data, sequence will be -1
        If point outside of sequence size, index will be -1

        Parameters
        ----------
        point : int
            global index of data point
        StartTime : datetime, optional
            starting time for global index. If not set, the channel's
            defined will be used
        freqMultiplier : int, optional
            frequency multiplier for calculation of global frequency
            If not set, channel's defined one will be used

        Returns
        -------
        (int, int)
            the tuple (index, sequence) of corresponding local index
        """
        if StartTime is None:
            StartTime = self._startTime
        if freqMultiplier is None:
            freqMultiplier = self._frMultiplier
        if not isinstance(point, int):
            raise TypeError("point must be int")
        if not isinstance(StartTime, datetime):
            raise TypeError("StartTime must be datetime object")
        if not (isinstance(freqMultiplier,int) or freqMultiplier > 0):
            raise TypeError("freqMultiplier must be a positive integer") 
        time = self._getTime(point, StartTime, freqMultiplier)
        return self._getLocalIndex(time)

    def GetTimeFromIndex(self, point, sequence=None, 
                         StartTime=None, freqMultiplier=None):
        """
        Converts local or global index to a corresponding time.
        Parameters sequence and (Starttime, frqMultiplier) are
        mutually exclusive as they used to distinguish between
        local and global index

        Parameters
        ----------
        point : int
            index to data point
        sequence : int, optional
            index to a sequence. If set, index will be concidered 
            as local
        StartTime : datetime, optional
            the start time for a global index. If not set and 
            index is global, channel-defined start time will 
            be used
        freqMultiplier : int, optional
            frequency multiplier used for calculating global frequency.
            If not set, channel-defined will be used

        Returns
        -------
        datetime
            time corresponding to current index

        Raises
        ------
        TypeError
            if passed parameters are of invalid type
        RuntimeError
            if passed parameters are incompatible
        IndexError
            if sequence index is invalid
        """
        if sequence is not None:
            if StartTime is not None or freqMultiplier is not None:
                raise RuntimeError("parameters sequence and (StartTime, "
                                   "freqMultiplier) are mutually exclusive")
        if StartTime is None:
            StartTime = self._startTime
        if freqMultiplier is None:
            freqMultiplier = self._frMultiplier

        if not isinstance(point, int):
            raise TypeError("point must be int")
        if not isinstance(sequence, int):
            raise TypeError("sequence must be int")
        if not isinstance(StartTime, datetime):
            raise TypeError("StartTime must be datetime")
        if not isinstance(freqMultiplier, int):
            raise TypeError("freqMultiplier must be int")

        if sequence is None:
            return self._getTime(point, StartTime, freqMultiplier)
        else:
            if sequence < 0 or sequence > self.GetNsequences():
                raise IndexError("sequence out of range")
            return self._getTime(point, self._seqStartTime[sequence], 1)

    def GetLocalIndexFromTime(self, time):
        """
        Converts time to local index. If time is before 
        the first sequence, returned sequence is set to -1.
        If there no data point at given time, returned index
        will be set to -1

        Parameters
        ----------
        time : datetime

        Returns
        -------
        (int, int)
            a tuple (index, sequence)

        Raises
        ------
        TypeError
            if passed parameter is of invalid type
        """
        if not isinstance(time, datetime):
            raise TypeError("time must be datetime object")
        return self._getLocalIndex(time)

    def GetGlobalIndexFromTime(self, time, 
                               StartTime=None, freqMultiplier=None):
        """
        Converts time to global index. 

        Parameters
        ----------
        time : datetime

        Returns
        -------
        (int, int)
            a tuple (index, sequence)

        Raises
        ------
        TypeError
            if passed parameter is of invalid type
        """
        if StartTime is None:
            StartTime = self._startTime
        if freqMultiplier is None:
            freqMultiplier = self._frMultiplier
        if not isinstance(time, datetime):
            raise TypeError("time must be datetime object")
        if not isinstance(StartTime, datetime):
            raise TypeError("StartTime must be datetime")
        if not isinstance(freqMultiplier, int):
            raise TypeError("freqMultiplier must be int")
        dt = (time - StartTime).total_seconds()
        return round(dt * self._frequency * freqMultiplier)

    def GetValue(self, point, default=0, 
                 sequence=None, StartTime=None, 
                 raw=False):
        """
        Retrieves value of a particular time point. If given hannel is 
        a copy of an original channel, the values are retrieved from 
        the original one. In such case the sequences and start times are
        also treated by original channel.

        This is virtual function, a particular implementation depends
        on daughter class.

        Parameters
        ----------
        point : int 
            the index of the point to be retrieved
            If sequence is not given, a global index is used
        point : datetime
            the time of point to be retrieved
        point : timedelta
            the index of the point to be retrieved 
            by time passed from beginning of sequence
        default : int, 0
            returned value if asked point not available
            e.g. not in sequence
        sequence : int, optional
            specifies the sequence in which data will be retrieved. 
            Points outside given sequence will return default value.
            If pont parameter is given by time, sequence is ignored
        StartTime : datetime, optional
            if point is given by timedelta, specifies the reference time.
            If set to None, the channel-defined value is used.
            If sequence is specified, StartTime is ignored and 
            the beginning of given sequence is used as reference
        raw : bool, False
            If set to true, the raw, unscaled value is retrieved

        Returns
        ---------
        float or int
            the value of required point

        Raises
        --------
        TypeError
            if given parameters are of wrong type
        NotImplementedError
            if class do not implements data retrieval in 
            _getValue function
        """
        # In case of copied channel, all sequences and times are
        # treated by original channel
        if self._baseChannel != self:
            return self._baseChannel.GetValue(point, default,
                                              sequence, StartTime, raw)

        if not (isinstance(point, int) 
                or isinstance(point, datetime) 
                or isinstance(point, timedelta)):
            raise TypeError("point must be either int, datetime or timedelta")
        if not (sequence is None or isinstance(sequence, int)):
            raise TypeError("sequence must be either None or int")
        if not isinstance(raw, bool):
            raise TypeError("raw must be a bool")

        if sequence is not None:
            if StartTime is not None:
                Logger.warning("StartTime is defined together "
                               "with sequence. StartTime will be ignored")
            if sequence < 0 or sequence > self.GetNsequences():
                return default
            StartTime = self.GetSequenceStart(sequence)
        if StartTime is None:
            StartTime = self._startTime

        # point by time
        if isinstance(point, datetime):
            if sequence is not None:
                Logger.warning("sequence parameter is defined "
                               "but point is passed by absolute time. "
                               "sequence will be ignored")
            # converting time to index
            point, sequence = self._getLocalIndex(point)

        # point by timedelta
        elif isinstance(point, timedelta):
            if sequence is not None:
                point = round(point.total_seconds() * self._frequency)
                if point > self.GetSequenceSize(sequence):
                    point = -1
            else:
                point = StartTime + point
                point, sequence = self._getLocalIndex(point)

        # point by index
        else:
            if sequence is not None: 
                if point > self.GetSequenceSize(sequence):
                    point = -1
            else:
                point = self._startTime + timedelta(seconds=point
                                                    / self._frequency)
                point, sequence = self._getLocalIndex(point)

        if point < 0 or sequence < 0:
            return default

        value = self._getValue(point, sequence)
        if raw:
            return value
        else:
            return self._fromRaw(value)

    def GetValueVector(self, timeStart, timeEnd, 
                       default=0, freq_mult=None, raw=False):
        """
        Reads and returns datapoints in range [timeStart, timeEnd[.
        The data point coresponding to timeEnd is not retrieved to avoid
        overlaps in sequential reading. If timeEnd - timeStart < 1/frequency
        no data will be readed.

        If given hannel is a copy of an original channel, the values 
        are retrieved from the original one. In such case the sequences 
        and start times are also treated by original channel.

        All values that are output data sequences are filled with 
        default value.

        This functions calls _getValueVector virtual function

        Parameters
        ----------
        timeStart : datetime
            Start time point for reading data
        timeEnd : datetime
            End time point for reading data. Must be equal or bigger than
            timeStart. Data point at timeEnd is not retrieved.
        timeEnd : timedelta
            time range from startTime to be read. Must be positive.
        default : float, 0
            default value for result, if data fals out of sequences
        freq_mult : int, None
            If set, resulting list will be oversampled by this value.
            Each additional cells will be filled with preceeding value
        raw : bool, False
            If set to true, the retrieved values will be unscaled

        Raises
        ------
        TypeError
            if passed parameters are of wrong type
        ValueError
            if timeStart is greater than stopTime
        NotImplemented
            if _getValueVector is not implemented for used format
        """
        if self._baseChannel != self:
            return self._baseChannel.GetValueVector(timeStart, timeEnd,
                                                    default, freq_mult, raw)
        if not (isinstance(timeStart, datetime)):
            raise TypeError("timeStart must be datetime")
        if not (isinstance(timeEnd, datetime)
                or isinstance(timeEnd, timedelta, float)):
            raise TypeError("timeEnd must be either "
                            "datetime, timedelta or float")
        if freq_mult is None:
            freq_mult = 1
        if not (isinstance(freq_mult, int)):
            raise TypeError("freq_mult must be int")
        if not (isinstance(raw, bool)):
            raise TypeError("raw must be boolean")

        dt = timeEnd
        if isinstance(dt, datetime):
            dt = (dt - timeStart).total_seconds()
        elif isinstance(dt, timedelta):
            timeEnd = timeStart + dt
            dt = dt.total_seconds()
        if dt < 0:
            raise ValueError("time span must be positif")

        # total size of data to retrieve
        points = int(dt * self._frequency)
        res = [default] * int(dt * self._frequency * freq_mult)
        seq = -1

        for seq_start, seq_size, seq_time\
                in zip(self._seqStart, self._seqSize, self._seqStartTime):
            seq += 1
            # Sequance starts after end time
            if seq_time >= timeEnd: break
            # offset of sequance start relative to start time
            offset = round((timeStart - seq_time).total_seconds()
                           * self._frequency)

            # Sequence ends before time start
            if (offset) >= seq_size:
                continue

            to_read = 0
            # Index to point in res
            index = 0
            read_start = 0

            # Case 1: sequence started before timeStart, offset is negative
            # We fill from beginning of res list,
            # but reading data from middle of sequence
            if offset >= 0 :
                # number of points to the end of sequence
                to_read = min(seq_size - offset, points)
                read_start = offset

            # Case 2: sequence starts after timeStart, offset is positive
            # We read from start of sequence,
            # but fill in the middle of res vector
            else:
                offset = -offset
                if offset * freq_mult > len(res): break
                to_read = min(seq_size, points - offset)
                index = offset * freq_mult

            d = self._getValueVector(read_start, to_read, seq)
            if len(d) != to_read:
                raise Exception("Sequence {}: readed {} points, "
                                "{} expected".format(
                                    seq, len(d), to_read))
            for i in range(0, to_read):
                # res[index] = struct.unpack(self.Endian\
                #              + self._Marks[b'\x20\x00\x00\x00'].Format,
                #              data[i:i+self._dataSize])[0]
                res[index] = d[i]
                if res[index] > self._digMax: res[index] = self._digMax
                if res[index] < self._digMin: res[index] = self._digMin
                if not raw:
                    res[index] = self._fromRaw(res[index])
                # filling the interpoint space with previous value
                for j in range(index + 1, index + freq_mult):
                    res[j] = res[index]
                index += freq_mult
        return res

    def _getLocalIndex(self, time):
        """
        Retrieves point index and sequence for a given time. If there 
        no corresponding index and/or sequence, will return -1 as 
        corresponding value.

        Do not checks for types

        Parameters
        ----------
        time : datetime

        Returns
        -------
        (int, int)
            a tuple of (point, sequence). If time is before the start 
            of first sequence, sequence will be set to -1, else 
            sequence will be the latest sequence before the given time.
            If time is after the sequence end, the index will be set to -1
        """
        ind = -1
        seq = -1
        for t in self._seqStartTime:
            if round((time - t).total_seconds()
                     * self._frequency) < 0:
                break
            seq += 1
        if seq >= 0:
            ind = round((time - self.GetSequenceStart(seq)).total_seconds()
                        * self._frequency)
            if ind >= self.GetSequenceSize(seq):
                ind = -1
        return (ind, seq)

    def _getTime(self, point, StartTime, freqMultiplier):
        """
        Retrieves time corresponding to a index given starting time 
        and frequency multiplier. 

        Do not check for parameters validity

        Parameters
        ----------
        point : int
            global index of a data point
        SatrtTime : datetime
            Starting time of data
        freqMultiplier : int
            frequency multiplier to convert channel frequency to 
            global one

        Returns
        -------
        datetime
            time corresponding to given index
        """
        return StartTime + self._getDeltaTime(point, freqMultiplier)

    def _getDeltaTime(self, point, freqMultiplier):
        """
        Retrieves the time passed since the reference time,
        given data point index and frequency multiplier.

        Do not check for parameters validity.

        Parameters
        ----------
        point : int
            index of data point
        freqMultiplier : int
            frequency multiplier to convert channel frequency to 
            global one

        Returns
        -------
        timedelta
            time passed since start
        """
        return timedelta(seconds=point / (self._frequency * freqMultiplier))

    def _getValue(self, point, sequence):
        """
        Retrieves value of a particular time point.
        This is virtual function and will always raise
        NotImplemented error.

        The reimplementation of function is not expected to check 
        the validity of parameters and ranges.

        Parameters
        ----------
        point : int
            the index of the point to be retrieved
        sequence : int
            specifies the sequence in which data will be retrieved

        Returns
        -------
        float or int
            the value of required point

        Raises
        ------
        NotImplementedError
            if _getValue is not implemented for given format
        """
        raise NotImplementedError("_getValue")

    def _getValueVector(self, index, size, sequence):
        """
        Reads maximum size points from a given sequence
        starting from index. If size is negative, will
        retrieve data till the end of sequence.
        Will stop at end of sequence.

        This is virtual function and will always raise
        NotImplemented error.

        The reimplementation of function is not expected to check 
        the validity of parameters and ranges.

        Parameters
        ----------
        index : int
            a valid index from where data will be read
        size : int
            number of data-points retrieved, will not stop if reaches 
            end of sequence or end of file
        sequence :
            index of sequence to be read from

        Returns
        -------
        list(int)
            a list of readed data

        Raises
        ------
        IOError
            if reaches EOF before reading requested data
        NotImplementedError
            if function is not defined for given format
        """
        raise NotImplementedError("_getValueVector")

    def __lt__(self, other):
        """
        Less operator for sorting

        Returns
        -------
        bool
        """
        return self._name < other._name












class Field(object):
    """ Class describes type of data and how to read it"""
    __slots__ = ["Name", "Size", "IsText", "Format", "Encoding", "Entries"]

    def __init__(self, Name, Format, Size=0, IsText=False,
                 Encoding="Latin-1", Entries=0, Unique=False):
        self.Name = Name
        self.Format = Format
        self.Size = Size  # 0 -- no size restriction
        self.IsText = IsText
        self.Encoding = Encoding
        if Unique :
            self.Entries = 1
        else:
            self.Entries = Entries

    def __str__(self):
        string = self.Name + ":"
        if (self.IsText):
            string = string + "text (" + self.Encoding + ")"
        else :
            string = string + self.Format       
        if self.Entries == 1:
            string = string + " Unique"
        elif self.Entries > 1 :
            string = string + "{} entries".format(self.Entries)
        return string

    def IsUnique(self):
        return (self.Entries == 1)


class EmbChannel(GenChannel):
    """ Class containing all information retrieved from ebm file.
    The data instead to be loaded in the memory, 
    are readed directly from file """

    # Minimum and maximum values for short integer
    _MAXINT = 32767
    _MININT = -32767

    """ A dictionary of fields in the ebm file,
    each entry will create a corresponding field in channel class"""
    _Marks = {
        b'\x80\x00\x00\x00' : Field("Version", "B", Size=2, Unique=True),
        b'\x81\x00\x00\x00' : Field("Header", "x", IsText=True, Unique=True),
        b'\x84\x00\x00\x00' : Field("Time", "HBBBBBB", Size=1),
        b'\x85\x00\x00\x00' : Field("Channel", "h", Unique=True),
        b'\x86\x00\x00\x00' : Field("Sampling", "L", Unique=True),
        b'\x87\x00\x00\x00' : Field("Gain", "L", Unique=True),
        b'\x88\x00\x00\x00' : Field("SCount", "I", Unique=True),
        b'\x89\x00\x00\x00' : Field("DBLsampling", "d", Unique=True),
        b'\x8a\x00\x00\x00' : Field("RateCorr", "d", Unique=True),
        b'\x8b\x00\x00\x00' : Field("RawRange", "d", Unique=True),
        b'\x8c\x00\x00\x00' : Field("TransRange", "d", Unique=True),
        b'\x8d\x00\x00\x00' : Field("Channel_32", "H", Unique=True),

        b'\x90\x00\x00\x00' : Field("ChannName", "x",
                                    IsText=True, Unique=True),
        b'\x95\x00\x00\x00' : Field("DMask_16", "h"),
        b'\x96\x00\x00\x00' : Field("SignData", "B", 
                                    Unique=True, Size=1),
        b'\x98\x00\x00\x00' : Field("CalFunc", "x",
                                    IsText=True, Unique=True),
        b'\x99\x00\x00\x00' : Field("CalUnit", "h",
                                    IsText=True, Unique=True),
        b'\x9A\x00\x00\x00' : Field("CalPoint", "h"),

        b'\xa0\x00\x00\x00' : Field("Event", "h"),

        b'\xc0\x00\x00\x00' : Field("SerialNumber", "x",
                                    IsText=True, Unique=True),
        b'\xc1\x00\x00\x00' : Field("DeviceType", "x",
                                    IsText=True, Unique=True),

        b'\xd0\x00\x00\x00' : Field("SubjectName", "x",
                                    IsText=True, Unique=True),
        b'\xd1\x00\x00\x00' : Field("SubjectId", "x",
                                    IsText=True, Unique=True),
        b'\xd2\x00\x00\x00' : Field("SubjectGroup", "x",
                                    IsText=True, Unique=True),
        b'\xd3\x00\x00\x00' : Field("SubjectAtten", "x",
                                    IsText=True, Unique=True),

        b'\xe0\x00\x00\x00' : Field("FilterSet", "h"),
        b'\x20\x00\x00\x00' : Field("Data", "f"),
        b'\x30\x00\x00\x00' : Field("DataGuId", "x",
                                    IsText=True, Unique=True),
        b'\x40\x00\x00\x00' : Field("RecGuId", "x",
                                    IsText=True, Unique=True),

        b'\xA0\x00\x00\x02' : Field("SigType", "h",
                                    IsText=True, Unique=True),
        b'\x20\x00\x00\x04' : Field("LowHight", "d", Unique=True),
        b'\x70\x00\x00\x03' : Field("SigRef", "h",
                                    IsText=True, Unique=True),
        b'\x72\x00\x00\x03' : Field("SigMainType", "h",
                                    IsText=True, Unique=True),
        b'\x74\x00\x00\x03' : Field("SigSubType", "h",
                                    IsText=True, Unique=True),
        b'\xff\xff\xff\xff' : Field("UnknownType", "h")
    } 

    __slots__ = [x.Name for x 
                 in list(_Marks.values())] + [
                         "Endian", "Wide", "_stream",
                         "_seqStart", "_totSize", "_dataSize"]

    def __init__(self, filename):
        super(EmbChannel, self).__init__()
        for f in self.__slots__:
            if f[0:1] != "_":
                setattr(self, f, None)

        self._seqStart = []
        self._totSize = 0
        self._dataSize = 0

        self._stream = open(filename, "rb")
        if not isinstance(self._stream, (io.RawIOBase, io.BufferedIOBase)):
            raise Exception("Stream is not valid")
        self._stream.seek(0)

        # Reading header
        buff = b''
        ch = self._stream.read(1)

        while ch != b'\x1a':
            buff = buff + ch
            ch = self._stream.read(1)

        if (buff.decode('ascii') != 'Embla data file')\
                and (buff.decode('ascii') != 'Embla results file')\
                and (buff.decode('ascii') != 'Embla raw file'):
            raise Exception("We are not reading either Embla results "
                            "or Embla data")
        ch = self._stream.read(1)
        if ch == b'\xff':
            self.Endian = '>'
        elif ch == b'\x00':
            self.Endian = '<'
        else:
            raise Exception("Can't determine endian")

        self.Wide = False
        ch = self._stream.read(1)
        if (ch == b'\xff'):
            ch = self._stream.read(4)
            if ch == b'\xff\xff\xff\xff':   
                self.Wide = True
                self._stream.seek(32 - 6,1)

        if self.Wide:
            self._Marks[b'\x20\x00\x00\x00'].Format = 'h'
            self._dataSize = 2
        else:
            self._Marks[b'\x20\x00\x00\x00'].Format = 'b'
            self._dataSize = 1

        while True:
            start = self._stream.tell()
            if self.Wide :
                index = self._stream.read(4)
            else:
                index = self._stream.read(2)
                index = index + b'\x00\x00'
            if(index == b''):break
            size = self._stream.read(4)
            size = struct.unpack("<L", size)[0]
            readed = self._read(index, size)
            if (readed != size):
                Logger.warning('In file "{}" at {}'
                               .format(self._stream.name, start))
                Logger.warning("Readed {} bytes, {} expected. "
                               "File seems to be corrupted"
                               .format(readed, size))
                self._stream.seek(0,2)
        self._totSize = sum(self._seqSize)

        # Finalizing initialization
        self._name = self.ChannName
        self._type = self.SigType
        self._id = self.SigMainType + "_" + self.SigSubType
        self._description = self.SigMainType
        if self.SigSubType != "":
            self._description += "-" + self.SigSubType
        self._reference = self.SigRef
        self._unit = self.CalUnit
        self._seqStartTime = self.Time
        self._frequency = int(self.DBLsampling + 0.5)
        if abs(self.DBLsampling / self._frequency - 1) > 1e-4 :
            Logger.warning("{}: Sample frequency is not integer."
                           "Correction factor is 1{:+}"
                           .format(self.GetName(),
                                   self.DBLsampling / self._frequency - 1))
        if (self.RateCorr is not None and self.RateCorr > 1e-4):
            Logger.warning("{}: Sample frequency is not integer. "
                           "Correction factor is 1{:+}"
                           .format(self.GetName(),self.RateCorr))

        self._startTime = self._seqStartTime[0]

        self._digMin = self._MININT
        self._digMax = self._MAXINT

        if (self.RawRange[2] == 0.):
            if (abs(self.RawRange[1]) != abs(self.RawRange[0])):
                self.SetScale(max(abs(self.RawRange[1]),
                                  abs(self.RawRange[0]))
                              / self._digMax)
            else:
                self.SetPhysicalRange(self.RawRange[0], self.RawRange[1])
        else:
            self._digMin = int(self.RawRange[0] / self.RawRange[2])
            self._digMax = int(self.RawRange[1] / self.RawRange[2])
            self.SetScale(self.RawRange[2])
        if isinstance(self.CalFunc, str) and self.CalFunc != "":
            # God help us all
            # x is used implicetly in eval
            Logger.warning("Channel uses calibration function '" 
                           + self.CalFunc + 
                           "'. Actually only linear calibrations "
                           "are supported. If function is not linear, "
                           "retrieved values will be incorrect.")
            x = self.GetPhysMin()
            new_min = eval(self.CalFunc)
            x = self.GetPhysMax()
            new_max = eval(self.CalFunc)
            self.SetPhysicalRange(new_min, new_max)
            # Just for PEP8 conformity
            x

        self.OptimizeMagnitude()

    def __str__(self):
        string = ""
        for f in self.__slots__:
            if f[0:2] == "__":
                f = "_Channel" + f
            attr = getattr(self, f)
            if attr is not None:
                if type(attr) is list:
                    if len(attr) < 5 and len(attr) > 0:
                        if type(attr[0]) is list:
                            string = string + f + '\t'\
                                     + str((attr[0])[0:5]) + '\n'
                        else:
                            string = string + f + '\t'\
                                     + str(attr[0:5]) + '\n'
                    else:
                        string = string + f + '\t[{} entries]\n'\
                                 .format(len(attr))
                else:
                    string = string + f + '\t' + str(getattr(self, f)) + '\n'
        return string

    def __del__(self):
        self._stream.close()

    def _read(self, marker, size):
        start = self._stream.tell()
        if marker not in self._Marks:
            raise KeyError("Marker {} not in the list for channel from {}"
                           .format(marker, self._stream.name))
        dtype = self._Marks[marker]
        fname = dtype.Name 
        fsize = dtype.Size 
        ftype = dtype.Format 
        fenc = dtype.Encoding 
        if getattr(self, fname) is None and not dtype.IsUnique():
            setattr(self, fname, [])
        # tsize represents the a size of the entry, for text it is fixed to 1
        if dtype.IsText:
            tsize = 1
        else:
            tsize = int(struct.calcsize(self.Endian + ftype))
        dsize = size  # Lenth of the field
        nwords = int(dsize / tsize)  # Number of entries in the field

        if fsize > 0 and nwords != fsize:
            raise Exception("Field contains {} words, {} requested"
                            .format(nwords, fsize))

        if dtype.IsText:
            text = self._stream.read(size).decode(fenc).strip('\0')
            if dtype.IsUnique():
                setattr(self, fname, text)
            else:
                setattr(self, fname, getattr(self, fname) + [text])
        else: 
            if fname == "UnknownType":
                # Put warning here: unknown size, corrupted file?
                Logger.warning("Unknown data type")
                # Jumping to EOF
                return self._stream.tell() - start
            if fname == "Data":
                self._seqStart.append(self._stream.tell())
                self._seqSize.append(nwords)
                self._stream.seek(size, 1)
                return self._stream.tell() - start
            dec = self.Endian + ftype * nwords + 'x' * (dsize - tsize * nwords)
            unpacked = struct.unpack(dec, self._stream.read(size))
            if fname == "Version":
                if self.Endian == '>':
                    big, small = unpacked
                else:
                    small, big = unpacked
                if small > 100: small = small / 100
                else: small = small / 10
                self.Version = big + small / 10
            elif fname == "Time":
                year, mon, day, h, m, s, us = unpacked
                time = datetime(year, mon, day, h, m, s, us * 10000)
                self.Time = self.Time + [time]
            else:
                if dtype.IsUnique():
                    if len(unpacked) == 1:
                        setattr(self, fname, unpacked[0])
                    else:
                        setattr(self, fname, list(unpacked))
                else:
                    setattr(self, fname,
                            getattr(self, fname) + [list(unpacked)])
        return self._stream.tell() - start

    def _getValue(self, point, sequence):
        """
        Retrieves value of a particular time point.
        This is reimplementation of Generic _getValue for Embla format

        It doesn't check the validity of parameters.

        Parameters
        ----------
        point : int
            the index of the point to be retrieved
        sequence : int
            specifies the sequence in which data will be retrieved

        Returns
        -------
        float or int
            the value of required point
        """

        self._stream.seek(self._seqStart[sequence] + (point) * self._dataSize)
        val = struct.unpack(
                self.Endian + self._Marks[b'\x20\x00\x00\x00'].Format,
                self._stream.read(self._dataSize))[0]
        if val > self._digMax: val = self._digMax
        if val < self._digMin: val = self._digMin
        return val

    def _getValueVector(self, index, size, sequence):
        """
        Reads maximum size points from a given sequence
        starting from index. If size is negative, will
        retrieve data till the end of sequence.

        Parameters
        ----------
        index : int
            a valid index from where data will be read
        size : int
            number of data-points retrieved
        sequence :
            index of sequence to be read from

        Returns
        -------
        list(int)
            a list of readed data

        Raises
        ------
        IOError
            if reaches EOF before reading requested data
        """
        if size < 0 or size > self._seqSize[sequence] - index:
            size = self._seqSize[sequence] - index
        self._stream.seek(self._seqStart[sequence] + index * self._dataSize)
        data = self._stream.read(self._dataSize * size)
        if len(data) != size * self._dataSize:
            raise IOError("Got {} entries insted of expected {} "
                          "while reading {}".format(len(data), size, 
                                                    self._stream.name)
                          )
        d = struct.unpack(self.Endian
                          + self._Marks[b'\x20\x00\x00\x00'].Format
                          * size, data)
        return d

    def __lt__(self, other):
        if type(other) != type(self):
            raise TypeError("Comparaison arguments must be of the same class")
        if self.Channel_32[1] < other.Channel_32[1]: return True
        if self.Channel_32[1] > other.Channel_32[1]: return False
        return self.Channel_32[0] < other.Channel_32[0]



def readsleepstagedata(filename):
    """Extract sleep stage data from .esedb file"""
    
    import olefile        

    esedb  = olefile.OleFileIO(filename).openstream('Event Store/Events')
    # parcel = Parcel()
    root   = Parcel(esedb)
    evs    = root.get("Events")
    grp_l  = root.getlist("Event Types")[0].getlist()
    times  = root.getlist("EventsStartTimes")[0]
    
    
    stagelabels = []
    stagestarts = []
    for ev,time in zip(evs, times):
        try:
            name = grp_l[ev.GroupTypeIdx]
        except:
            name = ""
        
        if 'SLEEP' in name: 
            stagelabels.append(name)
            stagestarts.append(time)
    
    stagelabels  = [x.replace('SLEEP-','') for x in stagelabels]
    stagelabels  = [x.replace('S0','W') for x in stagelabels]
    stagelabels  = [x.replace('S','N') for x in stagelabels]
    stagelabels  = [x.replace('REM','R') for x in stagelabels]
    
    stagecodes  = [x.replace('W','0') for x in stagelabels]
    stagecodes  = [x.replace('N1','1') for x in stagecodes]
    stagecodes  = [x.replace('N2','2') for x in stagecodes]
    stagecodes  = [x.replace('N3','3') for x in stagecodes]
    stagecodes  = [x.replace('R','4') for x in stagecodes]
    
    return stagelabels,stagecodes,stagestarts