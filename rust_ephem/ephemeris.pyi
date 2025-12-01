import abc

from ._rust_ephem import (
    GroundEphemeris as GroundEphemeris,
)
from ._rust_ephem import (
    OEMEphemeris as OEMEphemeris,
)
from ._rust_ephem import (
    SPICEEphemeris as SPICEEphemeris,
)
from ._rust_ephem import (
    TLEEphemeris as TLEEphemeris,
)

class Ephemeris(abc.ABC):
    @property
    @abc.abstractmethod
    def timestamp(self): ...
    @property
    @abc.abstractmethod
    def gcrs_pv(self): ...
    @property
    @abc.abstractmethod
    def itrs_pv(self): ...
    @property
    @abc.abstractmethod
    def itrs(self): ...
    @property
    @abc.abstractmethod
    def gcrs(self): ...
    @property
    @abc.abstractmethod
    def earth(self): ...
    @property
    @abc.abstractmethod
    def sun(self): ...
    @property
    @abc.abstractmethod
    def moon(self): ...
    @property
    @abc.abstractmethod
    def sun_pv(self): ...
    @property
    @abc.abstractmethod
    def moon_pv(self): ...
    @property
    @abc.abstractmethod
    def obsgeoloc(self): ...
    @property
    @abc.abstractmethod
    def obsgeovel(self): ...
    @property
    @abc.abstractmethod
    def latitude(self): ...
    @property
    @abc.abstractmethod
    def latitude_deg(self): ...
    @property
    @abc.abstractmethod
    def latitude_rad(self): ...
    @property
    @abc.abstractmethod
    def longitude(self): ...
    @property
    @abc.abstractmethod
    def longitude_deg(self): ...
    @property
    @abc.abstractmethod
    def longitude_rad(self): ...
    @property
    @abc.abstractmethod
    def height(self): ...
    @property
    @abc.abstractmethod
    def height_m(self): ...
    @property
    @abc.abstractmethod
    def height_km(self): ...
    @property
    @abc.abstractmethod
    def sun_radius(self): ...
    @property
    @abc.abstractmethod
    def sun_radius_deg(self): ...
    @property
    @abc.abstractmethod
    def moon_radius(self): ...
    @property
    @abc.abstractmethod
    def moon_radius_deg(self): ...
    @property
    @abc.abstractmethod
    def earth_radius(self): ...
    @property
    @abc.abstractmethod
    def earth_radius_deg(self): ...
    @property
    @abc.abstractmethod
    def sun_radius_rad(self): ...
    @property
    @abc.abstractmethod
    def moon_radius_rad(self): ...
    @property
    @abc.abstractmethod
    def earth_radius_rad(self): ...
    @abc.abstractmethod
    def index(self, time): ...
    @property
    @abc.abstractmethod
    def begin(self): ...
    @property
    @abc.abstractmethod
    def end(self): ...
    @property
    @abc.abstractmethod
    def step_size(self): ...
    @property
    @abc.abstractmethod
    def polar_motion(self): ...

EphemerisType = TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris
