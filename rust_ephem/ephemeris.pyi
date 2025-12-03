import abc
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from ._rust_ephem import (
    GroundEphemeris as GroundEphemeris,
)
from ._rust_ephem import (
    OEMEphemeris as OEMEphemeris,
)
from ._rust_ephem import (
    PositionVelocityData,
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
    def timestamp(self) -> npt.NDArray[np.datetime64]: ...
    @property
    @abc.abstractmethod
    def gcrs_pv(self) -> PositionVelocityData: ...
    @property
    @abc.abstractmethod
    def itrs_pv(self) -> PositionVelocityData: ...
    @property
    @abc.abstractmethod
    def itrs(self) -> Any: ...  # Returns astropy.coordinates.SkyCoord
    @property
    @abc.abstractmethod
    def gcrs(self) -> Any: ...  # Returns astropy.coordinates.SkyCoord
    @property
    @abc.abstractmethod
    def earth(self) -> Any: ...  # Returns astropy.coordinates.SkyCoord
    @property
    @abc.abstractmethod
    def sun(self) -> Any: ...  # Returns astropy.coordinates.SkyCoord
    @property
    @abc.abstractmethod
    def moon(self) -> Any: ...  # Returns astropy.coordinates.SkyCoord
    @property
    @abc.abstractmethod
    def sun_pv(self) -> PositionVelocityData: ...
    @property
    @abc.abstractmethod
    def moon_pv(self) -> PositionVelocityData: ...
    @property
    @abc.abstractmethod
    def obsgeoloc(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def obsgeovel(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def latitude(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def latitude_deg(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def latitude_rad(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def longitude(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def longitude_deg(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def longitude_rad(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def height(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def height_m(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def height_km(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def sun_radius(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def sun_radius_deg(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def moon_radius(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def moon_radius_deg(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def earth_radius(self) -> Any: ...  # Returns astropy.units.Quantity
    @property
    @abc.abstractmethod
    def earth_radius_deg(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def sun_radius_rad(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def moon_radius_rad(self) -> npt.NDArray[np.float64]: ...
    @property
    @abc.abstractmethod
    def earth_radius_rad(self) -> npt.NDArray[np.float64]: ...
    @abc.abstractmethod
    def index(self, time: datetime) -> int: ...
    @property
    @abc.abstractmethod
    def begin(self) -> datetime: ...
    @property
    @abc.abstractmethod
    def end(self) -> datetime: ...
    @property
    @abc.abstractmethod
    def step_size(self) -> int: ...
    @property
    @abc.abstractmethod
    def polar_motion(self) -> bool: ...

EphemerisType = TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris
