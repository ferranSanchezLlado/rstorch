use std::convert::Infallible;
use std::error;
use std::fmt;

pub(crate) mod const_check;
mod data;
mod device;
mod dtype;
mod shape;

pub use data::DataError;
pub use device::DeviceError;
pub use dtype::DTypeError;
pub use shape::ShapeError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Shape(ShapeError),
    Backend(Box<dyn error::Error + Send + Sync + 'static>),
    Device(DeviceError),
    DType(DTypeError),
    Data(DataError),
}

impl Error {
    pub(crate) fn backend<E>(err: E) -> Self
    where
        E: error::Error + Send + Sync + 'static,
    {
        Self::Backend(Box::new(err))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(err) => write!(f, "shape error: {err}"),
            Self::Backend(err) => write!(f, "backend error: {err}"),
            Self::Device(err) => write!(f, "device error: {err}"),
            Self::DType(err) => write!(f, "dtype error: {err}"),
            Self::Data(err) => write!(f, "data error: {err}"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Backend(err) => Some(err.as_ref()),
            Self::Data(err) => err.source(),
            _ => None,
        }
    }
}

impl From<ShapeError> for Error {
    fn from(err: ShapeError) -> Self {
        Self::Shape(err)
    }
}

impl From<DeviceError> for Error {
    fn from(err: DeviceError) -> Self {
        Self::Device(err)
    }
}

impl From<DTypeError> for Error {
    fn from(err: DTypeError) -> Self {
        Self::DType(err)
    }
}

impl From<DataError> for Error {
    fn from(err: DataError) -> Self {
        Self::Data(err)
    }
}

impl From<Infallible> for Error {
    fn from(err: Infallible) -> Self {
        match err {}
    }
}
