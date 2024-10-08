//! The different kind of elements supported in Torch.

use half;

/// The different kind of elements that a Tensor can hold.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
    QUInt4x2,
    QUInt2x4,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    Float8e5m2,
    Float8e4m3fn,
    Float8e5m2fnuz,
    Float8e4m3fnuz,
    UInt16,
    UInt32,
    UInt64,
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7
}

impl Kind {
    pub fn c_int(self) -> libc::c_int {
        // These values should be in sync with include/c10/core/ScalarType.h
        // https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/c10/core/ScalarType.h#L57
        match self {
            Kind::Uint8 => 0,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 3,
            Kind::Int64 => 4,
            Kind::Half => 5,
            Kind::Float => 6,
            Kind::Double => 7,
            Kind::ComplexHalf => 8,
            Kind::ComplexFloat => 9,
            Kind::ComplexDouble => 10,
            Kind::Bool => 11,
            Kind::QInt8 => 12,
            Kind::QUInt8 => 13,
            Kind::QInt32 => 14,
            Kind::BFloat16 => 15,
            Kind::QUInt4x2 => 16,
            Kind::QUInt2x4 => 17,
            Kind::Bits1x8 => 18,
            Kind::Bits2x4 => 19,
            Kind::Bits4x2 => 20,
            Kind::Bits8 => 21,
            Kind::Bits16 => 22,
            Kind::Float8e5m2 => 23,
            Kind::Float8e4m3fn => 24,
            Kind::Float8e5m2fnuz => 25,
            Kind::Float8e4m3fnuz => 26,
            Kind::UInt16 => 27,
            Kind::UInt32 => 28,
            Kind::UInt64 => 29,
            Kind::UInt1 => 30,
            Kind::UInt2 => 31,
            Kind::UInt3 => 32,
            Kind::UInt4 => 33,
            Kind::UInt5 => 34,
            Kind::UInt6 => 35,
            Kind::UInt7 => 36,
        }
    }

    pub fn from_c_int(v: libc::c_int) -> Result<Kind, crate::TchError> {
        match v {
            0 => Ok(Kind::Uint8),
            1 => Ok(Kind::Int8),
            2 => Ok(Kind::Int16),
            3 => Ok(Kind::Int),
            4 => Ok(Kind::Int64),
            5 => Ok(Kind::Half),
            6 => Ok(Kind::Float),
            7 => Ok(Kind::Double),
            8 => Ok(Kind::ComplexHalf),
            9 => Ok(Kind::ComplexFloat),
            10 => Ok(Kind::ComplexDouble),
            11 => Ok(Kind::Bool),
            12 => Ok(Kind::QInt8),
            13 => Ok(Kind::QUInt8),
            14 => Ok(Kind::QInt32),
            15 => Ok(Kind::BFloat16),
            16 => Ok(Kind::QUInt4x2),
            17 => Ok(Kind::QUInt2x4),
            18 => Ok(Kind::Bits1x8),
            19 => Ok(Kind::Bits2x4),
            20 => Ok(Kind::Bits4x2),
            21 => Ok(Kind::Bits8),
            22 => Ok(Kind::Bits16),
            23 => Ok(Kind::Float8e5m2),
            24 => Ok(Kind::Float8e4m3fn),
            25 => Ok(Kind::Float8e5m2fnuz),
            26 => Ok(Kind::Float8e4m3fnuz),
            27 => Ok(Kind::UInt16),
            28 => Ok(Kind::UInt32),
            29 => Ok(Kind::UInt64),
            30 => Ok(Kind::UInt1),
            31 => Ok(Kind::UInt2),
            32 => Ok(Kind::UInt3),
            33 => Ok(Kind::UInt4),
            34 => Ok(Kind::UInt5),
            35 => Ok(Kind::UInt6),
            36 => Ok(Kind::UInt7),
            _ => Err(crate::TchError::UnknownKind(v)),
        }
    }

    pub fn elt_size_in_bytes(self) -> usize {
        match self {
            Kind::Uint8 => 1,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 4,
            Kind::Int64 => 8,
            Kind::Half => 2,
            Kind::Float => 4,
            Kind::Double => 8,
            Kind::ComplexHalf => 4,
            Kind::ComplexFloat => 8,
            Kind::ComplexDouble => 16,
            Kind::Bool => 1,
            Kind::QInt8 => 1,
            Kind::QUInt8 => 1,
            Kind::QInt32 => 4,
            Kind::BFloat16 => 2,
            Kind::QUInt4x2 => 1,
            Kind::QUInt2x4 => 1,
            Kind::Bits1x8 => 1,
            Kind::Bits2x4 => 1,
            Kind::Bits4x2 => 1,
            Kind::Bits8 => 1,
            Kind::Bits16 => 2,
            Kind::Float8e5m2 => 1,
            Kind::Float8e4m3fn => 1,
            Kind::Float8e5m2fnuz => 1,
            Kind::Float8e4m3fnuz => 1,
            Kind::UInt16 => 2,
            Kind::UInt32 => 4,
            Kind::UInt64 => 8,
            Kind::UInt1 => 1,
            Kind::UInt2 => 1,
            Kind::UInt3 => 1,
            Kind::UInt4 => 1,
            Kind::UInt5 => 1,
            Kind::UInt6 => 1,
            Kind::UInt7 => 1,
        }
    }
}

pub const FLOAT_CPU: (Kind, crate::Device) = (Kind::Float, crate::Device::Cpu);
pub const DOUBLE_CPU: (Kind, crate::Device) = (Kind::Double, crate::Device::Cpu);
pub const INT64_CPU: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cpu);

pub const FLOAT_CUDA: (Kind, crate::Device) = (Kind::Float, crate::Device::Cuda(0));
pub const DOUBLE_CUDA: (Kind, crate::Device) = (Kind::Double, crate::Device::Cuda(0));
pub const INT64_CUDA: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cuda(0));

/// Kinds for tensor elements
///
/// # Safety
/// The specified Kind must be for a type that has the same length as Self.
pub unsafe trait Element: Clone {
    const KIND: Kind;
    const ZERO: Self;
}

unsafe impl Element for u8 {
    const KIND: Kind = Kind::Uint8;
    const ZERO: Self = 0;
}

unsafe impl Element for i8 {
    const KIND: Kind = Kind::Int8;
    const ZERO: Self = 0;
}

unsafe impl Element for i16 {
    const KIND: Kind = Kind::Int16;
    const ZERO: Self = 0;
}

unsafe impl Element for i32 {
    const KIND: Kind = Kind::Int;
    const ZERO: Self = 0;
}

unsafe impl Element for i64 {
    const KIND: Kind = Kind::Int64;
    const ZERO: Self = 0;
}

unsafe impl Element for half::f16 {
    const KIND: Kind = Kind::Half;
    const ZERO: Self = half::f16::ZERO;
}

unsafe impl Element for half::bf16 {
    const KIND: Kind = Kind::Half;
    const ZERO: Self = half::bf16::ZERO;
}

unsafe impl Element for f32 {
    const KIND: Kind = Kind::Float;
    const ZERO: Self = 0.;
}

unsafe impl Element for f64 {
    const KIND: Kind = Kind::Double;
    const ZERO: Self = 0.;
}

unsafe impl Element for bool {
    const KIND: Kind = Kind::Bool;
    const ZERO: Self = false;
}
