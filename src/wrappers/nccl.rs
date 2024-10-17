use crate::{Device, TchError, Tensor};
use libc::{c_int, c_uchar};
use std::sync::Arc;

pub struct CNCCL {
    cnccl: *mut torch_sys::CNCCL_,
    rank: i64,
    size: i64,
    _store: Arc<CStore>,
}

unsafe impl Send for CNCCL {}

impl std::fmt::Debug for CNCCL {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NCCL (rank={} size={})", self.rank, self.size)
    }
}

pub struct CStore {
    cstore: *mut torch_sys::CStore_,
}

unsafe impl Send for CStore {}
unsafe impl Sync for CStore {}

impl std::fmt::Debug for CStore {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Store")
    }
}

impl CNCCL {
    pub fn new(store: Arc<CStore>, rank: i64, size: i64, device: Device) -> Result<Self, TchError> {
        let cnccl = unsafe_torch_err!(torch_sys::atd_new_process_group_nccl(
            store.cstore,
            rank as c_int,
            size as c_int,
            device.c_int()
        ));
        Ok(Self { cnccl, rank, size, _store: store })
    }

    pub fn rank(&self) -> i64 {
        self.rank
    }

    pub fn size(&self) -> i64 {
        self.size
    }

    pub fn all_reduce<T: AsRef<Tensor>>(
        &self,
        tensors: &[T],
        reduction: ReduceOpType,
    ) -> Result<(), TchError> {
        let c_tensors = tensors.iter().map(|x| x.as_ref().c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_group_allreduce(
            self.cnccl,
            c_tensors.as_ptr(),
            c_tensors.len() as c_int,
            reduction.c_uchar()
        ));
        Ok(())
    }

    pub fn barrier(&self, device: Device) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_barrier(self.cnccl, device.c_int()));
        Ok(())
    }

    pub fn differentiable_all_reduce_sum(&self, tensor: &Tensor) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_group_differentiable_allreduce_sum(
            self.cnccl,
            tensor.c_tensor
        ));
        Ok(())
    }
}

impl Drop for CNCCL {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::atd_free_process_group_nccl(self.cnccl));
    }
}

impl CStore {
    pub fn new() -> Self {
        Self { cstore: unsafe_torch!(torch_sys::atd_new_hash_store()) }
    }
}

impl Drop for CStore {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::atd_free_hash_store(self.cstore))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReduceOpType {
    Sum,
    Avg,
    Product,
    Min,
    Max,
    BAnd,
    BOr,
    BXOr,
    PremulSum,
}

impl ReduceOpType {
    pub(super) fn c_uchar(self) -> c_uchar {
        match self {
            ReduceOpType::Sum => 0,
            ReduceOpType::Avg => 1,
            ReduceOpType::Product => 2,
            ReduceOpType::Min => 3,
            ReduceOpType::Max => 4,
            ReduceOpType::BAnd => 5,
            ReduceOpType::BOr => 6,
            ReduceOpType::BXOr => 7,
            ReduceOpType::PremulSum => 8,
        }
    }
}
