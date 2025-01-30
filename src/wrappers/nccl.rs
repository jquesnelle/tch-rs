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
    pub fn new(store: &CStore, rank: i64, size: i64, device: Device) -> Result<Self, TchError> {
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
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_allreduce(
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

    pub fn copy_to_model_parallel(&self, tensor: &Tensor) -> Result<Tensor, TchError> {
        let output = unsafe_torch_err!(torch_sys::atd_process_group_nccl_copy_to_model_parallel(
            self.cnccl,
            tensor.c_tensor
        ));
        Ok(Tensor { c_tensor: output })
    }

    pub fn reduce_from_model_parallel(&self, tensor: &Tensor) -> Result<Tensor, TchError> {
        let output = unsafe_torch_err!(torch_sys::atd_process_group_nccl_reduce_from_model_parallel(
            self.cnccl,
            tensor.c_tensor
        ));
        Ok(Tensor { c_tensor: output })
    }

    pub fn scatter_to_model_parallel(&self, tensor: &Tensor) -> Result<Tensor, TchError> {
        let output = unsafe_torch_err!(torch_sys::atd_process_group_nccl_scatter_to_model_parallel(
            self.cnccl,
            tensor.c_tensor,
            self.size,
            self.rank
        ));
        Ok(Tensor { c_tensor: output })
    }

    pub fn gather_from_model_parallel(&self, tensor: &Tensor) -> Result<Tensor, TchError> {
        let output = unsafe_torch_err!(torch_sys::atd_process_group_nccl_gather_from_model_parallel(
            self.cnccl,
            tensor.c_tensor,
            self.size,
            self.rank
        ));
        Ok(Tensor { c_tensor: output })
    }

    pub fn send<T: AsRef<Tensor>>(&self, tensors: &[T], dst_rank: i64) -> Result<(), TchError> {
        let c_tensors = tensors.iter().map(|x| x.as_ref().c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_send(
            self.cnccl,
            c_tensors.as_ptr(),
            c_tensors.len() as c_int,
            dst_rank as c_int,
        ));
        Ok(())
    }

    pub fn recv<T: AsRef<Tensor>>(&self, tensors: &[T], src_rank: i64) -> Result<(), TchError> {
        let c_tensors = tensors.iter().map(|x| x.as_ref().c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_recv(
            self.cnccl,
            c_tensors.as_ptr(),
            c_tensors.len() as c_int,
            src_rank as c_int,
        ));
        Ok(())
    }

    pub fn all_gather<T: AsRef<Tensor>>(
        &self,
        output_tensors: &[T],
        input_tensor: &Tensor,
    ) -> Result<(), TchError> {
        let c_output_tensors =
            output_tensors.iter().map(|x| x.as_ref().c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_allgather(
            self.cnccl,
            c_output_tensors.as_ptr(),
            c_output_tensors.len() as c_int,
            input_tensor.c_tensor
        ));
        Ok(())
    }

    pub fn scatter<T: AsRef<Tensor>>(
        &self,
        output_tensor: &Tensor,
        input_tensors: &[T],
        root_rank: i64,
    ) -> Result<(), TchError> {
        let c_input_tensors = input_tensors.iter().map(|x| x.as_ref().c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_scatter(
            self.cnccl,
            output_tensor.c_tensor,
            c_input_tensors.as_ptr(),
            c_input_tensors.len() as c_int,
            root_rank as c_int
        ));
        Ok(())
    }

    pub fn group_start(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_group_start(self.cnccl));
        Ok(())
    }

    pub fn group_end(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atd_process_group_nccl_group_end(self.cnccl));
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
