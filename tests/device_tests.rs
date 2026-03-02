use tch::{Cuda, Device, Tensor};

#[test]
fn tensor_device() {
    let t = Tensor::from_slice(&[3, 1, 4]);
    assert_eq!(t.device(), Device::Cpu)
}

#[test]
fn cuda_device_capability() {
    if Cuda::is_available() {
        let (major, minor) = Cuda::get_device_capability(0).unwrap();
        assert!(major > 0, "expected a positive major compute capability, got {major}");
        assert!(minor >= 0, "expected a non-negative minor compute capability, got {minor}");
    } else {
        assert!(Cuda::get_device_capability(0).is_err());
    }
}
