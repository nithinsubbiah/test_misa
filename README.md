NOTE: 

MISA filter is NHWC
IREE filter is HWCN

Also, the driver code takes in f32 vectors and converts to f16 using the utility provided. The npy files are consistently named and used. 



# Compiling MISA and running MISA kernel with cpp code

>>  hipcc driver.cpp && ./a.out
This code does have a couple of hard coded paths to the input and the hsaco file (sorry!) so please edit it accordingly. We load the f32 input, convert it into f16 which then gets used in MISA.

# Compiling and running IREE conv kernel

>> iree-compile --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx942 --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode/ --iree-codegen-llvmgpu-use-vector-distribution=true --iree-llvmgpu-enable-prefetch k5.mlir -o k5_iree.vmfb

>> iree-run-module --module=k5_iree.vmfb --function=forward --input=@k5_input_f16.npy --input=@k5_filter_f16.npy --device=rocm --device_allocator=caching --output=@k5_iree_out.npy

Uses the same input as the one used for MISA cpp code except in f16 directly. 

Both the results above match and seem to be within tolerance. If its not acceptable, then need to pay attention to the order of accumulation types.

# Compiling and running IREE-MISA integrated kernel

>> iree-compile --iree-hal-target-backends=rocm --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx942 --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode/ --iree-preprocessing-transform-spec-filename=conv_spec_microkernel.mlir k5.mlir -o k5_misa.vmfb

>> iree-run-module --module=k5_misa.vmfb --function=forward --input=@k5_input_f16.npy --input=@k5_filter_f16.npy --device=rocm --device_allocator=caching --output=@k5_misa_out.npy

This outputs repeated values of `-1.686` mostly which could suggest that buffer allocation/reading is off. This is the problem I was trying to solve last.  
