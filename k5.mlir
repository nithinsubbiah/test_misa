module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<2x66x66x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) -> tensor<2x64x64x1280xf16> {
    %cst_31 = arith.constant 0.000000e+00 : f32
    %84 = tensor.empty() : tensor<2x64x64x1280xf32>
    %87 = linalg.fill ins(%cst_31 : f32) outs(%84 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
    %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x66x66x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%87 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
    %7 = arith.truncf %6 : tensor<2x64x64x1280xf32> to tensor<2x64x64x1280xf16>
    return %7 : tensor<2x64x64x1280xf16>
  }
}
