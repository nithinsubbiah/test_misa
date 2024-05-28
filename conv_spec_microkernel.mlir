// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module attributes {transform.with_named_sequence} {

  util.func private @conv_entry_point_k5(%arg0: tensor<2x66x66x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) 
                                          -> tensor<2x64x64x1280xf16> {
    %wei_empty = tensor.empty() : tensor<1280x3x3x1280xf16>
    %wei_tr = linalg.transpose ins(%arg1: tensor<3x3x1280x1280xf16>) outs(%wei_empty : tensor<1280x3x3x1280xf16>) permutation = [3, 0, 1, 2] 
    
    %hi = arith.constant 66 : i32
    %wi = arith.constant 66 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 1280 : i32 // number of filters
    %c = arith.constant 1280 : i32
    %ho = arith.constant 64 : i32
    %wo = arith.constant 64 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(%hi, %wi,
        %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group, %arg0, %wei_tr) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<2x66x66x1280xf16>, tensor<1280x3x3x1280xf16>) -> tensor<2x64x64x1280xf16>
      count(%device: !hal.device) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        %c1280_0 = arith.constant 2560 : index
        hal.return %c1280_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 16, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/nmeganat/MISA/out/igemm_fwd_gtc_gfx940_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    util.return %5 : tensor<2x64x64x1280xf16>
  }

  transform.named_sequence @cast_and_call_dag_k5(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k5 into %module if undefined : (!transform.any_op) -> !transform.any_op
    
    transform.util.cast_and_call %func(%ins) -> %out after %root {
  } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k5(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x66x66x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>):
        %cst_31 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x64x64x1280xf32>
        %87 = linalg.fill ins(%cst_31 : f32) outs(%84 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x66x66x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%87 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
        %7 = arith.truncf %6 : tensor<2x64x64x1280xf32> to tensor<2x64x64x1280xf16>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv_k5 -> @cast_and_call_dag_k5
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

