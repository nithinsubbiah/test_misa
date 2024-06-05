#include "half.hpp"
#include "utils.h"
#include "npy.h"

#include <chrono>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using float16 = half_float::half;

// Datatype defined as in MISA for convolution kernel parameters
typedef struct {
    void *p_in;
    void *p_wei;
    void *p_out;
    int hi;
    int wi;
    int n;
    int k;                      // this is indeed k_per_group
    int c;                      // this is indeed c_per_group
    int ho;
    int wo;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int y;
    int x;
    int group;
    uint32_t magic_0;                       // denom: (gemm_n + n_per_block - 1) / n_per_block
    uint32_t magic_1;                       // denom: ho*wo
    uint32_t magic_2;                       // denom: wo
    uint32_t magic_3;                       // denom: (gemm_m/m_per_block) * (gemm_n/n_per_block)
    uint32_t magic_4;                       // denom: x*c
    uint32_t magic_5;                       // denom: c
    uint32_t shift_pack_0;
    uint32_t shift_pack_1;
    uint32_t ks;
    uint32_t __pack_0;
} __attribute__((packed)) igemm_fwd_gtc_nhwc_karg_t;

std::vector<char> readFileIntoVector(const char *filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return std::vector<char>();
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();
  return buffer;
}

static inline size_t naive_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

/************************** nhwc ****************************/
static inline void naive_conv_fwd_nhwc(const float *src, const float *filter,
                                       float *dst, size_t n, size_t w, size_t h,
                                       size_t c, size_t k, size_t fx, size_t fy,
                                       size_t px, size_t py, size_t sx,
                                       size_t sy, size_t dx, size_t dy, size_t group) {
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    assert((group >= 1) && (c % group == 0) && (k % group == 0));
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ioh = 0; ioh < oh; ioh++) {
                for (iow = 0; iow < ow; iow++) {
                    for (ik = 0; ik < k_per_group; ik++) {
                        // sliding window for this filter
                        float value = .0f;
                        o_idx = in * oh * ow * k + + ioh * ow * k_per_group + iow * k_per_group + ig * k_per_group + ik;
                        for (ir = 0; ir < fy; ir++) {
                            cur_h = sy * ioh - py + dy * ir;
                            if (cur_h < 0 || cur_h >= h)
                                continue;
                            for (is = 0; is < fx; is++) {
                                cur_w = sx * iow - px + dx * is;
                                if (cur_w < 0 || cur_w >= w)
                                    continue;
                                for (ic = 0; ic < c_per_group; ic++) {
                                    i_idx = in * h * w * c + cur_h * w * c + cur_w * c + ig * c_per_group + ic;
                                    f_idx = ig * k_per_group * fy * fx * c_per_group + ik * fy * fx * c_per_group +
                                                            ir * fx * c_per_group + is * c_per_group + ic;
                                    value += src[i_idx] * filter[f_idx];
                                }
                            }
                        }
                        dst[o_idx] = value;
                    }
                }
            }
        }
    }
}

std::vector<float> load_from_npy(const std::string path) {
  npy::npy_data<float> d = npy::read_npy<float>(path);

  std::vector<float> data = d.data;
  std::vector<unsigned long> shape = d.shape;
  bool fortran_order = d.fortran_order;

  return data;
}

void run(const char *kernel_name, const char *hsaco_file) {

  igemm_fwd_gtc_nhwc_karg_t karg;
  karg.hi = 66;
  karg.wi = 66;
  karg.n = 2;
  karg.k = 1280;
  karg.c = 1280;
  karg.ho = 64;
  karg.wo = 64;
  karg.stride_h = 1;
  karg.stride_w = 1;
  karg.dilation_h = 1;
  karg.dilation_w = 1;
  karg.pad_h = 0;
  karg.pad_w = 0;
  karg.y = 3;
  karg.x = 3;
  karg.group = 1;
  karg.magic_0 = 2576980378;
  karg.magic_1 = 1;
  karg.magic_2 = 1;
  karg.magic_3 = 2576980378;
  karg.magic_4 = 0;
  karg.magic_5 = 0;
  karg.shift_pack_0 = 151391236;
  karg.shift_pack_1 = 0;
  karg.ks = 3;
  karg.__pack_0 = 0;

  // init host side
  auto loaded_input = load_from_npy("/data/home/perf/nithin/test_misa/k5_input_f32.npy");
  auto loaded_weight = load_from_npy("/data/home/perf/nithin/test_misa/k5_filter_f32_nhwc.npy");

  float *host_input = loaded_input.data();
  float *host_weight = loaded_weight.data();

  // float *host_input = (float *)malloc(static_cast<size_t>(karg.n) * karg.c * karg.hi * karg.wi * sizeof(float));
  // float *host_weight = (float *)malloc(static_cast<size_t>(karg.k) * karg.c * karg.y * karg.x * sizeof(float));

  // gen_rand_vector<float, float>(host_input, static_cast<size_t>(karg.n) * karg.c * karg.hi * karg.wi, 0.0, 1.0);
  // gen_rand_vector<float, float>(host_weight, static_cast<size_t>(karg.k) * karg.c * karg.y * karg.x, -0.5, 0.5);
  
  float *host_output = (float *)malloc(static_cast<size_t>(karg.n) * karg.k * karg.ho * karg.wo * sizeof(float));

  float *device_input;
  float *device_weight;
  float *device_output;

  CHECK_HIP_ERROR(hipMalloc(&device_input, static_cast<size_t>(karg.n) * karg.c * karg.hi * karg.wi * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&device_weight, static_cast<size_t>(karg.k) * karg.c * karg.y * karg.x * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&device_output, static_cast<size_t>(karg.n) * karg.k * karg.ho * karg.wo * sizeof(float)));

  void *host_input_dtype;
  void *host_weight_dtype;
  void *host_output_dtype;

  void *device_input_dtype;
  void *device_weight_dtype;
  void *device_output_dtype;

  // for half
  size_t data_byte = 2;

  host_input_dtype  = malloc(karg.n * karg.c * karg.hi * karg.wi * data_byte);
  host_weight_dtype = malloc(karg.k * karg.c * karg.y * karg.x * data_byte);
  host_output_dtype = malloc(karg.n * karg.k * karg.ho * karg.wo * data_byte);

  CHECK_HIP_ERROR(hipMalloc(&device_input_dtype, karg.n * karg.c * karg.hi * karg.wi * data_byte));
  CHECK_HIP_ERROR(hipMalloc(&device_weight_dtype, karg.k * karg.c * karg.y * karg.x * data_byte));
  CHECK_HIP_ERROR(hipMalloc(&device_output_dtype, karg.n * karg.k * karg.ho * karg.wo * data_byte));

  // for(int i=0; i < 10; i++) {
  //   std::cout<<*(host_weight+i) <<", ";
  // }
  // std::cout<<"\n";

  tensor_copy<float16, float>(static_cast<float16*>(host_input_dtype), host_input, static_cast<size_t>(karg.n) * karg.c * karg.hi * karg.wi);
  tensor_copy<float16, float>(static_cast<float16*>(host_weight_dtype), host_weight, static_cast<size_t>(karg.k) * karg.c * karg.y * karg.x);
  
  void *device_output_to_host = malloc(static_cast<size_t>(karg.n) * karg.k * karg.ho * karg.wo * sizeof(float));

  CHECK_HIP_ERROR(hipMemcpy(device_input_dtype, host_input_dtype,
              static_cast<size_t>(karg.n) * karg.c * karg.hi * karg.wi * data_byte, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(device_weight_dtype, host_weight_dtype,
              static_cast<size_t>(karg.k) * karg.c * karg.y * karg.x * data_byte, hipMemcpyHostToDevice));
  
  CHECK_HIP_ERROR(hipMemset(device_output_dtype,
                    0, static_cast<size_t>(karg.n) * karg.k * karg.ho * karg.wo * data_byte));
  
  karg.p_in = (void *)device_input_dtype;
  karg.p_wei = (void *)device_weight_dtype;
  karg.p_out = (void *)device_output_dtype;
  
  // naive conv on cpu
  // naive_conv_fwd_nhwc(host_input, host_weight, host_output, karg.n, karg.wi, karg.hi, karg.c,
  //               karg.k, karg.x, karg.y, karg.pad_w, karg.pad_h, karg.stride_w, karg.stride_h,
  //               karg.dilation_w, karg.dilation_h, karg.group);
  
  // for(int i=0; i < 10; i++) {
  //   std::cout<<*(((float16*)device_output_dtype)+i) <<", ";
  // }
  // std::cout<<"\n";

  hipModule_t module;
  hipFunction_t kernelFunc;
  std::vector<char> hsacoVec = readFileIntoVector(hsaco_file);
  if (hipModuleLoadDataEx(&module, hsacoVec.data(), 0, NULL, NULL) !=
      hipSuccess) {
    std::cout << "Failed to load module!\n";
    return;
  }
  if (hipModuleGetFunction(&kernelFunc, module, kernel_name) != hipSuccess) {
    std::cout << "Failed to get function!\n";
    return;
  }

  size_t arg_size = sizeof(karg);
  std::vector<size_t> grid_size{655360, 1, 1};
  std::vector<size_t> block_size{256, 1, 1};
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void *)&karg,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};
  hipEvent_t start;
  hipEvent_t stop;
  CHECK_HIP_ERROR(hipEventCreate(&start));
  CHECK_HIP_ERROR(hipEventCreate(&stop));
  float ms = .0;

  CHECK_HIP_ERROR(hipExtModuleLaunchKernel(
      kernelFunc, grid_size[0], grid_size[1], grid_size[2], block_size[0],
      block_size[1], block_size[2], 0, 0, NULL, (void **)&config, start, stop));

  CHECK_HIP_ERROR(hipEventSynchronize(stop));
  CHECK_HIP_ERROR(hipEventElapsedTime(&ms, start, stop));
  CHECK_HIP_ERROR(hipEventDestroy(start));
  CHECK_HIP_ERROR(hipEventDestroy(stop));

  std::vector<float> op;
  op.reserve(karg.ho * karg.wo * karg.k * karg.n);
  const std::vector<unsigned long> op_shape{static_cast<unsigned long>(karg.n), static_cast<unsigned long>(karg.ho), static_cast<unsigned long>(karg.wo), static_cast<unsigned long>(karg.k)};

  for(int i=0; i < karg.ho * karg.wo * karg.k * karg.n; i++) {
    // std::cout<<*(((float16*)device_output_dtype)+i) <<", ";
    op[i] = (float)*(((float16*)device_output_dtype)+i);
  }
  // std::cout<<"\n";

  const npy::npy_data_ptr<float> op_npy{op.data(), op_shape, false};
  write_npy("/data/home/perf/nithin/test_misa/out11_ptr_g3.npy", op_npy);

  // free(host_input);
  // free(host_weight);
  free(host_output);

  free(device_output_to_host);

  hipFree(device_input);
  hipFree(device_weight);
  hipFree(device_output);

  free(host_input_dtype);
  free(host_weight_dtype);
  free(host_output_dtype);

  hipFree(device_input_dtype);
  hipFree(device_weight_dtype);
  hipFree(device_output_dtype);
}

int main() {
  // MISA convolution kernel object code compiled for gfx940
  const char *hsaco_file = "/data/home/perf/nithin/MISA/out/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco";
  const char *kernel_name =
      "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs";
  
  run(kernel_name, hsaco_file);

  return 0;
}