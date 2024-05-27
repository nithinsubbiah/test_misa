#pragma once

#include <cstdlib>
#include <functional>

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <float.h>
#include <cmath>
#include <algorithm>
#include <limits>

#include "half.hpp"

using float16 = half_float::half;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                                                \
  if (status != hipSuccess) {                                                  \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(status), status, __FILE__, __LINE__);            \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

template <typename T>
struct distribution_t{
};

template <>
struct distribution_t<int>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int operator()(URNG & rng){ return distribution(rng);}
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<float>{
    distribution_t(float min, float max) : distribution(min, max) {}
    template<class URNG>
    float operator()(URNG & rng){ return distribution(rng);}
    std::uniform_real_distribution<float> distribution;
};

template <typename Dst_T, typename Src_T>
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, size_t total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Src_T> distribution(min,max);
    for (size_t i = tid; i < total_size; i += block_size) {
        p[i] = static_cast<Dst_T>(scale * distribution(rng));
    }
}

template <typename Dst_T, typename Src_T>
void gen_rand_vector(Dst_T *vec, size_t vec_size, Src_T fmin, Src_T fmax, Src_T scale = 1) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_rand_generator<Dst_T, Src_T>,
            vec, t, num_threads, vec_size, fmin, fmax, scale));
    }
    for (auto &th : threads)
        th.join();
}


#include <stddef.h>

typedef struct
{
    union
    {
        int8_t v;
        struct
        {
            int lo : 4;
            int hi : 4;
        };
    };
}int4x2_t;

template <typename Dst_T, typename Src_T>
void block_wise_tensor_copy(Dst_T *p_dst, Src_T *p_src, int tid, size_t block_size, size_t total_size)
{
    for (int i = tid; i < total_size; i += block_size) {
        p_dst[i] = static_cast<Dst_T>(p_src[i]);
    }
}

template <>
void block_wise_tensor_copy<int4x2_t, float>(int4x2_t *p_dst, float *p_src, int tid, size_t block_size, size_t total_size)
{
    // sizeof(int4x2_t) is 4. So need to find a way to avoid seg fault
    int8_t *tmp_dst = (int8_t*)(p_dst);
    for (int i = tid; i < (total_size / 2); i += block_size) {
        int8_t lo = static_cast<int8_t>(p_src[2 * i]);
        int8_t hi = static_cast<int8_t>(p_src[2 * i + 1]);

        lo = lo & 0xf;
        hi = hi & 0xf;

        int8_t composed = (hi << 4) + lo;
        tmp_dst[i] = composed;
    }
}

template <typename Dst_T, typename Src_T>
void tensor_copy(Dst_T *p_dst, Src_T *p_src, size_t tensor_size) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_tensor_copy<Dst_T, Src_T>,
            p_dst, p_src, t, num_threads, tensor_size));
    }
    for (auto &th : threads)
        th.join();
}
