//  Copyright (C) 2023 Raphael Javaux
//  raphaeljavaux@gmail.com
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 3 of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program; if not, write to the Free Software Foundation,
//  Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#ifdef USE_FLOAT16_BUFFERS
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    typedef half native_float;
#else
    typedef float native_float;
#endif

__kernel void corr_coeffs(
    __global const int *offsets,
    __global const native_float *ref,
    __global const char *ref_mask,
    const int ref_size,
    __global const native_float *target,
    __global const char *target_mask,
    const int target_size,
    __global float *coeffs,
    __global int *match_lens
);


void _corr_coeff(
    __global const native_float *a,
    __global const char *mask_a,
    __global const native_float *b,
    __global const char *mask_b,
    int size,
    __global float *coeff,
    __global int *match_len
);


__kernel void corr_coeffs(
    __global const int *offsets,
    __global const native_float *ref,
    __global const char *ref_mask,
    const int ref_size,
    __global const native_float *target,
    __global const char *target_mask,
    const int target_size,
    __global float *coeffs,
    __global int *match_lens
)
{
    int i = get_global_id(0);

    int offset = offsets[i];

    int ref_offset = max(0, offset);
    int target_offset = max(0, -offset);

    int size = min(ref_size - ref_offset, target_size - target_offset);

    const __global native_float *ref_begin = ref + ref_offset;
    const __global char *ref_mask_begin = ref_mask + ref_offset;
    const __global native_float *target_begin = target + target_offset;
    const __global char *target_mask_begin = target_mask + target_offset;

    _corr_coeff(
        ref_begin, ref_mask_begin, target_begin, target_mask_begin, size, &coeffs[i], &match_lens[i]
    );
}


void _corr_coeff(
    __global const native_float *a,
    __global const char *mask_a,
    __global const native_float *b,
    __global const char *mask_b,
    int size,
    __global float *coeff,
    __global int *match_len
)
{
    // Means and number of masked values

    int match_len_local = 0;
    float mean_a = 0.0f;
    float mean_b = 0.0f;

    for (int i = 0; i < size; ++i) {
        char mask = mask_a[i] & mask_b[i];

        match_len_local += mask;

        mean_a += a[i] * mask;
        mean_b += b[i] * mask;
    }

    if ((float)match_len_local / size < 0.2f) {
        *coeff = NAN;
        *match_len = 0;
        return;
    }

    mean_a /= match_len_local;
    mean_b /= match_len_local;

    // Cov and stdevs

    float std_a = 0.0f;
    float std_b = 0.0f;
    float cov = 0.0f;

    for (int i = 0; i < size; ++i) {
        char mask = mask_a[i] & mask_b[i];

        float diff_a = (a[i] - mean_a) * mask;
        float diff_b = (b[i] - mean_b) * mask;

        std_a += diff_a * diff_a;
        std_b += diff_b * diff_b;
        cov += diff_a * diff_b;
    }

    std_a = sqrt(std_a / match_len_local);
    std_b = sqrt(std_b / match_len_local);
    cov /= match_len_local;

    // Writes results.

    *coeff = cov / (std_a * std_b);
    *match_len = match_len_local;
}