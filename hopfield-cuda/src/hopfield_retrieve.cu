extern "C" {

#include <stdbool.h>
#include <stdint.h>

#include "../include/hopfield_retrieve.h"
#include "../include/hopfield_network.h"
#include "../include/hopfield_pattern.h"

#define checkCudaErrors(x) x

#define ITER_AMOUNT 100

__device__ void vvdot(int32_t n, float *a, float *b, float* out) {
    float z = 0;
    for (int32_t i = 0; i < n; ++i) {
        z += a[i] * b[i];
    }
    *out = z;
}

__global__ void hopfield_retrieve_kernel(int32_t size, float *weights, float *pattern_data) {
    int32_t i_neuron = blockIdx.x;

    /* cause a delay to desync blocks a bit (does this work???) */
    float dummy = 1.0;
    for (int32_t i = 0; i < i_neuron+999999; ++i) {
        dummy += ((float) i) / dummy;
    }

    float *neuron_weights = (weights + size*i_neuron);

    int32_t iters_remaining = ITER_AMOUNT;
    float z_old = 0.0f;
    float z_clamped_old = 0.0f;

    while (iters_remaining > 0) {
        --iters_remaining;
        float z = 0.0f;
        vvdot(size, pattern_data, neuron_weights, &z);
        float z_clamped = copysignf(1.0f, z);
        pattern_data[i_neuron] = z_clamped;
        if (z_clamped != z_clamped_old || fabs(z_old - z) >= 1e-5f) {
            iters_remaining = ITER_AMOUNT;
        }
        z_old = z;
        z_clamped_old = z_clamped;
    }
}

bool hopfield_retrieve(hopfield_network *net, hopfield_pattern *pattern) {
    cudaError status = cudaSuccess;
    bool result = false;
    float* d_net_weights = NULL;
    float* d_pattern_data = NULL;
    int32_t size = net->size;
    int32_t n_weights = size*size;

    if (net->size != pattern->size) {
        goto cleanup_exit;
    }
    status = checkCudaErrors( cudaMalloc(&d_net_weights, n_weights * sizeof(float)) );
    if (status != cudaSuccess) {
        fprintf(stderr, "cuda error (%i): in cudaMalloc for d_net_weights\n", status);
        result = false;
        goto cleanup_exit;
    }
    status = checkCudaErrors( cudaMalloc(&d_pattern_data, size * sizeof(float)) );
    if (status != cudaSuccess) {
        fprintf(stderr, "cuda error (%i): in cudaMalloc for d_pattern_data\n", status);
        result = false;
        goto cleanup_exit;
    }

    status = checkCudaErrors( cudaMemcpy(d_net_weights, net->weights, n_weights*sizeof(float), cudaMemcpyHostToDevice) );
    if (status != cudaSuccess) {
        fprintf(stderr, "cuda error (%i): in cudaMemcpy for d_net_weights\n", status);
        result = false;
        goto cleanup_exit;
    }
    
    status = checkCudaErrors( cudaMemcpy(d_pattern_data, pattern->data, size*sizeof(float), cudaMemcpyHostToDevice) );
    if (status != cudaSuccess) {
        fprintf(stderr, "cuda error (%i): in cudaMemcpy for d_pattern_data\n", status);
        result = false;
        goto cleanup_exit;
    }

    /* finally, our kernel call! */
    hopfield_retrieve_kernel<<<size, 1>>>(size, d_net_weights, d_pattern_data);
    
    status = checkCudaErrors( cudaMemcpy(pattern->data, d_pattern_data, size*sizeof(float), cudaMemcpyDeviceToHost) );
    if (status != cudaSuccess) {
        fprintf(stderr, "cuda error (%i): in cudaMemcpy for d_pattern_data to host\n", status);
        goto cleanup_exit;
    }
    result = true;

cleanup_exit:
    if (d_net_weights) {
        status = cudaFree(d_net_weights);
        if (status != cudaSuccess) {
            fprintf(stderr, "cuda error (%i): in cudaFree for d_net_weights\n", status);
        }
    }
    if (d_pattern_data) {
        status = cudaFree(d_pattern_data);
        if (status != cudaSuccess) {
            fprintf(stderr, "cuda error (%i): in cudaFree for d_pattern_data\n", status);
        }
    }
    return result;
}

} // extern "C"
