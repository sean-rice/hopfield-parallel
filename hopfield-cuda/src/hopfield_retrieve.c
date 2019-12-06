#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/hopfield_retrieve.h"
#include "../include/hopfield_network.h"
#include "../include/hopfield_pattern.h"
#include "../include/util.h"

void range_fill(size_t *array, size_t start, size_t end) {
    for (size_t value = start; value < end; ++value) {
        size_t index = value - start;
        array[index] = value;
    }
}

/* vector-vector dot product */
float vvdot(float *a, float *b, size_t n) {
    float z = 0;
    for (size_t i = 0; i < n; ++i) {
        z += a[i] * b[i];
    }
    return z;
}

/* scalar-vector "dot product" */
float svdot(float a, float *b, size_t n) {
    float z = 0;
    for (size_t i = 0; i < n; ++i) {
        float term = a * b[i];
        z += term;
    }
    return z;
}

bool hopfield_retrieve(hopfield_network *net, hopfield_pattern *pattern) {
    if (net->size != pattern->size) {
        return false;
    }

    size_t *indices = malloc(net->size * sizeof(size_t));
    if (NULL == indices) {
        return false;
    }
    range_fill(indices, 0, net->size);

    bool did_update = true;
    const int32_t MAX_ITER = 1000;
    int32_t iter = 0;
    while (iter < MAX_ITER && did_update) {
        shuffle(indices, net->size);
        did_update = false;
        for (size_t i = 0; i < net->size; ++i) {
            size_t i_neuron = indices[i];

            float neuron_value = pattern->data[i_neuron];
            float *neuron_weights = ((net->weights) + net->size * i_neuron);
            float z = vvdot(pattern->data, neuron_weights, net->size);
            float z_clamped = clamp_sign(z);
            if (NAN == z_clamped) {
                fprintf(stderr, "error: got NAN for z_clamped; z = %f", z);
                return false;
            }
            if (z_clamped != neuron_value) {
                pattern->data[i_neuron] = z_clamped;
                did_update = true;
            }
        }
        iter += 1;
    }
    return true;
}
