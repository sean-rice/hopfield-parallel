#include <stdio.h>
#include <stdlib.h>

#include "../include/hopfield_network.h"

hopfield_network load_hopfield_network(FILE *fp) {
    hopfield_network net;
    net.magic[0] = -1;
    net.magic[1] = -1;
    net.size = -1;
    net.weights = NULL;

    /* read magic */
    if (2 != fread(net.magic, sizeof(net.magic[0]), 2, fp)) {
        fprintf(stderr, "error reading magic.");
        goto net_load_err;
    }
    if (!check_hopfield_net_magic(&net)) {
        goto net_load_err;
    }

    /* read size */
    if (1 != fread(&net.size, sizeof(net.size), 1, fp)) {
        goto net_load_err;
    }

    /* allocate weight matrix */
    int n_weights = net.size * net.size;
    float *weights = (float*) malloc(sizeof(float) * n_weights);
    if (NULL == weights) {
        goto net_load_err;
    }

    /* read weights */
    if (n_weights != fread(weights, sizeof(weights[0]), n_weights, fp)) {
        goto net_load_err;
    }

    /* set weights */
    net.weights = weights;
    return net;

net_load_err:
    /* for now, just return bad net; onus is on caller to check it's ok */
    return net;
}

bool check_hopfield_net_magic(hopfield_network* net) {
    return net->magic[0] == HOPFIELD_MAGIC_0 && net->magic[1] == HOPFIELD_MAGIC_1;
}

bool hopfield_net_ok(hopfield_network* net) {
    if (!check_hopfield_net_magic(net)) {
        return false;
    }
    if (1 > net->size) {
        return false;
    }
    if (NULL == net->weights) {
        return false;
    }
    return true;
}
