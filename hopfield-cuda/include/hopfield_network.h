#ifndef _HOPFIELD_NETWORK_H_
#define _HOPFIELD_NETWORK_H_

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

/* magic contants are "flipped" for endianness reasons */
/* could probably fix this by not comparing them as int32 but works for now */
#define HOPFIELD_MAGIC_0 0x46504f48
#define HOPFIELD_MAGIC_1 0x444c4549

typedef struct {
    int32_t magic[2];
    int32_t size;
    float *weights; // address with: *(weights + size*r + c) == w[r][c]
} hopfield_network;

hopfield_network load_hopfield_network(FILE *fp);

bool check_hopfield_net_magic(hopfield_network* net);
bool hopfield_net_ok(hopfield_network* net);

#endif // #ifndef _HOPFIELD_NETWORK_H_
