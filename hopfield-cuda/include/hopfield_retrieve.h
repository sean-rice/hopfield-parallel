#ifndef _HOPFIELD_RETRIEVE_H_
#define _HOPFIELD_RETRIEVE_H_

#if defined(__CUDACC__) || defined(__cplusplus)
extern "C" {
#endif

#include <stdbool.h>

#include "./hopfield_network.h"
#include "./hopfield_pattern.h"

bool hopfield_retrieve(hopfield_network *net, hopfield_pattern *pattern);

#if defined(__CUDACC__) || defined(__cplusplus)
} // extern "C"
#endif

#endif // #ifndef _HOPFIELD_RETRIEVE_H_
