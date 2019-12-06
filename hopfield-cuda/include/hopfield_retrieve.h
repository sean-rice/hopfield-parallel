#ifndef _HOPFIELD_RETRIEVE_H_
#define _HOPFIELD_RETRIEVE_H_

#include <stdbool.h>

#include "./hopfield_network.h"
#include "./hopfield_pattern.h"

bool hopfield_retrieve(hopfield_network *net, hopfield_pattern *pattern);

#endif // #ifndef _HOPFIELD_RETRIEVE_H_
