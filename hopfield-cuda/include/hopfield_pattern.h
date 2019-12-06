#ifndef _HOPFIELD_PATTERN_H_
#define _HOPFIELD_PATTERN_H_

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
    int32_t size;
    float *data;
} hopfield_pattern;

hopfield_pattern load_hopfield_pattern(FILE *fp);

bool hopfield_pattern_ok(hopfield_pattern *pattern);
void print_pattern(hopfield_pattern *pattern);

#endif // #ifndef _HOPFIELD_PATTERN_H_
