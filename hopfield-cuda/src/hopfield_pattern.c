#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/hopfield_pattern.h"
#include "../include/util.h"

hopfield_pattern load_hopfield_pattern(FILE *fp) {
    hopfield_pattern pattern;
    pattern.size = -1;
    pattern.data = NULL;

    /* read size */
    if (1 != fread(&pattern.size, sizeof(pattern.size), 1, fp)) {
        goto pattern_load_err;
    }

    /* allocate data buffer */
    float *data = malloc(pattern.size * sizeof(float));
    if (NULL == data) {
        goto pattern_load_err;
    }
    for (size_t i = 0; i < pattern.size; ++i) { data[i] = NAN; }

    /* read example data */
    if (pattern.size != fread(data, sizeof(data[0]), pattern.size, fp)) {
        goto pattern_load_err;
    }

    /* set data, clamp to {NAN, -1, 0, 1} */
    for (size_t i = 0; i < pattern.size; ++i) {
        float value = clamp_sign(data[i] - 0.5);
        data[i] = value;
        if (NAN == value) {
            /* any fp edge cases / malformed data? bail. */
            fprintf(stderr, "warning: edge case clamping pattern data[%zu]; value = %f", i, value);
            goto pattern_load_err;
        }
    }
    pattern.data = data;
    return pattern;

pattern_load_err:
    return pattern;
}

bool hopfield_pattern_ok(hopfield_pattern *pattern) {
    if (1 > pattern->size) {
        return false;
    }
    for (size_t i = 0; i < pattern->size; ++i) {
        if (pattern->data[i] == NAN) {
            return false;
        }
    }
    return true;
}

void print_seq(char c, size_t n, bool newline) {
    for (size_t i = 0; i < n; ++i) {
        fprintf(stdout, "%c", c);
    }
    if (newline) { fprintf(stdout, "\n"); }
}

void print_pattern(hopfield_pattern *pattern) {
    /* TODO: expand to not just assuming a square image. */
    int32_t w = (int32_t) floorf(sqrtf((float) pattern->size));
    int32_t h = w;

    fprintf(stdout, " ");
    print_seq('-', 2*w, true);
    for (int32_t row = 0; row < h; ++row) {
        fprintf(stdout, "|");
        for (int32_t col = 0; col < w; ++col) {
            int32_t off = row*w + col;
            float val = *(pattern->data + off);
            char c = '?';
            if (val > 0.0) { c = '#'; }
            else if (val < 0.0) { c = ' '; }
            fprintf(stdout, "%c%c", c, c);
        }
        fprintf(stdout, "|\n");
    }
    fprintf(stdout, " ");
    print_seq('-', 2*w, true);
}
