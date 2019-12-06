#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/hopfield_retrieve.h"
#include "../include/hopfield_network.h"
#include "../include/hopfield_pattern.h"
#include "../include/util.h"

#define LOAD_PATH_SIZE 1024

int main(int argc, char *argv[]) {
    if (2 != argc) {
        return -1;
    }

    char* fname = argv[1];
    FILE *fp = fopen(fname, "r");
    if (NULL == fp) {
        fprintf(stderr, "error: could not find file: %s\n", fname);
        return -1;
    }

    hopfield_network net = load_hopfield_network(fp);
    fclose(fp);

    bool net_ok = hopfield_net_ok(&net);
    printf("net_ok: %s\n", net_ok ? "true" : "false");

    seed_rng();

    char load_path[LOAD_PATH_SIZE] = {0};
    while (1) {
        memset(load_path, '\0', LOAD_PATH_SIZE);
        fprintf(stdout, "path to pattern (! to quit): ");
        fgets(load_path, LOAD_PATH_SIZE, stdin);
        load_path[strcspn(load_path, "\r\n")] = 0; /* delete newline */
        if (load_path[0] == '!') { break; }

        /* pattern loading */
        FILE *fp = fopen(load_path, "r");
        if (NULL == fp) {
            fprintf(stdout, "file not found: %s\n", load_path);
            continue;
        }
        hopfield_pattern pattern = load_hopfield_pattern(fp);
        bool pattern_ok = hopfield_pattern_ok(&pattern);
        printf("pattern_ok: %s\n", pattern_ok ? "true" : "false");
        if (pattern_ok == true) {
            print_pattern(&pattern);
        }

        /* network retrieval */
        hopfield_retrieve(&net, &pattern);
        bool retrieved_ok = hopfield_pattern_ok(&pattern);
        printf("retrieved_ok: %s\n", retrieved_ok ? "true" : "false");
        if (retrieved_ok == true) {
            print_pattern(&pattern);
        }
    }
    fprintf(stdout, "done!\n");
    return 0;
}
