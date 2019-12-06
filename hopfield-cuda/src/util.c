#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <sys/time.h>

void seed_rng(void) {
    static bool seeded = false;
    if (!seeded) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        int usec = tv.tv_usec;
        srand48(usec);
        seeded = true;
    }
}

void shuffle(size_t *array, size_t n) {
    /* Fisher-Yates shuffle */
    /* https://stackoverflow.com/a/10072899 */
    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
            size_t j = (unsigned int) (drand48()*(i+1));
            size_t t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

/* coerces a float into one of {0.0, 1.0, -1.0, NAN}. */
float clamp_sign(float x) {
    if (x == 0.0f || x == -0.0f) { 
        x = 0.0f;
    }
    else if (x > 0.0f) { 
        x = 1.0f;
    }
    else if (x < 0.0f) {
        x = -1.0f;
    }
    else {
        x = NAN;
    }
    return x;
}
