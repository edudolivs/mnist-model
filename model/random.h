#ifndef RANDOM
#define RANDOM

#include <stdint.h>

void seed(uint32_t s);

uint32_t randUint32();

float randFloat();

float randGauss();

int shuffleArray(uint32_t *array, uint32_t len);

void printGauss();

#endif
