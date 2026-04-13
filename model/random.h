#ifndef RANDOM
#define RANDOM

#include <stdint.h>

void seed(uint32_t s);

uint32_t randUint32();

float randFloat();

float randGauss();

void printGauss();

#endif
