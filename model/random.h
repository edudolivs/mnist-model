#ifndef RANDOM
#define RANDOM

#include <stdint.h>

void seed(uint64_t s);

uint64_t randUint64();

double randDouble();

double randGauss();

void printGauss();

#endif
