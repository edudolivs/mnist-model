#include "random.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

uint32_t state;

void seed(uint32_t s) {
  state = s;
}

uint32_t randUint32() {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state >> 5;
  return state * 1597334677U;
}

float randFloat() {
  uint32_t x = (randUint32() >> 9) | 0x3f800000;
  float f;
  memcpy(&f, &x, sizeof(f));
  return f - 1.0f;
}

float randGauss() {
  float u1 = 1 - randFloat();
  float u2 = randFloat();

  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

void printGauss() {
  seed(time(NULL));

  float val;
  float n = 100000;
  float std = 6;
  float h = 32;
  int bins = 30;

  float distribution[bins];
  for (int i = 0; i < bins; i++) {
    distribution[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    val = randGauss() * std;
    for (int j = 0; j < bins; j ++) {
      int k = j - (bins / 2);
      if (val > k & val <= k + 1) {
        distribution[j] += 1;
        break;
      }
    }
  }

  int max = 0;
  for (int i = 0; i < bins; i++) {
    if (distribution[i] > max) {
      max = distribution[i];
    }
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < bins; j++) {
      if (distribution[j] / max > (h - i) / h) {
        printf("0000 ");
      } else {
        printf(".... ");
      }
    }
    printf("\n");
  }

  for (int i = 0; i < bins; i++) {
    printf("%04.0f ", distribution[i]);
  }
  printf("\n");
}
