#include "random.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

uint64_t state;

void seed(uint64_t s) {
  state = s;
}

uint64_t randUint64() {
  state ^= state << 12;
  state ^= state >> 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}

double randDouble() {
  return (randUint64() >> 11) * (1.0 / 9007199254740992.0);
}

double randGauss() {
  double u1 = 1 - randDouble();
  double u2 = randDouble();

  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

void printGauss() {
  seed(time(NULL));

  double val;
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
