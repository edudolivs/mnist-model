#include "loader.h"
#include <stdio.h>

int main() {
  array_t *trainImages = getArray();
  array_t *trainLabels = getArray();

  char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";

  loadIdx(trainImages, trainImagesPath);
  loadIdx(trainLabels, trainLabelsPath);

  for (int i = 0; i < 10; i ++) {
    printf("%.0f\n", trainLabels->data[i]);
    displayImage(trainImages, i);
  }

  freeArray(trainImages);
  return 0;
}
