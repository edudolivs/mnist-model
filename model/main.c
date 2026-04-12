#include "loader.h"
#include <stdio.h>

int main() {
  tensor_t *trainImages = getTensor();
  tensor_t *trainLabels = getTensor();

  char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";

  loadIdx(trainImages, trainImagesPath);
  loadIdx(trainLabels, trainLabelsPath);

  for (int i = 0; i < 10; i ++) {
    printf("%.0f\n", trainLabels->data[i]);
    displayImage(trainImages, i);
  }

  freeTensor(trainImages);
  return 0;
}
