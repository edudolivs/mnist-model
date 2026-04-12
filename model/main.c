#include "loader.h"
#include "tensor.h"
#include "random.h"
#include <stdio.h>
#include <time.h>

int main() {
  tensor_t *trainImages = getNullTensor();
  tensor_t *trainLabels = getNullTensor();

  char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";

  loadIdx(trainImages, trainImagesPath);
  loadIdx(trainLabels, trainLabelsPath);

  seed(time(NULL));

  int imageId;
  for (int i = 0; i < 10; i ++) {
    imageId = randUint64() % 60000;

    printf("%.0f\n", trainLabels->data[imageId]);
    displayImage(trainImages, imageId);
  }

  freeTensor(trainImages);

  return 0;
}
