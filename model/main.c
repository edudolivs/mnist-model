#include "loader.h"
#include "tensor.h"
#include "random.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

int main() {
  seed(time(NULL));

  // char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  // char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";
  //
  // tensor_t *trainImages = loadIdx(trainImagesPath);
  // tensor_t *trainLabels = loadIdx(trainLabelsPath);
  //
  // uint32_t imageId = randUint32() % 60000;
  //
  // printf("label: %.0f\n", trainLabels->data[imageId]);
  // tensor_t *view = getView(trainImages, imageId);
  // displayImage(view);
  //
  // freeTensor(view);
  //
  // freeTensor(trainImages);
  // freeTensor(trainLabels);

  uint32_t shape[] = {3, 3};
  tensor_t *a = getTensor(2, shape);
  fillGaussTensor(a);
  display2dTensor(a);
  tensor_t *b = getTensor(2, shape);
  fillGaussTensor(b);
  display2dTensor(b);

  tensor_t *prod = getTensor(a->dim, a->shape);
  multiply2dTensor(prod, a, b);
  display2dTensor(prod);

  float filler = (float)(randUint32() % 3) - 1;
  fillTensor(a, filler);
  display2dTensor(a);
  addTensor(prod, a);

  display2dTensor(prod);

  freeTensor(b);
  freeTensor(a);
  freeTensor(prod);

  return 0;
}
