/***********************************************************************************************************
*  @File:                    layer_normalization.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   layer_normalization算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    layer_normalization算子执行异或操作
                                    输入输出数据类型为int32，维度为1；
                                    两种rvv寄存器分组模式都为m4
************************************************************************************************************/


#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#include "layer_normalization.h"

#include "layer_common.h"
#include "layer_registry.h"
#include "util/common.h"
#include "util/math/math_functions.h"

//初始化函数，从属性中读取 epsilon 和 axis 的值。
int LayerNormalization_Init(VarientMap const *attributes, float *epsilon, int *axis) {
  int status = ReadValueWithDefault(attributes, epsilon, "epsilon", 1e-5f, "LayerNormalization");
  if (status != 0) return status;
  status = ReadValue(attributes, axis, "axis", "LayerNormalization");
  return status;
}

//主要的归一化操作，包括计算均值和方差，然后归一化每个值。
static int LayerNormalizationHelper(TShape const *X_shape, const float *X_ptr,
                                    const float *scale_ptr, NDArray const *B, int axis,
                                    float epsilon, float *Y_ptr) {
    // 计算外部尺寸和内部尺寸
    int outter_size = 1;
    for (int i = 0; i < axis; ++i) {
        outter_size *= (int)(X_shape->dims[i]);
    }
    int inner_size = 1;
    for (int i = axis; i < X_shape->ndim; ++i) {
        inner_size *= (int)(X_shape->dims[i]);
    }

    // 计算均值
    for (int i = 0; i < outter_size; ++i) {
        const float *iptr = X_ptr + i * inner_size;
        float *optr = Y_ptr + i * inner_size;

        float sum = 0;
        for (int j = 0; j < inner_size; ++j) {
            sum += iptr[j];
        }
        float mean = sum / (float)inner_size;

        // 计算方差
        float square_sum = 0;
        for (int j = 0; j < inner_size; ++j) {
            square_sum += (iptr[j] - mean) * (iptr[j] - mean);
        }
        float variable = square_sum / (float)inner_size;
        variable = 1.0f / sqrtf(variable + epsilon);

        // 阶段2: 标准化，缩放和移位
        for (int j = 0; j < inner_size; ++j) {
            float bias = (B == NULL) ? 0 : B->Dptr[j];
            optr[j] = (iptr[j] - mean) * variable * scale_ptr[j] + bias;
        }
    }
    return 0;  // DNN_SUCCESS
}


static int ValidInputandOutput(TypeFlag from_type, NDArray *scale,
                               NDArray *B, NDArray *Y) {
  if (from_type != Y->dtype) return -1;  // Error: X's dtype is not equal Y's dtype
  if (from_type != scale->dtype) return -1;  // Error: X's dtype is not equal scale's dtype
  if (B != NULL && from_type != B->dtype) return -1;  // Error: X's dtype is not equal B's dtype
  return 0;  // DNN_SUCCESS
}

int LayerNormalization_Forward(std::vector<NDArray *> const *bottom_blobs,
                               std::vector<NDArray *> *top_blobs,
                               float epsilon, int axis) {
  NDArray const *X = (*bottom_blobs)[(size_t)layer_normalization_LayerInputs_kX];
  NDArray *scale = (*bottom_blobs)[(size_t)layer_normalization_LayerInputs_kScale];
  NDArray *B = (bottom_blobs->size() == 3) ? (*bottom_blobs)[(size_t)layer_normalization_LayerInputs_kB] : NULL;
  NDArray *Y = (*top_blobs)[(size_t)layer_normalization_LayerOutputs_kY];

  axis = X->CanonicalAxis(axis);
  TypeFlag from_type = X->dtype;

  int status = ValidInputandOutput(from_type, scale, B, Y);
  if (status != 0) return status;

  switch (from_type) {
    case TypeFlag_kFloat32:
      return LayerNormalizationHelper(X->shape, X->Dptr_float, scale->Dptr_float, B, axis, epsilon, Y->Dptr_float);
    case TypeFlag_kFloat64:
      return LayerNormalizationHelper(X->shape, (float *)X->Dptr_double, (float *)scale->Dptr_double, B, axis, epsilon, (float *)Y->Dptr_double);
    default:
      return -1;  // Error: values dtype only support float32 and float64
  }
}
