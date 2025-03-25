/***********************************************************************************************************
*  @File:                    layer_normalization.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   layer_normalization算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    layer_normalization算子执行异或操作
                                    输入输出数据类型为int32，维度为1；
                                    两种rvv寄存器分组模式都为m4
************************************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>

#define DNN_SUCCESS 0


/* // 模拟数据类型标志
#define TypeFlag_kFloat32 1
#define TypeFlag_kFloat64 2 */

/* // 读取属性的模拟函数
int ReadValueWithDefault(void *attributes, float *value, const char *key, float default_value, const char *layer_name) {
  *value = default_value;
  return DNN_SUCCESS;
}

int ReadValue(void *attributes, int *value, const char *key, const char *layer_name) {
  *value = -1; // 默认值
  return DNN_SUCCESS;
} */

static int LayerNormalization(int ndim, const int *dims, const float *X_ptr,
                                    const float *scale_ptr, const float *B_ptr, int axis,float epsilon, 
                                    float *Y_ptr, float *mean_ptr, float *inv_stddev_ptr, int stash_type) {
    int outter_size = 1;
    for (int i = 0; i < axis; ++i) {
        outter_size *= dims[i];
    }
    int inner_size = 1;
    for (int i = axis; i < ndim; ++i) {
        inner_size *= dims[i];
    }

    for (int i = 0; i < outter_size; ++i) {
        const float *iptr = X_ptr + i * inner_size;
        float *optr = Y_ptr + i * inner_size; //Y_ptr[i * inner_size + j]

        // 计算均值
        float sum = 0;
        for (int j = 0; j < inner_size; ++j) {
            sum += iptr[j];
        }
        float mean = sum / inner_size;

        // 计算方差和逆标准差
        float square_sum = 0;
        for (int j = 0; j < inner_size; ++j) {
            square_sum += (iptr[j] - mean) * (iptr[j] - mean);
        }
        float variance = square_sum / inner_size;
        float stddev = sqrtf(variance + epsilon);
        float inv_stddev = 1.0f / stddev;

        if (mean_ptr != NULL && inv_stddev_ptr != NULL) {
            mean_ptr[i] = mean;
            inv_stddev_ptr[i] = inv_stddev;
        }

        // 归一化并应用缩放和偏移
        for (int j = 0; j < inner_size; ++j) {
            float bias = (B_ptr == NULL) ? 0 : B_ptr[j];
            optr[j] = (iptr[j] - mean) * inv_stddev * scale_ptr[j] + bias;
        }
    }

    return DNN_SUCCESS;
}


int LayerNormalization_intr(int ndim, const int *dims, const float *X_ptr,
                                    const float *scale_ptr, const float *B_ptr, int axis,float epsilon, 
                                    float *Y_ptr, float *mean_ptr, float *inv_stddev_ptr, int stash_type) {
    int outter_size = 1;
    for (int i = 0; i < axis; ++i) {
        outter_size *= dims[i];
    }
    int inner_size = 1;
    for (int i = axis; i < ndim; ++i) {
        inner_size *= dims[i];
    }

    for (int i = 0; i < outter_size; ++i) {
        const float *iptr = X_ptr + i * inner_size;
        float *optr = Y_ptr + i * inner_size; //Y_ptr[i * inner_size + j]
        
        // 计算均值
        size_t vl = vsetvl_e32m1(inner_size);
        //float sum = 0;
        vfloat32m1_t intr_sum = vfmv_v_f_f32m1(0.0f, vl); 
        for (int j = 0; j < inner_size; j+=vl) {
            //vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t vinput = vle32_v_f32m1(iptr + j, vl);
            intr_sum =  vfadd_vv_f32m1 (intr_sum, vinput, vl);
        }
        vfloat32m1_t vsum = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), intr_sum, vundefined_f32m1(), vl);
        float mean = vfmv_f_s_f32m1_f32(vfdiv_vf_f32m1(vsum, inner_size, vl));
/*         sum = vfmv_f_s_f32m1_f32(vsum);
        float mean = sum / inner_size; */

        // 计算方差和逆标准差
        //float square_sum = 0;
        vfloat32m1_t intr_square_sum = vfmv_v_f_f32m1(0.0f, vl); 
        for (int j = 0; j < inner_size; j += vl) {
            //vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t v = vle32_v_f32m1(iptr + j, vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vdiff_sq = vfmul_vv_f32m1(vdiff, vdiff, vl);
            intr_square_sum = vfadd_vv_f32m1(intr_sum, vdiff_sq, vl);
        }
        vfloat32m1_t vsquare_sum = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), intr_square_sum, vundefined_f32m1(), vl);
        float inv_stddev = vfmv_f_s_f32m1_f32(vfrec7_v_f32m1(vfsqrt_v_f32m1( vfdiv_vf_f32m1(vsquare_sum, inner_size, vl), vl), vl));
  /*         square_sum = vfmv_f_s_f32m1_f32(vsquare_sum);
          float variance = square_sum / inner_size;
          float stddev = sqrtf(variance + epsilon);
          float inv_stddev = 1.0f / stddev; */

        if (mean_ptr != NULL && inv_stddev_ptr != NULL) {
            mean_ptr[i] = mean;
            inv_stddev_ptr[i] = inv_stddev;
        }

        // 归一化并应用缩放和偏移
        // optr[j] = (iptr[j] - mean) * inv_stddev * scale_ptr[j] + bias;
        for (int j = 0; j < inner_size; j += vl) {
            //vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t v = vle32_v_f32m1(iptr + j, vl);
            vfloat32m1_t vscale = vle32_v_f32m1(scale_ptr + j, vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vnorm = vfmul_vf_f32m1(vdiff, inv_stddev, vl);
            vnorm = vfmul_vv_f32m1(vnorm, vscale, vl);
            if (B_ptr != NULL) {
                vfloat32m1_t vbias = vle32_v_f32m1(B_ptr + j, vl);
                vnorm = vfadd_vv_f32m1(vnorm, vbias, vl);
            }
            vse32_v_f32m1(optr + j, vnorm, vl);
        }
    }

    return DNN_SUCCESS;
}

/* 均值计算：使用vfredosum_vs_f32m1矢量指令对矢量段进行求和。
方差计算：使用矢量指令计算每个元素与均值的差异平方，并对这些平方和进行求和。
归一化处理：使用矢量指令对每个元素进行标准化、缩放和偏移操作。 */
/* void normalize_rvv(float *iptr, float *mean_ptr, float *inv_stddev_ptr, float *scale_ptr, float *B_ptr, float *optr, int inner_size, int outer_size, float epsilon) {
    for (int i = 0; i < outer_size; ++i) {
        float *inner_iptr = iptr + i * inner_size;
        float *inner_optr = optr + i * inner_size;

        // 计算均值
        float sum = 0;
        size_t vl;
        for (int j = 0; j < inner_size; j += vl) {
            vl = vsetvl_e32m1(inner_size - j);  // 设置矢量长度
            vfloat32m1_t v = vle32_v_f32m1(inner_iptr + j, vl);
            vfloat32m1_t vsum = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), v, vl);
            sum += vfmv_f_s_f32m1_f32(vsum);
        }
        float mean = sum / inner_size;

        // 计算方差和逆标准差
        float square_sum = 0;
        for (int j = 0; j < inner_size; j += vl) {
            vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t v = vle32_v_f32m1(inner_iptr + j, vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vdiff_sq = vfmul_vv_f32m1(vdiff, vdiff, vl);
            vfloat32m1_t vsquare_sum = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), vdiff_sq, vl);
            square_sum += vfmv_f_s_f32m1_f32(vsquare_sum);
        }
        float variance = square_sum / inner_size;
        float stddev = sqrtf(variance + epsilon);
        float inv_stddev = 1.0f / stddev;

        if (mean_ptr != NULL && inv_stddev_ptr != NULL) {
            mean_ptr[i] = mean;
            inv_stddev_ptr[i] = inv_stddev;
        }

        // 归一化并应用缩放和偏移
        for (int j = 0; j < inner_size; j += vl) {
            vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t v = vle32_v_f32m1(inner_iptr + j, vl);
            vfloat32m1_t vscale = vle32_v_f32m1(scale_ptr + j, vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vnorm = vfmul_vf_f32m1(vdiff, inv_stddev, vl);
            vnorm = vfmul_vv_f32m1(vnorm, vscale, vl);
            if (B_ptr != NULL) {
                vfloat32m1_t vbias = vle32_v_f32m1(B_ptr + j, vl);
                vnorm = vfadd_vv_f32m1(vnorm, vbias, vl);
            }
            vse32_v_f32m1(inner_optr + j, vnorm, vl);
        }
    }
} */


int main() {
  // 初始化变量
  int dims[] = {2, 3, 4, 5}; // 例如 batch_size=2, height=3, width=4, channels=5
  int ndim = 4;

  // 输入
  float X[120]; // 输入数据
  float scale[5]; // 缩放参数，假设axis=-1
  float B[5]; // 偏置参数
  // 输出
  float Y[120], Y_i[120], Y_a[120]; // 输出数据
  float mean[24], mean_i[24], mean_a[24]; // 均值输出
  float inv_stddev[24], inv_stddev_i[24], inv_stddev_a[24]; // 逆标准差输出

  // 填充一些测试数据
  for (int i = 0; i < 120; ++i) X[i] = (float)(rand() % 100) / 100.0f;
  for (int i = 0; i < 5; ++i) scale[i] = 1.0f;
  for (int i = 0; i < 5; ++i) B[i] = 0.0f;

  // 设置属性
  float epsilon = 1e-5f;
  int axis = -1;
  int stash_type = 1; //Mean 和 InvStdDev的计算精度，float16/float32

  // 处理axis负值
  if (axis < 0) {
    axis += ndim;
  }

  // 调用LayerNormalization函数
  int status = LayerNormalization(ndim, dims, X, scale, B, axis, epsilon, Y, mean, inv_stddev, stash_type);
  if (status != DNN_SUCCESS) {
    printf("Layer normalization failed\n");
    return -1;
  }

/*   // 调用LayerNormalization函数
  int status_intr = LayerNormalization_intr(ndim, dims, X, scale, B, axis, epsilon, Y_i, mean_i, inv_stddev_i, stash_type);
  if (status_intr != DNN_SUCCESS) {
    printf("Layer normalization failed\n");
    return -1;
  } */

  printf("Input X:\n");
  for (int i = 0; i < 120; ++i) {
    printf("%f ", X[i]);
    if ((i + 1) % 5 == 0) printf("\n");
  }

  // 输出结果
  printf("\nOutput Y:\n");
  for (int i = 0; i < 120; ++i) {
    printf("%f ", Y[i]);
    if ((i + 1) % 5 == 0) printf("\n");
  }

  printf("\nOutput Y_i:\n");
  for (int i = 0; i < 120; ++i) {
    printf("%f ", Y_i[i]);
    if ((i + 1) % 5 == 0) printf("\n");
  }

  printf("\n");

  return 0;
}
