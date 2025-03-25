/***********************************************************************************************************
*  @File:                    layer_normalization.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   layer_normalization算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    layer_normalization算子执行异或操作
                                    输入输出数据类型为int32，维度为1；
                                    两种rvv寄存器分组模式都为m1
************************************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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


/* 均值计算：使用vfredosum_vs_f32m1矢量指令对矢量段进行求和。
方差计算：使用矢量指令计算每个元素与均值的差异平方，并对这些平方和进行求和。
归一化处理：使用矢量指令对每个元素进行标准化、缩放和偏移操作。 */
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
        float sum = 0;
        size_t vl_end = (inner_size)%(vl);
        vfloat32m1_t vsum = vfmv_v_f_f32m1(0.0f, vl); 
        vfloat32m1_t intr_sum = vfmv_v_f_f32m1(0.0f, vl); 
        for (int j = 0; j < inner_size-vl_end; j+=vl) {
            //vl = vsetvl_e32m1(inner_size - j);
            vfloat32m1_t vinput = vle32_v_f32m1(iptr + j, vl);
            intr_sum =  vfadd_vv_f32m1 (intr_sum, vinput, vl);
        }
        vsum = vfredosum_vs_f32m1_f32m1(vsum, intr_sum, vsum, vl);
        sum = vfmv_f_s_f32m1_f32(vsum); 
        while (vl_end > 0){
            int m = inner_size - vl_end;
            sum = sum + iptr[m];
            vl_end--;
        }
        float mean = sum / inner_size;

        // 计算方差和逆标准差
        float square_sum = 0;
        vl_end = (inner_size)%(vl);
        vfloat32m1_t vsquare_sum = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t intr_square_sum = vfmv_v_f_f32m1(0.0f, vl); 
        for (int j = 0; j < inner_size-vl_end; j += vl) {
            vfloat32m1_t v = vle32_v_f32m1(iptr + j, vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vdiff_sq = vfmul_vv_f32m1(vdiff, vdiff, vl);
            intr_square_sum = vfadd_vv_f32m1(intr_square_sum, vdiff_sq, vl);
        }
        vsquare_sum = vfredosum_vs_f32m1_f32m1(vsquare_sum, intr_square_sum, vsquare_sum, vl);
        square_sum = vfmv_f_s_f32m1_f32(vsquare_sum); 
        while (vl_end > 0){
            int m = inner_size - vl_end;
            square_sum = square_sum + (iptr[m] - mean) * (iptr[m] - mean);
            vl_end--;
        }
        float variance = square_sum / inner_size;
        float stddev = sqrtf(variance + epsilon);
        float inv_stddev = 1.0f / stddev;

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


int LayerNormalization_asm(int ndim, const int *dims, const float *X_ptr,
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
        size_t vl = vsetvlmax_e32m1();
        size_t vl_end = (inner_size)%(vl);
        asm volatile(
            "vmv.v.x                v8, %[zero]\n\t"
            "vmv.v.x                v24, %[zero]\n\t"
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m1\n\t"
            "vle.v                 v16, (%[da])\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "vfadd.vv              v8, v8, v16\n\t"
            "bnez                     %[len], 1b\n\t"
            "vfredosum.vs  v24, v8, v24\n\t"
            "vfmv.f.s                %[s], v24\n\t"

            : [s] "=f" (sum)
            : [da] "r" (&iptr[0]),
            [len]"r"(inner_size - vl_end),
            [zero] "r"(0.0f)
            : "v8", "v16", "v24", "t0"
        );
        while (vl_end > 0){
            int m =  inner_size - vl_end;
            sum = sum + iptr[m];
            vl_end--;
        }
        float mean = sum / inner_size;

        // 计算方差和逆标准差
        float square_sum = 0;
        vl_end = (inner_size)%(vl);
        asm volatile(
            "vmv.v.x                v8, %[zero]\n\t" //v8=intr_square_sum
            "vmv.v.x                v24, %[zero]\n\t" //v24=vsquare_sum
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m1\n\t"
            "vle.v                 v16, (%[da])\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "vfsub.vf               v16, v16, %[m]\n\t"
            "vfmul.vv             v16, v16, v16\n\t"
            "vfadd.vv              v8, v8, v16\n\t"
            "bnez                     %[len], 1b\n\t"
            "vfredosum.vs  v24, v8, v24\n\t"
            "vfmv.f.s                %[s], v24\n\t"
            : [s] "=f" (square_sum)
            : [da] "r" (&iptr[0]),
            [len]"r"(inner_size - vl_end),
            [zero] "r"(0),
            [m] "f" (mean)
            : "v8", "v16", "v24", "t0"
        );
        while (vl_end > 0){
            int m = inner_size - vl_end;
            square_sum = square_sum + (iptr[m] - mean) * (iptr[m] - mean);
            vl_end--;
        }
        float variance = square_sum / inner_size;
        float stddev = sqrtf(variance + epsilon);
        float inv_stddev = 1.0f / stddev;

        if (mean_ptr != NULL && inv_stddev_ptr != NULL) {
            mean_ptr[i] = mean;
            inv_stddev_ptr[i] = inv_stddev;
        }

        // 归一化并应用缩放和偏移
/*         float zero=0.0f;
        asm volatile(
            "1: \n\t"
            "vsetvli                  t0, %3, e32, m1\n\t"
            "vle32.v                 v8, (%0)\n\t"   // iptr
            "vle32.v                v16, (%1)\n\t" //scale_ptr
            "vle32.v                v24, (%2)\n\t" //B_ptr
            "vfmv.v.f               v1, %7\n\t"
            "sub                       %3, %3, t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %0, %0, t0\n\t"
            "add                       %1, %1, t0\n\t"
            "add                       %2, %2, t0\n\t"
            "vfsub.vf               v8, v8, %4\n\t"
            "vfmul.vf              v8, v8, %5\n\t"
            "vfmul.vv             v8, v8, v16\n\t"
            //"vmford.vv           v0, v8, v24\n\t"
            //"vmerge.vvm      v24, v1, v24, v0\n\t"
            "vfadd.vv              v8, v8, v24\n\t"
            "vse32.v                v8, (%6)\n\t"
            "add                       %6, %6, t0\n\t"
            "bnez                     %3, 1b\n\t"
             : "=r" (iptr),
            "=r" (scale_ptr),
            "=r" (B_ptr),
            "=r"(inner_size),
            "=f" (mean),
            "=f" (inv_stddev),
            "=r" (optr),
            "=f" (zero)
            :"0" (iptr),
            "1" (scale_ptr),
            "2" (B_ptr),
            "3"(inner_size),
            "4" (mean),
            "5" (inv_stddev),
            "6" (optr),
            "7" (zero) 
            : "v0", "v1", "v8", "v16", "v24", "t0"
        );  */

        float zero=0.0f;
        asm volatile(
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m1\n\t"
            "vle.v                 v8, (%[da])\n\t"   // iptr
            "vle.v                v16, (%[s])\n\t" //scale_ptr
            "vle.v                v24, (%[b])\n\t" //B_ptr
            "vfmv.v.f               v1, %[z]\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "add                       %[s], %[s], t0\n\t"
            "add                       %[b], %[b], t0\n\t"
            "vfsub.vf               v8, v8, %[m]\n\t"
            "vfmul.vf              v8, v8, %[inv]\n\t"
            "vfmul.vv             v8, v8, v16\n\t"
            "vmford.vv           v0, v8, v24\n\t"
            "vmerge.vvm      v24, v1, v24, v0\n\t"
            "vfadd.vv              v8, v8, v24\n\t"
            "vse.v                v8, (%[o])\n\t"
            "add                       %[o], %[o], t0\n\t"
            "bnez                     %[len], 1b\n\t"
            : 
            : [da] "r" (&iptr[0]),
            [s] "r" (&scale_ptr[0]),
            [b] "r" (&B_ptr[0]),
            [len]"r"(inner_size),
            [m] "f" (mean),
            [inv] "f" (inv_stddev),
            [z] "f" (zero),
            [o] "r" (&optr[0])
            : "v0", "v1", "v8", "v16", "v24", "t0"
        ); 
    }

    return DNN_SUCCESS;
}

int compare_result(float *result0, float *result1, int size)
{  
    int isEqual = 1; 
    for (int i = 0; i < size; i++)
    {
        if (result0[i] != result1[i])
        {
            isEqual = 0; 
            printf("not equal: result0=%f result1=%f index=%d \n", result0[i], result1[i], i);
        }
    }
    return isEqual;
}


int main() {
    // 初始化变量
    int N = 1, H = 112, W = 112, C = 16;
    int dims[] = {N, H, W, C}; // 例如 batch_size=2, height=3, width=4, channels=5
    int ndim = 4;
    int size = N*H*W*C;

    // 输入
    float X[size]; // 输入数据
    float scale[C]; // 缩放参数，假设axis=-1
    float B[C]; // 偏置参数
    // 输出
    float Y[size], Y_i[size], Y_a[size]; // 输出数据
    float mean[N*H*W], mean_i[N*H*W], mean_a[N*H*W]; // 均值输出
    float inv_stddev[N*H*W], inv_stddev_i[N*H*W], inv_stddev_a[N*H*W]; // 逆标准差输出
    clock_t start, end, start_i, end_i, start_a, end_a;
    double cpu_time, cpu_time_i, cpu_time_a;

    // 填充一些测试数据
    for (int i = 0; i < size; ++i) X[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < C; ++i) scale[i] = 1.0f;
    for (int i = 0; i < C; ++i) B[i] = 0.0f;

    // 设置属性
    float epsilon = 1e-5f;
    int axis = -1;
    int stash_type = 1; //Mean 和 InvStdDev的计算精度，float16/float32

    // 处理axis负值
    if (axis < 0) {
        axis += ndim;
    }

    /*   printf("Input X:\n");
    for (int i = 0; i < size; ++i) {
        printf("%f ", X[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    } */

    // 调用LayerNormalization函数
    start = clock();
    int status = LayerNormalization(ndim, dims, X, scale, B, axis, epsilon, Y, mean, inv_stddev, stash_type);
    end = clock();
    cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    if (status != DNN_SUCCESS) {
        printf("Layer normalization failed\n");
        return -1;
    }
    printf("\nOutput Y:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", Y[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }

    // 调用LayerNormalization函数
    start_i = clock();
    int status_intr = LayerNormalization_intr(ndim, dims, X, scale, B, axis, epsilon, Y_i, mean_i, inv_stddev_i, stash_type);
    end_i = clock();
    cpu_time_i = (double)(end_i - start_i) / CLOCKS_PER_SEC;
    if (status_intr != DNN_SUCCESS) {
        printf("Layer normalization failed\n");
        return -1;
    }
    printf("\nOutput Y_i:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", Y_i[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }
    //int isEqual_intr = compare_result(Y, Y_i, size); 

        // 调用LayerNormalization函数
/*     start_a = clock();
    int status_asm = LayerNormalization_asm(ndim, dims, X, scale, B, axis, epsilon, Y_a, mean_a, inv_stddev_a, stash_type);
    end_a = clock();
    cpu_time_a = (double)(end_a - start_a) / CLOCKS_PER_SEC;
    if (status_asm != DNN_SUCCESS) {
        printf("Layer normalization failed\n");
        return -1;
    }
    printf("\nOutput Y_a:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", Y_a[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    } */
    //int isEqual_a = compare_result(Y, Y_a, size);

/*           //判断结果是否一致
    printf("\n----Results Equal or NotEqual----\n");
    if (isEqual_intr) printf("rvv_intr_m1 and C: Equal.\n");
    else printf("rvv_intr_m1 and C: NotEqual.\n");
    if (isEqual_a) printf("rvv_asm_m1 and C: Equal.\n");
    else printf("rvv_asm_m1 and C: NotEqual.\n"); */

/*     printf("\nOutput mean:\n");
    for (int i = 0; i < N*H*W; ++i) {
        printf("%f ", mean[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }
    printf("\nOutput mean_a:\n");
    for (int i = 0; i < N*H*W; ++i) {
        printf("%f ", mean_a[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }

    printf("\nOutput inv_stddev:\n");
    for (int i = 0; i < N*H*W; ++i) {
        printf("%f ", inv_stddev[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }
    printf("\nOutput inv_stddev_a:\n");
    for (int i = 0; i < N*H*W; ++i) {
        printf("%f ", inv_stddev_a[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    } */

    printf("\n-----------Time Cost-----------\n");
    printf("C time is %f s\n", cpu_time);
    printf("rvv_intr_m1 time is %f s\n", cpu_time_i);
    printf("rvv_asm_m1 time is %f s\n", cpu_time_a);

    printf("\n");

    return 0;
}
