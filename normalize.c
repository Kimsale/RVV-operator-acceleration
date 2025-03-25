#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>
#include <time.h>

void normalize_c(float *iptr, float *optr, size_t inner_size, float mean, float *B_ptr,float *scale_ptr, float inv_stddev) {
    for (int j = 0; j < inner_size; ++j) {
        float bias = (B_ptr == NULL) ? 0 : B_ptr[j];
        optr[j] = (iptr[j] - mean) * inv_stddev * scale_ptr[j] + bias;
    }
}

void normalize_i(float *iptr, float *optr, size_t inner_size, float mean, float *B_ptr,float *scale_ptr, float inv_stddev) {
    size_t vl = vsetvl_e32m1(inner_size);
    for (int j = 0; j < inner_size; j += vl) {
        //vl = vsetvl_e32m1(inner_size - j);
        vfloat32m1_t v = vle32_v_f32m1(&iptr[j], vl);
        vfloat32m1_t vscale = vle32_v_f32m1(&scale_ptr[j], vl);
        vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
        vfloat32m1_t vnorm = vfmul_vf_f32m1(vdiff, inv_stddev, vl);
        vnorm = vfmul_vv_f32m1(vnorm, vscale, vl);
        if (B_ptr != NULL) {
            vfloat32m1_t vbias = vle32_v_f32m1(B_ptr + j, vl);
            vnorm = vfadd_vv_f32m1(vnorm, vbias, vl);
        }
        vse32_v_f32m1(&optr[j], vnorm, vl);
    }
}

void normalize_a(float *iptr, float *optr, size_t inner_size, float mean, float *B_ptr,float *scale_ptr, float inv_stddev){
    float zero=0.0f;
    size_t vl = vsetvl_e32m1(inner_size);
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
    );
}

/* void normalize_a(float *iptr, float *optr, size_t inner_size, float mean, float *B_ptr,float *scale_ptr, float inv_stddev){
    float square_sum = 0;
    size_t vl = vsetvl_e32m1(inner_size);
    asm volatile(
        "1: \n\t"
        "vsetvli                  t0, %[len], e32, m1\n\t"
        "vle32.v                 v8, (%[da])\n\t"
        "vle32.v                v16, (%[s])\n\t"
        "vle32.v                v24, (%[b])\n\t"
        "sub                       %[len], %[len], t0\n\t"
        "slli                         t0, t0, 2\n\t"
        "add                       %[da], %[da], t0\n\t"
        "add                       %[s], %[s], t0\n\t"
        "add                       %[b], %[b], t0\n\t"
        "vfsub.vf               v8, v8, %[m]\n\t"
        "vfmul.vf              v8, v8, %[inv]\n\t"
        "vfmul.vv             v8, v8, v16\n\t"
        "vfadd.vv              v8, v8, v24\n\t"
        "vse32.v                v8, (%[o])\n\t"
        "bnez                     %[len], 1b\n\t"
        : 
        : [da] "r" (&iptr[0]),
        [s] "r" (&scale_ptr[0]),
        [b] "r" (&B_ptr[0]),
        [len]"r"(inner_size),
        [m] "f" (mean),
        [inv] "f" (inv_stddev),
        [o] "f" (&optr[0])
        : "v8", "v16", "v24", "t0"
    );
} */

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
    int size = 21;
    float input[size];

    float scale[size]; // 缩放参数，假设axis=-1
    float B[size]; // 偏置参数
    float inv_stddev = 5.0f;

    float epsilon = 1e-5f;
    float mean = 0.1f;

    float output_c[size], output_i[size], output_a[size];
    clock_t start_c, end_c, start_i, end_i, start_a, end_a;
    double cpu_time_c, cpu_time_i, cpu_time_a;

    for (int i = 0; i < size; i++) {
        input[i] = rand()%5;
        scale[i] = 1.0f;
        B[i] = 0.0f;
        //printf("%f\n",input[i]);
    }
    printf("\n");

    start_c = clock();
	normalize_c(input, output_c, size, mean, B, scale, inv_stddev);
	end_c = clock();
	cpu_time_c = (double)(end_c-start_c)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time_c);
    for (int i = 0; i < size; ++i) {
        printf("%f ", output_c[i]);
    }
    printf("\n");

    start_i = clock();
	normalize_i(input, output_i, size, mean, B, scale, inv_stddev);
	end_i = clock();
	cpu_time_i = (double)(end_i-start_i)/CLOCKS_PER_SEC;
	printf("\nintrinsic execution time %f seconds\n",cpu_time_i);
/*     for (int i = 0; i < size; ++i) {
        printf("%f ", output_i[i]);
    }
    printf("\n"); */
    int isEqual_intr = compare_result(output_c, output_i, size);

    start_a = clock();
	normalize_a(input, output_a, size, mean, B, scale, inv_stddev);
	end_a = clock();
	cpu_time_a = (double)(end_a-start_a)/CLOCKS_PER_SEC;
	printf("\nassembly execution time %f seconds\n",cpu_time_a);
    for (int i = 0; i < size; ++i) {
        printf("%f ", output_a[i]);
    }
    printf("\n");
    int isEqual_a = compare_result(output_c, output_a, size);

        //判断结果是否一致
    printf("\n----Results Equal or NotEqual----\n");
    if (isEqual_intr) printf("rvv_intr_m1 and C: Equal.\n");
    else printf("rvv_intr_m1 and C: NotEqual.\n");
    if (isEqual_a) printf("rvv_asm_m1 and C: Equal.\n");
    else printf("rvv_asm_m1 and C: NotEqual.\n");
    
    printf("\n");
    return 0;
}