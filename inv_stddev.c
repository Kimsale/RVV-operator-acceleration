#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>
#include <time.h>

void inv_stddev_c(float *input, float *output, size_t inner_size, float epsilon, float mean) {
    float square_sum = 0;
    for (int j = 0; j < inner_size; ++j) {
        square_sum += (input[j] - mean) * (input[j] - mean);
    }
    float variance = square_sum / inner_size;
    float stddev = sqrtf(variance + epsilon);
    *output = 1.0f / stddev;
}

void inv_stddev_i(float *input, float *output, size_t inner_size, float epsilon, float mean) {
        float square_sum = 0;
        size_t vl = vsetvl_e32m1(inner_size);
        size_t vl_end = (inner_size)%(vl);
        vfloat32m1_t vsquare_sum = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t intr_square_sum = vfmv_v_f_f32m1(0.0f, vl); 
        for (int j = 0; j < inner_size-vl_end; j += vl) {
            vfloat32m1_t v = vle32_v_f32m1(&input[j], vl);
            vfloat32m1_t vdiff = vfsub_vf_f32m1(v, mean, vl);
            vfloat32m1_t vdiff_sq = vfmul_vv_f32m1(vdiff, vdiff, vl);
            intr_square_sum = vfadd_vv_f32m1(intr_square_sum, vdiff_sq, vl);
        }
        vsquare_sum = vfredosum_vs_f32m1_f32m1(vsquare_sum, intr_square_sum, vsquare_sum, vl);
        square_sum = vfmv_f_s_f32m1_f32(vsquare_sum); 
        while (vl_end > 0){
            int m = inner_size - vl_end;
            square_sum = square_sum + (input[m] - mean) * (input[m] - mean);
            vl_end--;
        }
        float variance = square_sum / inner_size;
        float stddev = sqrtf(variance + epsilon);
        *output = 1.0f / stddev;
}

void inv_stddev_a(float *input, float *output, size_t inner_size, float epsilon, float mean){
    float square_sum = 0;
    size_t vl = vsetvl_e32m1(inner_size);
    size_t vl_end = (inner_size)%(vl);
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
        : [da] "r" (&input[0]),
        [len]"r"(inner_size - vl_end),
        [zero] "r"(0),
        [m] "f" (mean)
        : "v8", "v16", "v24", "t0"
    );
    while (vl_end > 0){
        int m = inner_size - vl_end;
        square_sum = square_sum + (input[m] - mean) * (input[m] - mean);
        vl_end--;
    }
    float variance = square_sum / inner_size;
    float stddev = sqrtf(variance + epsilon);
    *output = 1.0f / stddev;
}


int main() {
    int size = 200000;
    float input[200000];

    float epsilon = 1e-5f;
    float mean = 0.1;

    float output_c, output_i, output_a;
    clock_t start_c, end_c, start_i, end_i, start_a, end_a;
    double cpu_time_c, cpu_time_i, cpu_time_a;

    for (int i = 0; i < size; i++) {
        input[i] = rand()%5;
        //printf("%f\n",input[i]);
    }
    printf("\n");

    start_c = clock();
	inv_stddev_c(input, &output_c, size, epsilon, mean);
	end_c = clock();
	cpu_time_c = (double)(end_c-start_c)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time_c);

    start_i = clock();
	inv_stddev_i(input, &output_i, size, epsilon, mean);
	end_i = clock();
	cpu_time_i = (double)(end_i-start_i)/CLOCKS_PER_SEC;
	printf("intrinsic execution time %f seconds\n",cpu_time_i);

    start_a = clock();
	inv_stddev_a(input, &output_a, size, epsilon, mean);
	end_a = clock();
	cpu_time_a = (double)(end_a-start_a)/CLOCKS_PER_SEC;
	printf("assembly execution time %f seconds\n",cpu_time_a);

    printf("c = %f", output_c);
    printf("\tintr = %f", output_i);
    printf("\tasm = %f\n", output_a);

    printf("\n");
    return 0;
}