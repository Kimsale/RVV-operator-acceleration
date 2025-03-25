#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <time.h>

void mean_c(float *input, float *output, size_t len) {
    float sum =0.0;
    for (size_t i = 0; i < len; i ++) {
        sum =sum + input[i];
    }
    *output = sum/len;
}

void mean_i(float *input, float *output, size_t len) {
    float sum = 0.0;
    size_t vl = vsetvlmax_e32m4(); 
    size_t vl_end = (len)%(vl);
    vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl);
    for (size_t i = 0; i < len - vl_end; i += vl) {
        vfloat32m4_t vinput = vle32_v_f32m4(&input[i], vl);
        intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
    }
    b = vfredosum_vs_f32m4_f32m1 (b, intr_sum, b, vl); 
    sum = vfmv_f_s_f32m1_f32(b); 
    while (vl_end > 0){
        int m = len - vl_end;
        sum = sum + input[m];
        vl_end--;
    }
    *output =  sum/len;
}

void mean_a(float *input, float *output, size_t len){
    float sum = 0;
    size_t vl = vsetvlmax_e32m4();
    size_t vl_end = (len)%(vl);
    asm volatile(
        "vmv.v.x                v8, %[zero]\n\t"
        "vmv.v.x                v24, %[zero]\n\t"
        "1: \n\t"
        "vsetvli                  t0, %[len], e32, m4\n\t"
        "vle32.v                 v16, (%[da])\n\t"
        "sub                       %[len], %[len], t0\n\t"
        "slli                         t0, t0, 2\n\t"
        "add                       %[da], %[da], t0\n\t"
        "vfadd.vv              v8, v8, v16\n\t"
        "bnez                     %[len], 1b\n\t"
        "vfredosum.vs  v24, v8, v24\n\t"
        "vfmv.f.s                %[s], v24\n\t"

        : [s] "=f" (sum)
        : [da] "r" (&input[0]),
        [len]"r"(len - vl_end),
        [zero] "r"(0.0f)
        : "v8", "v16", "v24", "t0"
    );
    while (vl_end > 0){
        int m =  len - vl_end;
        sum = sum + input[m];
        vl_end--;
    }
    *output =  sum/len;
}


int main() {
    int size = 200000;
    float input[200000];
    float output_c, output_i, output_a;
    clock_t start_c, end_c, start_i, end_i, start_a, end_a;
    double cpu_time_c, cpu_time_i, cpu_time_a;

    for (int i = 0; i < size; i++) {
        input[i] = rand()%5;
        //printf("%f\n",input[i]);
    }
    printf("\n");

    start_c = clock();
	mean_c(input, &output_c, size);
	end_c = clock();
	cpu_time_c = (double)(end_c-start_c)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time_c);

    start_i = clock();
	mean_i(input, &output_i, size);
	end_i = clock();
	cpu_time_i = (double)(end_i-start_i)/CLOCKS_PER_SEC;
	printf("intrinsic execution time %f seconds\n",cpu_time_i);

    start_a = clock();
	mean_a(input, &output_a, size);
	end_a = clock();
	cpu_time_a = (double)(end_a-start_a)/CLOCKS_PER_SEC;
	printf("assembly execution time %f seconds\n",cpu_time_a);

    printf("c = %f", output_c);
    printf("\tintr = %f", output_i);
    printf("\tasm = %f\n", output_a);

    printf("\n");
    return 0;
}