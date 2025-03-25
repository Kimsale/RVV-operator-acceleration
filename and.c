#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <riscv_vector.h>
#include <time.h>

void and_c(int *A, int *B, int *output, size_t len) {
    for (size_t i = 0; i < len; i++) {
        output[i]=A[i] & B[i];
    }
}

void and_i(int *A, int *B, int *output, size_t len) {
    vint32m4_t vinputA;
    vint32m4_t vinputB;
	vint32m4_t voutput;
	unsigned int gvl = 0;
    gvl = vsetvlmax_e32m4();
	for (int i = 0; i < len;) {
		vinputA = vle32_v_i32m4(&A[i], gvl);
        vinputB = vle32_v_i32m4(&B[i], gvl);
		voutput = vand_vv_i32m4(vinputA, vinputB,  gvl);
        vse32_v_i32m4(&output[i], voutput, gvl);
		i += gvl;
	}
}

void and_a(int *A, int *B, int *output, size_t len) {
    asm volatile(
        "1:\n\t"
        "vsetvli        t0, %3, e32, m4\n\t"
        "vle.v       v8, (%0)\n\t"
        "vle.v       v16, (%1)\n\t"
        "sub              %3, %3, t0\n\t"
        "slli                t0, t0, 2\n\t"
        "add              %0, %0, t0\n\t"
        "add              %1, %1, t0\n\t"
        "vand.vv         v24, v8, v16\n\t"
        "vse.v       v24, (%2)\n\t"
        "add               %2, %2, t0\n\t"
        "bnez             %3, 1b\n\t"
        :"=r"(A),
        "=r"(B),
        "=r"(output),
        "=r"(len)
        :"0"(A),
        "1"(B),
        "2"(output),
        "3"(len)
        :"t0", "v8", "v16", "v24"
    );
}

int main() {
    int size = 200000;
    int A[200000];
    int B[200000];
    int output_c[200000], output_i[200000], output_a[200000];
    clock_t start_c, end_c, start_i, end_i, start_a, end_a;
    double cpu_time_c, cpu_time_i, cpu_time_a;

    for (int i = 0; i < size; i++) {
        A[i] = rand() % 2;
        B[i] = rand() % 2;
    }
    printf("\n");

    start_c = clock();
	and_c(A, B, output_c, size);
	end_c = clock();
	cpu_time_c = (double)(end_c-start_c)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time_c);

    start_i = clock();
	and_i(A, B, output_i, size);
	end_i = clock();
	cpu_time_i = (double)(end_i-start_i)/CLOCKS_PER_SEC;
	printf("intrinsic execution time %f seconds\n",cpu_time_i);

    start_a = clock();
	and_a(A, B, output_a, size);
	end_a = clock();
	cpu_time_a = (double)(end_a-start_a)/CLOCKS_PER_SEC;
	printf("assembly execution time %f seconds\n",cpu_time_a);

    for (int i = 0; i < 10; i++) {
        printf("A = %d\t", A[i]);
        printf("B = %d\t", B[i]);
        printf("c = %d", output_c[i]);
        printf("\tintr = %d", output_i[i]);
        printf("\tasm = %d\n", output_a[i]);
    }

    printf("\n");
    return 0;
}