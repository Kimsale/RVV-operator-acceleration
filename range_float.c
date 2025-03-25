#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// Function for computing the Range operation
void ComputeRange(float start, float limit, float delta, float* output, size_t n) {
    for (int64_t i = 0; i < n; ++i) {
        output[i] = start;
        start += delta;
    }
}

 void ComputeRange_vector(float start, float limit, float delta, float* output, size_t len) {
    size_t vl;
    for (size_t i = 0; i < len;) {
        vl = vsetvl_e32m1(len - i); // Set vector length
        // Create RISC-V vectors
        vfloat32m1_t start_vec = vfmv_v_f_f32m1(start, vl);
        
        // Store the result
        vse32_v_f32m1(&output[i], start_vec, vl);

        // Update the start variable for the next iteration
        start += delta*vl;

        //start_vec = vfadd_vv_f32m1(start_vec, delta_vec, vl);
        i += vl;
    }
}


 
void ComputeRange_assembly(float start, float limit, float delta, float* output, size_t n) {
    asm volatile(
        "1:\n\t"
        "vsetvli    t0, %3, e32, m2\n\t"          // Set vector length register (vl) based on input (%4) size
        "vfmv.v.f    v8, %1\n\t"                           // Move scalar start value to vector register v8
        "sub        %3, %3, t0\n\t"                     // Decrement %4 (loop counter) by t0 (vector length)
        "slli       t0, t0, 2\n\t"                              // Left shift t0 (vector length) by 2 (equivalent to multiply by 4)
        "add        %1, %1, t0\n\t"                    // Update start by adding 4 times the vector length
        "vfadd.vf   v16, v8, %2\n\t"             // Vector float add: v16 = v8 + delta
        "vse.v      v16, (%0)\n\t"                     // Store the result vector (v16) to the memory location pointed by %0 (output)
        "add        %0, %0, t0\n\t"                  // Update output pointer by adding 4 times the vector length
        "bnez       %3, 1b\n\t"                         // Branch to label 1 if %4 is not zero, creating a loop

        : "=r"(output),  // %0
          "=r"(start),   // %1
          "=r"(delta),   // %2
          "=r"(n)              // %3
        : "0"(output), "1"(start), "2"(delta), "3"(n)
        : "v8", "v16", "t0" // v8: start; v16: output
    );
}

int main() {
    float start = 1.0;
    float limit = 50.0;
    float delta = 2.0;
    
/*     if (delta == 0) {
        fprintf(stderr, "Delta in Range operator cannot be zero!\n");
        exit(EXIT_FAILURE);
    } */
    int N = (int)ceil((1.0 * (limit - start)) / delta);

    if (N <= 0) {
        N = 0;
    }
    //--------------------------------------------original-------------------------------------------
    float output[N];
    clock_t start_time = clock();
    ComputeRange(start, limit, delta, output, N);
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    // 打印用时
    printf("Time taken: %f seconds\n", cpu_time_used);
    // 输出结果
    printf("Output after applying Range:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    //-------------------------------------------vector expansion-----------------------------------------
    float output_vector[N]; 
    clock_t start_time2 = clock();
    ComputeRange_vector(start, limit, delta, output_vector, N);
    clock_t end_time2 = clock();
    double cpu_time_used2 = ((double) (end_time2 - start_time2)) / CLOCKS_PER_SEC;
    // 打印用时
    printf("Vector Time taken: %f seconds\n", cpu_time_used2);
    // 输出结果
    printf("Output after applying Range_vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output_vector[i]);
    }
    printf("\n");


    //--------------------------------------------assembly-----------------------------------------------------
    float output_vector2[N]; 
    clock_t start_time3 = clock();
    ComputeRange_assembly(start, limit, delta, output_vector2, N);
    clock_t end_time3 = clock();
    double cpu_time_used3 = ((double) (end_time3 - start_time3)) / CLOCKS_PER_SEC;
     // 打印用时
    printf("Time taken: %f seconds\n", cpu_time_used3);
    // 输出结果
    printf("Output after applying Range_assembly:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output_vector2[i]);
    }
    printf("\n");


    return 0;
}

