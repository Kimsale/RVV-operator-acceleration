/*********************************************************************************************************************************
*  @File:                    Range.c
*  @Author:             JSE
*  @Date:                 2024-5-23
*  @Description:   Range算子的性能测试代码，包含普通C-for循环算子实现方法； 
                                    Range算子用于生成一个具有等差序列的张量;
                                    输入输出数据类型为float32
*********************************************************************************************************************************/

#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/*******************************************************************
*  Function:            Range_c
*  Description:       Range算子的C-for循环实现方法
*  Parameter:         
*  @start                  序列的起始值
*  @delta                 序列的步长
*  @output             输出数据
*  @n                         序列的个数
*  Return:                 None
*******************************************************************/
void Range_c(float start, float delta, float* output, size_t n) {
    for (int64_t i = 0; i < n; ++i) {
        output[i] = start;
        start += delta;
    }
}

int main() {
    float start = 1.0;             //序列的起始值
    float limit = 10000.0;   //序列的终止值
    float delta = 2.0;            //序列的步长
    
    int64_t N = (int64_t)ceil((1.0 * (limit - start)) / delta);
    if (N <= 0) N = 0;

    //--------------------------------测试方法1: C-for---------------------------------
    float output[N];
    clock_t start_time = clock();
    Range_c(start, delta, output, N);
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", cpu_time_used);
    printf("Output after applying Range_c:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}

