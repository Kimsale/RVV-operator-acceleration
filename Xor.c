/***********************************************************************************************************
*  @File:                    Xor.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   Xor算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    Xor算子执行异或操作
                                    输入输出数据类型为int32，维度为1；
                                    两种rvv寄存器分组模式都为m4
************************************************************************************************************/


#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#define N 255            //N为输入数据尺寸，可修改
int inputA[N];
int inputB[N];

/******************************************************************
*  Function:            Xor_c
*  Description:       Xor算子的C-for循环实现方法
*  Parameter:         
*  @inputA              输入数据
*  @inputB              输入数据
*  @output              输出数据
*  @size                    输入数据的尺寸
*  Return:                 None
******************************************************************/
void Xor(int size, int *inputA, int *inputB, int *output) {
    for (int i = 0; i < size; ++i) {
        output[i] = inputA[i] ^ inputB[i] ;
    }
}

/******************************************************************
*  Function:            Xor_intr
*  Description:       Xor算子的rvv-intrisic函数实现方法
*  Parameter:         同Xor_c
*  Return:                 None
******************************************************************/
void Xor_intr(int size, int *inputA, int *inputB, int *outputTensor_vector) {
    vint32m1_t vinputA;
    vint32m1_t vinputB;
	vint32m1_t voutput;
	unsigned int gvl = 0;
    gvl = vsetvlmax_e32m1();
	for (int i = 0; i < size;) {
		vinputA = vle32_v_i32m1(&inputA[i], gvl);
        vinputB = vle32_v_i32m1(&inputB[i], gvl);
		voutput = vxor_vv_i32m1(vinputA, vinputB,  gvl);
        vse32_v_i32m1(&outputTensor_vector[i], voutput, gvl);
		i += gvl;
	}
}

/******************************************************************
*  Function:            Xor_assembly
*  Description:       Xor算子的rvv-内联汇编实现方法
*  Parameter:         同Xor_c
*  Return:                 None
******************************************************************/
void Xor_assembly(int size, int *inputA, int *inputB,
 int *outputTensor_assembly) {
    asm volatile(
        "1:\n\t"
        "vsetvli        t0, %3, e32, m1\n\t"
        "vle32.v       v8, (%0)\n\t"
        "vle32.v       v16, (%1)\n\t"
        "sub              %3, %3, t0\n\t"
        "slli                t0, t0, 2\n\t"
        "add              %0, %0, t0\n\t"
        "add              %1, %1, t0\n\t"
        "vxor.vv         v24, v8, v16\n\t"
        "vse32.v       v24, (%2)\n\t"
        "add               %2, %2, t0\n\t"
        "bnez             %3, 1b\n\t"

        :"=r"(inputA),
        "=r"(inputB),
        "=r"(outputTensor_assembly),
        "=r"(size)
        :"0"(inputA),
        "1"(inputB),
        "2"(outputTensor_assembly),
        "3"(size)
        :"t0", "v8", "v16", "v24"
    );
}

/*******************************************************************
*  Function:             compare_result
*  Description:       用于判断rvv实现与C-for循环实现的输出结果是否相同
*  Parameter:         
*  @result0             用C-for循环实现的输出数据
*  @result1             用某种rvv形式实现的输出数据
*  @size                    输出数据的尺寸
*  Return:                 1--两种实现方法输出结果相同，0--输出结果不同
*******************************************************************/
int compare_result(int *result0, int *result1, int size)
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
    // 输入数据，随机值，0/1
    printf("输入数组: inputA[%d]", N);
    for(int k = 0; k < N; k++){
        inputA[k] = rand()%(2-0)+0;
        printf("%d ", inputA[k]);
    }
    printf("\n");
    printf("\n");
    printf("输入数组: inputB[%d]", N);
    for(int k = 0; k < N; k++){
        inputB[k] = rand()%2+0;
        printf("%d ", inputB[k]);
    }
    printf("\n");
    printf("\n");

    //--------------------------------------------测试方法1: C-for----------------------------------------------
    int output[N];
    clock_t start_time = clock();
    Xor(N, inputA, inputB, output);
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Output after applying Xor_c:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    //--------------------------------------测试方法2: rvv-intrisic-m4-----------------------------------------
    int output_intr[N]; 
    clock_t start_time_intr = clock();
    Xor_intr(N, inputA, inputB,  output_intr);
    clock_t end_time_intr = clock();
    double cpu_time_used_intr = ((double) (end_time_intr - start_time_intr)) / CLOCKS_PER_SEC;
    // 输出结果
    printf("Output after applying Xor_intr:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output_intr[i]);
    }
    printf("\n");
    int isEqual_intr = compare_result(output, output_intr, N);

    //--------------------------------------测试方法3: rvv-assembly-m4----------------------------------------
    int output_asm[N]; 
    clock_t start_time_asm = clock();
    Xor_assembly(N, inputA, inputB,  output_asm);
    clock_t end_time_asm = clock();
    double cpu_time_used_asm = ((double) (end_time_asm - start_time_asm)) / CLOCKS_PER_SEC;
    // 输出结果
    printf("Output after applying Xor_assembly:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output_asm[i]);
    }
    printf("\n");
    int isEqual_asm  = compare_result(output, output_asm, N);

    //判断结果是否一致
    printf("\n----Results Equal or NotEqual----\n");
    if (isEqual_intr) printf("rvv_intr_m4 and C: Equal.\n");
    else printf("rvv_intr_m4 and C: NotEqual.\n");
    if (isEqual_asm) printf("rvv_asm_m4 and C: Equal.\n");
    else printf("rvv_asm_m4 and C: NotEqual.\n");

    printf("\n-----------Time Cost-----------\n");
    printf("C time is %f s\n", cpu_time_used);
    printf("rvv_intr_m4 time is %f s\n", cpu_time_used_intr);
    printf("rvv_asm_m4 time is %f s\n", cpu_time_used_asm);

    return 0;
}

