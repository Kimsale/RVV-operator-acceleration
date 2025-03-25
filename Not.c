/***********************************************************************************************************
*  @File:                    Not.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   Not算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    Not算子执行按位非操作
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
int input[N];

/******************************************************************
*  Function:            Not_c
*  Description:       Not算子的C-for循环实现方法，按位非
*  Parameter:         
*  @inputA              输入数据
*  @inputB              输入数据
*  @output              输出数据
*  @size                    输入数据的尺寸
*  Return:                 None
******************************************************************/
void Not_c(int size, int *input, int *output) {
    for (int i = 0; i < size; ++i) {
        output[i] = ~input[i];
    }
}

/******************************************************************
*  Function:            Not_intr
*  Description:       Not算子的rvv-intrisic函数实现方法
*  Parameter:         同Not_c
*  Return:                 None
******************************************************************/
void Not_intr(int size, int *input, int *output_vector) {
    vint32m4_t vinput;
	vint32m4_t voutput;
	unsigned int gvl = 0;
    gvl = vsetvlmax_e32m4();
	for (int i = 0; i < size;) {
		vinput = vle32_v_i32m4(&input[i], gvl);
		voutput = vnot_v_i32m4(vinput,  gvl);
        vse32_v_i32m4(&output_vector[i], voutput, gvl);
		i += gvl;
	}
}

/******************************************************************
*  Function:            Not_assembly
*  Description:       Not算子的rvv-内联汇编实现方法
*  Parameter:         同Not_c
*  Return:                 None
******************************************************************/
void Not_assembly(int size, int *input, int *output_assembly) {
    asm volatile(
        "1:\n\t"
        "vsetvli        t0, %2, e32, m4\n\t"
        "vle32.v       v8, (%0)\n\t"
        "sub              %2, %2, t0\n\t"
        "slli                t0, t0, 2\n\t"
        "add              %0, %0, t0\n\t"
        "vnot.v         v16, v8\n\t"
        "vse32.v       v16, (%1)\n\t"
        "add               %1, %1, t0\n\t"
        "bnez             %2, 1b\n\t"

        :"=r"(input),
        "=r"(output_assembly),
        "=r"(size)
        :"0"(input),
        "1"(output_assembly),
        "2"(size)
        :"t0", "v8", "v16"
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
    printf("输入数组: input[%d]", N);
    for(int k = 0; k < N; k++){
        input[k] = rand()%2+0;
        printf("%d ", input[k]);
    }
    printf("\n");
    printf("\n");

    //--------------------------------------------测试方法1: C-for----------------------------------------------
    int output[N];
    clock_t start_time = clock();
    Not_c(N, input, output);
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Output after applying Not_c:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    //--------------------------------------测试方法2: rvv-intrisic-m4-----------------------------------------
    int output_intr[N]; 
    clock_t start_time_intr = clock();
    Not_intr(N, input, output_intr);
    clock_t end_time_intr = clock();
    double cpu_time_used_intr = ((double) (end_time_intr - start_time_intr)) / CLOCKS_PER_SEC;
    printf("\nOutput after applying Not_intr:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output_intr[i]);
    }
    printf("\n");
    int isEqual_intr = compare_result(output, output_intr, N);

    //--------------------------------------测试方法3: rvv-assembly-m4----------------------------------------
    int output_asm[N]; 
    clock_t start_time_asm = clock();
    Not_assembly(N, input, output_asm);
    clock_t end_time_asm = clock();
    double cpu_time_used_asm = ((double) (end_time_asm - start_time_asm)) / CLOCKS_PER_SEC;
    printf("\nOutput after applying Not_assembly:\n");
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

