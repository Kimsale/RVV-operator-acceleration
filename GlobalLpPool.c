/************************************************************************************************************************
*  @File:                    GlobalLpPool.c
*  @Author:             JSE
*  @Date:                 2024-5-22
*  @Description:   GlobalLpPool算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    GlobalLpPool算子对输入张量进行 Lp 范数池化;
                                    输入数据类型为float32，维度为4；
                                    输出数据类型为float32，维度为2；
                                    两种rvv寄存器分组模式都为m4
************************************************************************************************************************/


#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2     // 第一维大小
#define C 3     // 第二维大小
#define H 2   // 第三维大小
#define W 2   // 第四维大小

float data[N*C*H*W];

/*******************************************************************
*  Function:            GlobalLpPool_c
*  Description:       GlobalLpPool算子的C-for循环实现方法
*  Parameter:         
*  @data                   输入数据
*  @output              输出数据，输出N*C个
*  @p                          Lp 范数池化参数
*  Return:                 None
*******************************************************************/
void GlobalLpPool_c(float data[N * C * H * W], float* result, int p) {
    for (int n = 0; n < N*C; n++) {
        float sum = 0.0f;
        for (int i = 0; i < H*W; i++) {
            sum += powf(data[n*H*W+i], p);
        }
        result[n] = powf(sum, 1.0f/p);
    }
}

/******************************************************************
*  Function:            GlobalLpPool_intr
*  Description:       GlobalLpPool算子的rvv-intrisic函数实现方法
*  Parameter:         同GlobalLpPool_c
*  Return:                 None
******************************************************************/
void GlobalLpPool_intr(float data[N * C * H * W], float* result, int p) {
    size_t vl = vsetvlmax_e32m4(); 
    if (p==1) {
        for (size_t c = 0; c < N*C; c ++) {
            float sum=0.0f;
            size_t vl_end = H*W%(vl);
            vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl); 
            vfloat32m4_t vec_sum = vfmv_v_f_f32m4(0.0f, vl); 
            for (size_t i = 0; i < H*W-vl_end; i += vl) {
                vfloat32m4_t vec_data = vle32_v_f32m4(&data[c*H*W+i], vl); 
                vec_sum = vfadd_vv_f32m4 (vec_sum, vec_data, vl);
            }
            b = vfredosum_vs_f32m4_f32m1 (b, vec_sum, b, vl);
            sum = vfmv_f_s_f32m1_f32(b);
            while (vl_end > 0){
                int m = c*H*W + H*W - vl_end;
                sum = sum + data[m];
                vl_end--;
            }
            result[c]=sum;
        }
    }
    if (p==2){
        for (size_t c = 0; c < N*C; c ++) {
            float sum=0.0f;
            size_t vl_end = H*W%(vl);
            vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl); 
            vfloat32m4_t vec_sum = vfmv_v_f_f32m4(0.0f, vl); 
            for (size_t i = 0; i < H*W-vl_end; i += vl) {
                vfloat32m4_t vec_data = vle32_v_f32m4(&data[c*H*W+i], vl); 
                vfloat32m4_t powf_data = vfmul_vv_f32m4 (vec_data, vec_data, vl);
                vec_sum = vfadd_vv_f32m4 (vec_sum, powf_data, vl);
            }
            b = vfredosum_vs_f32m4_f32m1 (b, vec_sum, b, vl);
            sum = vfmv_f_s_f32m1_f32(b);
            while (vl_end > 0){
                int m = c*H*W+ H*W - vl_end;
                sum = sum + data[m] * data[m];
                vl_end--;
            }
            result[c]=powf(sum, 1.0f/p);
        }
    }	  
}

/***********************************************************************
*  Function:            GlobalLpPool_asm
*  Description:       GlobalLpPool算子的rvv-内联汇编实现方法
                                    加-O3编译会打乱原有顺序，导致结果有误
*  Parameter:         同GlobalLpPool_c
*  Return:                 None
***********************************************************************/
void GlobalLpPool_asm(float data[N * C * H * W], float* result, int p) {
    size_t vl = vsetvlmax_e32m4();
    if (p==1){
        for (int c = 0; c < N*C; c++) {
            size_t vl_end = H*W%(vl);
            float sum = 0.0f;
            asm volatile(
                "vmv.v.x                v8, %[zero]\n\t"
                "vmv.v.x                v24, %[zero]\n\t"
                "1:\n\t"
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
                : [len]"r"(H*W-vl_end),
                    [da] "r" (&data[c * H * W]),
                    [zero] "r"(0.0f)
                : "v8", "v16", "v24", "t0"
            );
            while (vl_end > 0){
                int m = c*H*W + H*W - vl_end;
                sum = sum + data[m];
                vl_end--;
            }
            result[c] = sum;
        }
    }
    if (p==2){
        for (int c = 0; c < N * C; c++) {
            size_t vl_end = H*W%(vl);
            float sum = 0.0f;
            asm volatile(
                "vmv.v.x                v8, %[zero]\n\t"
                "vmv.v.x                v24, %[zero]\n\t"
                "1:\n\t"
                "vsetvli                  t0, %[len], e32, m4\n\t"
                "vle32.v                 v16, (%[da])\n\t" 
                "sub                       %[len], %[len], t0\n\t"
                "slli                         t0, t0, 2\n\t"
                "add                       %[da], %[da], t0\n\t"
                "vfmul.vv              v16, v16, v16\n\t"
                "vfadd.vv              v8, v8, v16\n\t"
                "bnez                     %[len], 1b\n\t"
                "vfredosum.vs  v24, v8, v24\n\t"
                "vfmv.f.s                %[s], v24\n\t"
                : [s] "=f" (sum)
                : [len]"r"(H*W-vl_end),
                    [da] "r" (&data[c*H * W]),
                    [zero] "r"(0.0f)
                : "v8", "v16", "v24", "t0"
            );
            while (vl_end > 0){
                int m = c*H*W + H*W - vl_end;
                sum = sum + data[m] * data[m];
                vl_end--;
            }
            result[c] =powf(sum, 1.0f/p);
        }
    }
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
    int p = 1;  //Lp 范数池化参数，1/2
    float output_c[N*C], output_intr[N*C], output_asm[N*C];

    //初始化 输入数据，随机值，范围[-2, 2]，可修改
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H; k++) {
                for (int l = 0; l < W; l++) {
                data[i * C * H * W + j * H * W + k * W + l] = rand()%5 - 2;
                //printf(" %f",data[i * C * H * W + j * H * W + k * W + l]);           
            }
            //printf("\n");
        }
        //printf("\n");
    }
    printf("\n");

    //-------------------------------------测试方法1: C-for---------------------------------------
    clock_t start_time = clock();
    GlobalLpPool_c(data, output_c, p);
    clock_t end_time = clock();
    double cpu_time = (double) (end_time - start_time)/CLOCKS_PER_SEC;
    printf("Output after applying GlobalLpPool_c(p=%d):\n", p);
    for (int i = 0; i < N*C; i++) {
        printf("%f ", output_c[i]);
    }
    printf("\n");
    
    //----------------------------测试方法2: rvv-intrisic-m4-----------------------------
    clock_t start_intr= clock();
    GlobalLpPool_intr(data, output_intr, p);
    clock_t end_intr = clock();
    double cpu_intr = (double) (end_intr - start_intr)/CLOCKS_PER_SEC;
    printf("\nOutput after applying GlobalLpPool_intr(p=%d):\n", p);
    for (int i = 0; i < N*C; i++) {
        printf("%f ", output_intr[i]);
    }
    printf("\n");
    int isEqual_intr = compare_result(output_c, output_intr, N*C);
    
    //----------------------------测试方法3: rvv-assembly-m4---------------------------
    clock_t start_asm = clock();
    GlobalLpPool_asm(data, output_asm, p);
    clock_t end_asm = clock();
    double cpu_asm = (double) (end_asm - start_asm)/CLOCKS_PER_SEC;
    printf("\nOutput after applying GlobalLpPool_vector-a(p=%d):\n", p);
    for (int i = 0; i < N*C; i++) {
        printf("%f ", output_asm[i]);
    }
    printf("\n");
    int isEqual_asm  = compare_result(output_c, output_asm, N*C);

    //判断结果是否一致
    printf("\n----Results Equal or NotEqual----\n");
    if (isEqual_intr) printf("rvv_intr_m4 and C: Equal.\n");
    else printf("rvv_intr_m4 and C: NotEqual.\n");
    if (isEqual_asm) printf("rvv_asm_m4 and C: Equal.\n");
    else printf("rvv_asm_m4 and C: NotEqual.\n");

    printf("\n-----------Time Cost-----------\n");
    printf("C time is %f s\n", cpu_time);
    printf("rvv_intr_m4 time is %f s\n", cpu_intr);
    printf("rvv_asm_m4 time is %f s\n", cpu_asm);
    
    return 0;
    }
}
