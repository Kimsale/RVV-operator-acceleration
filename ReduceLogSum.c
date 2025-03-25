/*********************************************************************************************************************************
*  @File:                    ReduceLogSum.c
*  @Author:             JSE
*  @Date:                 2024-5-22
*  @Description:   ReduceLogSum算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    ReduceLogSum算子对输入张量 X 沿指定轴求累加和的对数
                                    输入数据类型为float32，维度为C*H*W；
                                    输出数据类型为float32，axes=0时维度为H*W，axes=1时维度为C*W，axes=2时维度为C*H，axes=None时维度为1；
                                    两种rvv寄存器分组模式都为m4
*********************************************************************************************************************************/


#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#define C 16
#define H 112
#define W 112

/* #define C 5
#define H 5
#define W 6 */

float data[C*H*W];

/******************************************************************
*  Function:               chw_to_hwc
*  Description:         将CHW数据重排为HWC形式
*  Parameter:         
*  @input_data       输入数据
*  @output_data    输出重排后的数据
*  Return:                   None
******************************************************************/
void chw_to_hwc(float *input_data, float *output_data) {
    int h, w, c;
    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (c = 0; c < C; ++c) {
                int input_index = c * H * W + h * W + w;
                int output_index = h * W * C + w * C + c;
                output_data[output_index] = input_data[input_index];
            }
        }
    }
}

/******************************************************************
*  Function:               chw_to_cwh
*  Description:         将CHW数据重排为CWH形式
*  Parameter:         
*  @input_data       输入数据
*  @output_data    输出重排后的数据
*  Return:                   None
******************************************************************/
void chw_to_cwh(float *input_data, float *output_data) {
    int h, w, c;
    for (c = 0; c < C; ++c) {
        for (h = 0; h < H; ++h) {
            for (w = 0; w < W; ++w) {
                int input_index = c * H * W + h * W + w;
                int output_index = c * H * W + w * H + h;
                output_data[output_index] = input_data[input_index];
            }
        }
    }
}

/*******************************************************************
*  Function:            ReduceLogSum_c_0
*  Description:       ReduceLogSum算子当axes=0时的C-for循环实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，输出H*W个
*  Return:                 None
*******************************************************************/
void ReduceLogSum_c_0(float data[C * H * W], float* result) {
    for (size_t i = 0; i < H*W; i++) {
        float sum = 0.0;
        for (size_t j = 0; j < C; j++) {
            sum += data[j*H*W + i];
        }
        result[i] = log(sum);
    }
}

/*********************************************************************
*  Function:            ReduceLogSum_c_0_re
*  Description:       ReduceLogSum算子当axes=0时的C-数据重排实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，输出H*W个
*  Return:                 None
*********************************************************************/
void ReduceLogSum_c_0_re(float data[C * H * W], float* result) {
    float tmp[H*W*C];
    chw_to_hwc(data, tmp);
    for (size_t i = 0; i < H*W; i++) {
        float sum = 0.0;
        for (size_t j = 0; j < C; j++) {
            sum += tmp[i*C + j];
        }
        result[i] = log(sum);
    }
}

/*************************************************************************
*  Function:            ReduceLogSum_intr_0
*  Description:       ReduceLogSum算子当axes=0时的rvv-intrisic函数实现方法
*  Parameter:         同ReduceLogSum_c_0
*  Return:                 None
*************************************************************************/
void ReduceLogSum_intr_0(float data[C * H * W], float* result) {
    float tmp[H*W*C];
    chw_to_hwc(data, tmp);
    size_t vl = vsetvl_e32m4(C);
    for (size_t i = 0; i < H*W; i++) {
        float sum = 0.0;
        size_t vl_end = (C)%(vl);
        vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl); 
        for (size_t j=0; j < C-vl_end; j+=vl){
            vfloat32m4_t vinput = vle32_v_f32m4(&tmp[i * C + j], vl); 
            intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
        }
        b = vfredosum_vs_f32m4_f32m1 (b, intr_sum, b, vl);
        sum = vfmv_f_s_f32m1_f32(b);
        while (vl_end > 0){
            int m =  (i+1)*C - vl_end;
            sum = sum + tmp[m];
            vl_end--;
        }
        result[i] = log(sum);
    }
}

/***********************************************************************
*  Function:            ReduceLogSum_a_0
*  Description:       ReduceLogSum算子当axes=0时的rvv-内联汇编实现方法
                                    加-O3编译会打乱原有顺序，导致结果有误
*  Parameter:         同ReduceLogSum_c_0
*  Return:                 None
***********************************************************************/
void ReduceLogSum_a_0(float data[C * H * W], float* result) {
    float tmp[H*W*C];
    chw_to_hwc(data, tmp);
    size_t vl = vsetvlmax_e32m4();
    for (size_t i = 0; i < H*W; i++) {
        float sum = 0;
        size_t vl_end = (C)%(vl);
        asm volatile(
            "vmv.v.x                v8, %[zero]\n\t"
            "vmv.v.x                v24, %[zero]\n\t"
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m4\n\t"
            "vle.v                 v16, (%[da])\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "vfadd.vv              v8, v8, v16\n\t"
            "bnez                     %[len], 1b\n\t"
            "vfredosum.vs  v24, v8, v24\n\t"
            "vfmv.f.s                %[s], v24\n\t"
            : [s] "=f" (sum)
            : [da] "r" (&tmp[i*C]),
            [len]"r"(C - vl_end),
            [zero] "r"(0.0f)
            : "v8", "v16", "v24", "t0"
        );
        while (vl_end > 0){
            int m =  (i+1)*C - vl_end;
            sum = sum + tmp[m];
            vl_end--;
        }
        result[i] =  log(sum);
    }
}

/*******************************************************************
*  Function:            ReduceLogSum_c_1
*  Description:       ReduceLogSum算子当axes=1时的C-for循环实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，输出C*W个
*  Return:                 None
*******************************************************************/
void ReduceLogSum_c_1(float data[C * H * W], float* result) {
    for (size_t i = 0; i < C*W; i++) {
        float sum = 0.0;
        for (size_t k = 0; k<H; k++){
            sum += data[k * W + i%W + (i / W) * H * W];
        }
        result[i] = log(sum);
    }
}

/*********************************************************************
*  Function:            ReduceLogSum_c_1_re
*  Description:       ReduceLogSum算子当axes=1时的C-数据重排实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，输出C*W个
*  Return:                 None
*********************************************************************/
void ReduceLogSum_c_1_re(float data[C * H * W], float* result) {
    float tmp[C*W*H];
    chw_to_cwh(data, tmp);
    for (size_t i = 0; i < C*W; i++) {
        float sum = 0.0;
        for (size_t k = 0; k<H; k++){
            sum += tmp[i*H + k];
        }
        result[i] = log(sum);
    }
}

/*************************************************************************
*  Function:            ReduceLogSum_intr_1
*  Description:       ReduceLogSum算子当axes=1时的rvv-intrisic函数实现方法
*  Parameter:         同ReduceLogSum_c_1
*  Return:                 None
*************************************************************************/
void ReduceLogSum_intr_1(float data[C * H * W], float* result) {
    float tmp[C*W*H];
    chw_to_cwh(data, tmp);
    size_t vl = vsetvl_e32m4(H);
    for (size_t i = 0; i < C*W; i++) {
        float sum = 0.0;
        size_t vl_end = (H)%(vl);
        vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl); 
        for (size_t j = 0; j < H-vl_end; j += vl){
            vfloat32m4_t vinput = vle32_v_f32m4(&tmp[i * H + j], vl);   
            intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
        }
        b = vfredosum_vs_f32m4_f32m1 (b, intr_sum, b, vl); 
        sum = vfmv_f_s_f32m1_f32(b);
        while (vl_end > 0){
            int m =  (i+1)*H - vl_end;
            sum = sum + tmp[m];
            vl_end--;
        }
        result[i] = log(sum);
    }
}

/***********************************************************************
*  Function:            ReduceLogSum_a_1
*  Description:       ReduceLogSum算子当axes=1时的rvv-内联汇编实现方法
                                    加-O3编译会打乱原有顺序，导致结果有误
*  Parameter:         同ReduceLogSum_c_1
*  Return:                 None
***********************************************************************/
void ReduceLogSum_a_1(float data[C * H * W], float* result) {
    float tmp[C*W*H];
    chw_to_cwh(data, tmp);
    size_t vl = vsetvlmax_e32m4();
    for (size_t i = 0; i < C*W; i++) {
        float sum = 0;
        size_t vl_end = (H)%(vl);
        asm volatile(
            "vmv.v.x                v8, %[zero]\n\t"
            "vmv.v.x                v24, %[zero]\n\t"
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m4\n\t"
            "vle.v                 v16, (%[da])\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "vfadd.vv              v8, v8, v16\n\t"
            "bnez                     %[len], 1b\n\t"
            "vfredosum.vs  v24, v8, v24\n\t"
            "vfmv.f.s                %[s], v24\n\t"
            : [s] "=f" (sum)
            : [da] "r" (&tmp[i*H]),
            [len]"r"(H - vl_end),
            [zero] "r"(0.0f)
            : "v8", "v16", "v24", "t0"
        );
        while (vl_end > 0){
            int m =  (i+1)*H - vl_end;
            sum = sum + tmp[m];
            vl_end--;
        }
        result[i] =  log(sum);
    }
}

/*******************************************************************
*  Function:            ReduceLogSum_c_2
*  Description:       ReduceLogSum算子当axes=2时的C-for循环实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，输出C*H个，不用数据重排
*  Return:                 None
*******************************************************************/
void ReduceLogSum_c_2(float data[C * H * W], float* result) {
    for (size_t i = 0; i < C*H; i++) {
        float sum = 0.0;
        for (size_t k = 0; k<W; k++){
            sum += data[k + i*W];
        }
        result[i] = log(sum);
    }
}

/*************************************************************************
*  Function:            ReduceLogSum_intr_2
*  Description:       ReduceLogSum算子当axes=2时的rvv-intrisic函数实现方法
*  Parameter:         同ReduceLogSum_c_2
*  Return:                 None
*************************************************************************/
void ReduceLogSum_intr_2(float data[C * H * W], float* result) {
    size_t vl = vsetvl_e32m4(W);
    for (size_t i = 0; i < C*H; i++) {
        float sum = 0.0;
        size_t vl_end = (W)%(vl);
        vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl); 
        for (size_t j = 0; j < W-vl_end; j += vl){
            vfloat32m4_t vinput = vle32_v_f32m4(&data[i * W + j], vl);  
            intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
        }
        b = vfredosum_vs_f32m4_f32m1 (b, intr_sum, b, vl);
        sum = vfmv_f_s_f32m1_f32(b);
        while (vl_end > 0){
            int m =  (i+1)*W - vl_end;
            sum = sum + data[m];
            vl_end--;
        }
        result[i] = log(sum);
    }
}

/***********************************************************************
*  Function:            ReduceLogSum_a_2
*  Description:       ReduceLogSum算子当axes=2时的rvv-内联汇编实现方法
                                    加-O3编译会打乱原有顺序，导致结果有误
*  Parameter:         同ReduceLogSum_c_2
*  Return:                 None
***********************************************************************/
void ReduceLogSum_a_2(float data[C * H * W], float* result) {
    size_t vl = vsetvlmax_e32m4();
    for (size_t i = 0; i < C*H; i++) {
        float sum = 0;
        size_t vl_end = (W)%(vl);
        asm volatile(
            "vmv.v.x                v8, %[zero]\n\t"
            "vmv.v.x                v24, %[zero]\n\t"
            "1: \n\t"
            "vsetvli                  t0, %[len], e32, m4\n\t"
            "vle.v                 v16, (%[da])\n\t"
            "sub                       %[len], %[len], t0\n\t"
            "slli                         t0, t0, 2\n\t"
            "add                       %[da], %[da], t0\n\t"
            "vfadd.vv              v8, v8, v16\n\t"
            "bnez                     %[len], 1b\n\t"
            "vfredosum.vs  v24, v8, v24\n\t"
            "vfmv.f.s                %[s], v24\n\t"
            : [s] "=f" (sum)
            : [da] "r" (&data[i*W]),
            [len]"r"(W - vl_end),
            [zero] "r"(0.0f)
            : "v8", "v16", "v24", "t0"
        );
        while (vl_end > 0){
            int m =  (i+1)*W - vl_end;
            sum = sum + data[m];
            vl_end--;
        }
        result[i] =  log(sum);
    }
}

/***********************************************************************
*  Function:            ReduceLogSum_c_d
*  Description:       ReduceLogSum算子当axes=None时的C-for循环实现方法
*  Parameter:         
*  @data                   输入数据
*  @result                输出数据，所有维度的数一起计算，得到一个值
*  Return:                 None
***********************************************************************/
void ReduceLogSum_c_d(float data[C * H * W], float *result) {
    float sum = 0.0;
    for (size_t i = 0; i < C*H*W; i++) {
        sum += data[i];
    }
    *result = log(sum);
}

/*****************************************************************************
*  Function:            ReduceLogSum_intr_d
*  Description:       ReduceLogSum算子当axes=None时的rvv-intrisic函数实现方法
*  Parameter:         同ReduceLogSum_c_d
*  Return:                 None
*****************************************************************************/
void ReduceLogSum_intr_d(float data[C * H * W], float *result) {
    float sum = 0.0;
    size_t vl = vsetvlmax_e32m4(); 
    size_t vl_end = (C * H * W)%(vl);
    vfloat32m1_t b = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl);
    for (size_t i = 0; i < C * H * W - vl_end; i += vl) {
        vfloat32m4_t vinput = vle32_v_f32m4(&data[i], vl);
        intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
    }
    b = vfredosum_vs_f32m4_f32m1 (b, intr_sum, b, vl); 
    sum = vfmv_f_s_f32m1_f32(b); 
    while (vl_end > 0){
        int m = C * H * W - vl_end;
        sum = sum + data[m];
        vl_end--;
    }
    *result =  log(sum);
}

/***************************************************************************
*  Function:            ReduceLogSum_a_d
*  Description:       ReduceLogSum算子当axes=None时的rvv-内联汇编实现方法
*  Parameter:         同ReduceLogSum_c_d
*  Return:                 None
***************************************************************************/
void ReduceLogSum_a_d(float data[C * H * W], float *result) {
    float sum = 0;
    size_t vl = vsetvlmax_e32m4();
    size_t vl_end = (C * H * W)%(vl);
    asm volatile(
        "vmv.v.x                v8, %[zero]\n\t"
        "vmv.v.x                v24, %[zero]\n\t"
        "1: \n\t"
        "vsetvli                  t0, %[len], e32, m4\n\t"
        "vle.v                 v16, (%[da])\n\t"
        "sub                       %[len], %[len], t0\n\t"
        "slli                         t0, t0, 2\n\t"
        "add                       %[da], %[da], t0\n\t"
        "vfadd.vv              v8, v8, v16\n\t"
        "bnez                     %[len], 1b\n\t"
        "vfredosum.vs  v24, v8, v24\n\t"
        "vfmv.f.s                %[s], v24\n\t"

        : [s] "=f" (sum)
        : [da] "r" (&data[0]),
        [len]"r"(C * H * W - vl_end),
        [zero] "r"(0.0f)
        : "v8", "v16", "v24", "t0"
    );
    while (vl_end > 0){
        int m =  C * H * W - vl_end;
        sum = sum + data[m];
        vl_end--;
    }
    *result =  log(sum);
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
    int axes  = 2;                    //指定哪个维度上运算, 0 / 1 / 2 / 其他
    float output_c, output_c0[H*W], output_c0r[H*W], output_c1[C*W], output_c1r[C*W], output_c2[C*H];
    float output_intr, output_intr0[H*W], output_intr1[C*W], output_intr2[C*H];
    float output_a, output_a0[H*W], output_a1[C*W], output_a2[C*H];
    clock_t start, end, start2, end2, start3, end3, start4, end4;
    double cpu_time, cpu_time2, cpu_time3, cpu_time4;

    printf("input_tensor[%d*%d*%d]: \n", C, H, W);
    //初始化 输入数据，随机值，1/2
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                data[i * H * W + j * W + k] = rand()%2+1;
                //printf("%f ", data[i * H * W + j * W + k]);   
            }
            //printf("\n");
        }
        //printf("\n");
    }

    switch (axes) {
        case 0: // ----------------------沿着第一个维度求和 -------------------------
            //--------------------------------测试方法1: C-for---------------------------------
            start = clock();
            ReduceLogSum_c_0(data, output_c0);
            end = clock();
            cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_c:\n", axes);
            for (int i = 0; i < H*W; i++) {
                printf("%f ", output_c0[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            } */
            printf("\n");

            //------------------------------测试方法2: C-数据重排-------------------------------
            start2 = clock();
            ReduceLogSum_c_0_re(data, output_c0r);
            end2 = clock();
            cpu_time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_c_re:\n", axes);
            for (int i = 0; i < H*W; i++) {
                printf("%f ", output_c0r[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            int isEqual_re0 = compare_result(output_c0, output_c0r, H*W);

           //----------------------------测试方法3: rvv-intrisic-m4-----------------------------
           start3 = clock();
            ReduceLogSum_intr_0(data, output_intr0);
            end3 = clock();
            cpu_time3 = (double)(end3 - start3) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_intr:\n", axes);
            for (int i = 0; i < H*W; i++) {
                printf("%f ", output_intr0[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            int isEqual_intr0 = compare_result(output_c0, output_intr0, H*W);

            //----------------------------测试方法4: rvv-assembly-m4---------------------------
            start4 = clock();
            ReduceLogSum_a_0(data, output_a0);
            end4 = clock();
            cpu_time4 = (double)(end4 - start4) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_a:\n", axes);
            for (int i = 0; i < H*W; i++) {
                printf("%f ", output_a0[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            //int isEqual_asm0  = compare_result(output_c0, output_a0, H*W);

            //判断结果是否一致
/*             printf("\n----Results Equal or NotEqual----\n");
            if (isEqual_re0) printf("C-re and C: Equal.\n");
            else printf("C-re and C: NotEqual.\n");
            if (isEqual_intr0) printf("rvv_intr_m4 and C: Equal.\n");
            else printf("rvv_intr_m4 and C: NotEqual.\n");
            if (isEqual_asm0) printf("rvv_asm_m4 and C: Equal.\n");
            else printf("rvv_asm_m4 and C: NotEqual.\n"); */

            printf("\n-----------Time Cost-----------\n");
            printf("C time is %f s\n", cpu_time);
            printf("C_re_time is %f\n", cpu_time2); 
            printf("rvv_intr_m4 time is %f s\n", cpu_time3);
            printf("rvv_asm_m4 time is %f s\n", cpu_time4);

            break;

        case 1: // ---------------------沿着第二个维度求和 -------------------------
            //--------------------------------测试方法1: C-for---------------------------------
            start = clock();
            ReduceLogSum_c_1(data, output_c1);
            end = clock();
            cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_c:\n", axes);
            for (int i = 0; i < C*W; i++) {
                printf("%f ", output_c1[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            } */
            printf("\n");

            //------------------------------测试方法2: C-数据重排-------------------------------
            start2 = clock();
            ReduceLogSum_c_1_re(data, output_c1r);
            end2 = clock();
            cpu_time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_c_re:\n", axes);
            for (int i = 0; i < C*W; i++) {
                printf("%f ", output_c1r[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            int isEqual_re1 = compare_result(output_c1, output_c1r, C*W);

            //----------------------------测试方法3: rvv-intrisic-m4-----------------------------
            start3 = clock();
            ReduceLogSum_intr_1(data, output_intr1);
            end3 = clock();
            cpu_time3 = (double)(end3 - start3) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_intr:\n", axes);
            for (int i = 0; i < C*W; i++) {
                printf("%f ", output_intr1[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            int isEqual_intr1 = compare_result(output_c1, output_intr1, C*W);

            //----------------------------测试方法4: rvv-assembly-m4---------------------------
            start4 = clock();
            ReduceLogSum_a_1(data, output_a1);
            end4 = clock();
            cpu_time4 = (double)(end4 - start4) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_a:\n", axes);
            for (int i = 0; i < C*W; i++) {
                printf("%f ", output_a1[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */
            //int isEqual_asm1  = compare_result(output_c1, output_a1, C*W);

            //判断结果是否一致
/*             printf("\n----Results Equal or NotEqual----\n");
            if (isEqual_re1) printf("C-re and C: Equal.\n");
            else printf("C-re and C: NotEqual.\n");
            if (isEqual_intr1) printf("rvv_intr_m4 and C: Equal.\n");
            else printf("rvv_intr_m4 and C: NotEqual.\n");
            if (isEqual_asm1) printf("rvv_asm_m4 and C: Equal.\n");
            else printf("rvv_asm_m4 and C: NotEqual.\n"); */

            printf("\n-----------Time Cost-----------\n");
            printf("C time is %f s\n", cpu_time);
            printf("C_re_time is %f\n", cpu_time2); 
            printf("rvv_intr_m4 time is %f s\n", cpu_time3);
            printf("rvv_asm_m4 time is %f s\n", cpu_time4);

            break;

        case 2: // ---------------------沿着第三个维度求和 -------------------------
            //--------------------------------测试方法1: C-for---------------------------------
            start = clock();
            ReduceLogSum_c_2(data, output_c2);
            end = clock();
            cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_c:\n", axes);
            for (int i = 0; i < C*H; i++) {
                printf("%f ", output_c2[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n"); */

            //----------------------------测试方法2: rvv-intrisic-m4----------------------------
            start2 = clock();
            ReduceLogSum_intr_2(data, output_intr2);
            end2 = clock();
            cpu_time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_intr:\n", axes);
            for (int i = 0; i < C*H; i++) {
                printf("%f ", output_intr2[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n");
            int isEqual_intr2 = compare_result(output_c2, output_intr2, C*H); */

            //----------------------------测试方法3: rvv-assembly-m4---------------------------
            start4 = clock();
            ReduceLogSum_a_2(data, output_a2);
            end4 = clock();
            cpu_time4 = (double)(end4 - start4) / CLOCKS_PER_SEC;
/*             printf("Output after applying axes: %d ReduceLogSum_a:\n", axes);
            for (int i = 0; i < C*H; i++) {
                printf("%f ", output_a2[i]);
                if (i % 5 == 4) {
                    printf("\n");
                }
            }
            printf("\n");
            int isEqual_asm2  = compare_result(output_c2, output_a2, C*H); */

            //判断结果是否一致
/*             printf("\n----Results Equal or NotEqual----\n");
            if (isEqual_intr2) printf("rvv_intr_m4 and C: Equal.\n");
            else printf("rvv_intr_m4 and C: NotEqual.\n");
            if (isEqual_asm2) printf("rvv_asm_m4 and C: Equal.\n");
            else printf("rvv_asm_m4 and C: NotEqual.\n"); */

            printf("\n-----------Time Cost-----------\n");
            printf("C time is %f s\n", cpu_time);
            printf("rvv_intr_m4 time is %f\n", cpu_time2); 
            printf("rvv_asm_m4 time is %f s\n", cpu_time4);

            break;

        default:   // ---------------------所有元素平方求和 -------------------------
            //--------------------------------测试方法1: C-for---------------------------------
            start = clock();
            ReduceLogSum_c_d(data, &output_c);
            end = clock();
            cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
            printf("Output after applying all elements ReduceLogSum_c:\n");
            printf("%f \n", output_c);

            //----------------------------测试方法2: rvv-intrisic-m4----------------------------
            start2 = clock();
            ReduceLogSum_intr_d(data, &output_intr);
            end2 = clock();
            cpu_time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
            printf("Output after applying all elements ReduceLogSum_intr:\n");
            printf("%f \n", output_intr);
            int isEqual_intr_d = compare_result(&output_c, &output_intr, 1);

            start4 = clock();
            ReduceLogSum_a_d(data, &output_a);
            end4 = clock();
            cpu_time4 = (double)(end4 - start4) / CLOCKS_PER_SEC;
            printf("Output after applying all elements ReduceLogSum_assembly:\n");
            printf("%f \n", output_a);
            int isEqual_asm_d  = compare_result(&output_c, &output_a, 1);

            //判断结果是否一致
            printf("\n----Results Equal or NotEqual----\n");
            if (isEqual_intr_d) printf("rvv_intr_m4 and C: Equal.\n");
            else printf("rvv_intr_m4 and C: NotEqual.\n");
            if (isEqual_asm_d) printf("rvv_asm_m4 and C: Equal.\n");
            else printf("rvv_asm_m4 and C: NotEqual.\n");

            printf("\n-----------Time Cost-----------\n");
            printf("C time is %f s\n", cpu_time);
            printf("rvv_intr_m4 time is %f\n", cpu_time2); 
            printf("rvv_asm_m4 time is %f s\n", cpu_time4);

            break;
    }

    return 0;
}