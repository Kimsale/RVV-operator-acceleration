/***********************************************************************************************************
*  @File:                    layer_normalization.c
*  @Author:             JSE
*  @Date:                 2024-5-16
*  @Description:   layer_normalization算子的性能测试代码，包含普通C-for循环、rvv-intrisic函数、rvv-内联汇编3种算子实现方法； 
                                    layer_normalization算子执行异或操作
                                    输入输出数据类型为int32，维度为1；
                                    两种rvv寄存器分组模式都为m4
************************************************************************************************************/


#include <riscv_vector.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

// 计算张量的均值
void reduce_mean(float *input, float *mean, int *shape, int rank, int axis, int norm_size) {
    int outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= shape[i];
    }

    int inner_size = norm_size;

    for (int i = 0; i < outer_size; ++i) {
        mean[i] = 0;
        for (int j = 0; j < inner_size; ++j) {
            mean[i] += input[i * inner_size + j];
        }
        mean[i] /= inner_size;
    }
}

// 计算张量的标准差
void compute_stddev(float *input, float *mean, float *stddev, int *shape, int rank, int axis, int norm_size, float epsilon) {
    int outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= shape[i];
    }

    int inner_size = norm_size;

    for (int i = 0; i < outer_size; ++i) {
        stddev[i] = 0;
        for (int j = 0; j < inner_size; ++j) {
            float diff = input[i * inner_size + j] - mean[i];
            stddev[i] += diff * diff;
        }
        stddev[i] /= inner_size;
        stddev[i] = sqrt(stddev[i] + epsilon);
    }
}

// Layer Normalization核心函数
void layer_normalization(float *X, float *Scale, float *B, float *Y, float *Mean, float *InvStdDev, int *shape, int rank, int axis, float epsilon) {
    int norm_size = 1;
    for (int i = axis; i < rank; ++i) {
        norm_size *= shape[i];
    }

    // 阶段1: 计算均值和标准差
    reduce_mean(X, Mean, shape, rank, axis, norm_size);
    compute_stddev(X, Mean, InvStdDev, shape, rank, axis, norm_size, epsilon);

    int outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= shape[i];
    }

    // 计算InvStdDev (逆标准差)
    for (int i = 0; i < outer_size; ++i) {
        InvStdDev[i] = 1.0f / InvStdDev[i];
    }

    // 阶段2: 标准化，缩放和移位
    for (int i = 0; i < outer_size; ++i) {
        for (int j = 0; j < norm_size; ++j) {
            int idx = i * norm_size + j;
            Y[idx] = (X[idx] - Mean[i]) * InvStdDev[i] * Scale[j] + B[j];
        }
    }
}

int main() {
    // 示例输入
    int rank = 3;
    int shape[] = {2, 3, 4}; // 假设输入张量的形状为 [2, 3, 4]
    float X[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0
    };
    float Scale[] = {1.0, 1.0, 1.0, 1.0}; // 假设缩放张量的形状与规范化维度匹配
    float B[] = {0.0, 0.0, 0.0, 0.0}; // 假设偏移张量的形状与规范化维度匹配
    float epsilon = 1e-05;
    int axis = 1; // 假设在第二个维度进行规范化

    float Y[24];
    float Mean[6];
    float InvStdDev[6];

    // 计算Layer Normalization
    layer_normalization(X, Scale, B, Y, Mean, InvStdDev, shape, rank, axis, epsilon);

    // 输出结果
    printf("Normalized Output:\n");
    for (int i = 0; i < 24; ++i) {
        printf("%f ", Y[i]);
        if ((i + 1) % 4 == 0) printf("\n");
    }

    return 0;
}
