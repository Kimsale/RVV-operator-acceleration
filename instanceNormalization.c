#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <time.h>
#include <math.h> 

// #define N 100
void scalar(float* input1, float* gamma, float* beta, int batch_size, int num_channels, int num_features, float epsilon, float* output_sca) {  
    for (int b = 0; b < batch_size; b++) {  
        for (int c = 0; c < num_channels; c++) {  
            int channel_start = b * num_channels * num_features + c * num_features;  
            float mean = 0.0f;  
            float variance = 0.0f;  
      
            // 计算均值  
            for (int f = 0; f < num_features; f++) {  
                mean += input1[channel_start + f];  
            }  
            mean /= num_features;  
  
            // 计算方差  
            for (int f = 0; f < num_features; f++) {  
                variance += pow(input1[channel_start + f] - mean, 2);  
            }  
            variance /= num_features;  
  
            // 归一化  
            for (int f = 0; f < num_features; f++) {  
                output_sca[channel_start + f] = gamma[c] * (input1[channel_start + f] - mean) / sqrt(variance + epsilon) + beta[c];  
            }   
        }  
    }  
}


void vector(float *input1, float *input2, float *gamma,float *beta,int batch_size,int num_channels,int num_features, float epsilon, float *output_vec, size_t len) {
    size_t vl;  
    size_t v2;
    size_t v3;

    for (int b = 0; b < batch_size; b++) {  
        for (int c = 0; c < num_channels; c++) {  
            int channel_start = b * num_channels * num_features + c * num_features;  
           
            float mean = 0.0f;  
            float variance= 0.0f;  
      
            // 计算均值  
/*             float final_sum = 0;
            for (int i = 0; i < num_features; i+= vl) { 
                vl = vsetvl_e32m4(len - i);
                //printf("vl = %d\t", vl);
                vfloat32m4_t va = vle32_v_f32m4(&input1[channel_start +i], vl);
                // vfloat32m4_t va = vle32_v_f32m4(&input1[channel_start +i], vl);
                vfloat32m4_t dest = vle32_v_f32m4(&input2[channel_start +i], vl);
                vfloat32m4_t scalar = vle32_v_f32m4(&input2[channel_start +i], vl);
                vfloat32m4_t sum = vfredosum_vs_f32m4_f32m4(dest, va, scalar, vl);
                float sum_2 =  vfmv_f_s_f32m4_f32(sum);
                final_sum = final_sum + sum_2;
            }  
            // printf("\n");
            mean = final_sum/num_features;
            //printf("mean: %f\t", mean); */

            size_t vl = vsetvl_e32m4(num_features);
            float sum = 0;
            size_t vl_end = (num_features)%(vl);
            vfloat32m1_t vsum = vfmv_v_f_f32m1(0.0f, vl); 
            vfloat32m4_t intr_sum = vfmv_v_f_f32m4(0.0f, vl); 
            for (int j = 0; j < num_features-vl_end; j+=vl) {
                //vl = vsetvl_e32m4(inner_size - j);
                vfloat32m4_t vinput = vle32_v_f32m4(&input1[channel_start +j], vl);
                intr_sum =  vfadd_vv_f32m4 (intr_sum, vinput, vl);
            }
            vsum = vfredosum_vs_f32m4_f32m1(vsum, intr_sum, vsum, vl);
            sum = vfmv_f_s_f32m1_f32(vsum); 
            while (vl_end > 0){
                int m = num_features - vl_end;
                sum = sum + input1[channel_start +m];
                vl_end--;
            } 
            mean = sum / num_features;

            // 计算方差  
/*             float final_sum_var = 0;
            for (int f = 0; f < num_features; f+=v2) { 
                 v2 = vsetvl_e32m4(len - f);
                vfloat32m4_t va = vle32_v_f32m4(&input1[channel_start +f], v2);

                // vfloat32m4_t vfmsub_vv_f32m4 (vfloat32m4_t vd, vfloat32m4_t vs1, vfloat32m4_t vs2, size_t vl);
                // vfloat32m4_t vfsub_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
                vfloat32m4_t op1 = vfsub_vf_f32m4 (va, mean, v2);
                vfloat32m4_t var = vfmul_vv_f32m4 (op1, op1, v2);

                vfloat32m4_t destv2 = vle32_v_f32m4(&input2[channel_start +f], v2);
                vfloat32m4_t scalarv2 = vle32_v_f32m4(&input2[channel_start +f], v2);
                vfloat32m4_t sumv2 = vfredosum_vs_f32m4_f32m4(destv2, var, scalarv2, v2);
                float sum_2v2 =  vfmv_f_s_f32m4_f32(sumv2);
                final_sum_var = final_sum_var + sum_2v2;
            }  
            variance = final_sum_var/num_features;
            //printf("variance: %f\n", variance);   */

            float square_sum = 0;
            vl_end = (num_features)%(vl);
            vfloat32m1_t vsquare_sum = vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m4_t intr_square_sum = vfmv_v_f_f32m4(0.0f, vl); 
            for (int j = 0; j < num_features-vl_end; j += vl) {
                vfloat32m4_t v = vle32_v_f32m4(&input1[channel_start +j], vl);
                vfloat32m4_t vdiff = vfsub_vf_f32m4(v, mean, vl);
                vfloat32m4_t vdiff_sq = vfmul_vv_f32m4(vdiff, vdiff, vl);
                intr_square_sum = vfadd_vv_f32m4(intr_square_sum, vdiff_sq, vl);
            }
            vsquare_sum = vfredosum_vs_f32m4_f32m1(vsquare_sum, intr_square_sum, vsquare_sum, vl);
            square_sum = vfmv_f_s_f32m1_f32(vsquare_sum); 
            while (vl_end > 0){
                int m = num_features - vl_end;
                square_sum = square_sum + (input1[channel_start +m] - mean) * (input1[channel_start +m] - mean);
                vl_end--;
            }
            variance = square_sum/num_features;
            float stddev = sqrtf(variance + epsilon);
            float inv_stddev = 1.0f / stddev;
  
            // // 归一化  
/*             for (int j = 0; j < num_features; j+=v3) {  
                // vfloat32m4_t vva =  vfsqrt_v_f32m4(vfadd_vv_f32m4(variance , epsilon, v3), v3);
                v3 = vsetvl_e32m4(len - j);
                float vva=sqrt(variance + epsilon);
                //  printf("vva: %f\n", vva); 
                vfloat32m4_t vvva = vle32_v_f32m4(&input1[channel_start + j], v3);
                vfloat32m4_t vvb = vfsub_vf_f32m4 (vvva, mean, v3);
                vfloat32m4_t vvc = vfmul_vf_f32m4(vvb, gamma[c], v3);
                vfloat32m4_t vvd = vfdiv_vf_f32m4(vvc, vva, v3);

                vfloat32m4_t vec_output = vfadd_vf_f32m4(vvd, beta[c], v3);
                // input[channel_start + f] = gamma[c] * (input[channel_start + f] - mean) / sqrt(variance + epsilon) + beta[c];  
                vse32_v_f32m4(&output_vec[channel_start+j], vec_output, v3);
            }   */
             for (int j = 0; j < num_features; j += vl) {
                vfloat32m4_t v = vle32_v_f32m4(&input1[channel_start + j], vl);
                vfloat32m4_t vdiff = vfsub_vf_f32m4(v, mean, vl);
                vfloat32m4_t vnorm = vfmul_vf_f32m4(vdiff, inv_stddev, vl);
                vnorm = vfmul_vf_f32m4(vnorm, gamma[c], vl);
                vnorm = vfadd_vf_f32m4(vnorm, beta[c], vl);
                vse32_v_f32m4(&output_vec[channel_start+j], vnorm, vl);
            }
        }  
    }  
}


// void scalar(int8_t *input1, float scale, int32_t zero_point, float *output, size_t len)  {
//     for (size_t i = 0; i < len; ++i) {
//         output[i] = (float)(input1[i] - zero_point) * scale;
//     }
// } 

int main() {
    int N = 1, H = 112, W = 112, C = 16;
    int nums = N*H*W*C;

    float input1[nums];
    float input2[nums];
    float output_vec[nums];
    float output_sca[nums];  

    float gamma[H*W];  
    float beta[H*W];  
  
/*     int batch_size = 2;  
    int num_channels = 3;  
    int num_features = 4;      */ 
    float epsilon = 1e-5f; 

    clock_t start, end;
    double cpu_time;

/*      for (int i = 0; i < nums; i++) {
        input1[i] = i;  // 示例输入数据，可以根据实际需求修改
    } */
    for (int i = 0; i < nums; i++) {
            input2[i] = 0.0f;
    } 

    for (int i = 0; i < nums; ++i) input1[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < C; ++i) gamma[i] = 1.0f;
    for (int i = 0; i < C; ++i) beta[i] = 0.0f;

    start = clock();
	scalar(input1, gamma, beta, N, C, H*W, epsilon, output_sca);
	end = clock();
	cpu_time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time);//输出运行时间

    start = clock();
	vector(input1, input2,  gamma, beta, N, C, H*W, epsilon, output_vec, nums);
	end = clock();
	cpu_time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("vector execution time %f seconds\n",cpu_time);//输出运行时间

    int pass = 1;
    for (int i = 0; i < nums; i++) {
        if(output_vec[i]-output_sca[i] > 1e-4){
            printf("index %d failed, %f=!%f\n", i, output_sca[i], output_vec[i]);
		    pass = 0;
        }
    }
    if (pass)
		printf("passed\n"); 
    return 0;
}



// #include <stdio.h>  
// #include <math.h>  
// #include <stdlib.h>  
  
// void instance_normalization(float* input, float* gamma, float* beta, int batch_size, int num_channels, int num_features, float epsilon) {  
//     for (int b = 0; b < batch_size; b++) {  
//         for (int c = 0; c < num_channels; c++) {  
//             int channel_start = b * num_channels * num_features + c * num_features;  
//             float mean = 0.0f;  
//             float variance = 0.0f;  
  
//             // 计算均值  
//             for (int f = 0; f < num_features; f++) {  
//                 mean += input[channel_start + f];  
//             }  
//             mean /= num_features;  
  
//             // 计算方差  
//             for (int f = 0; f < num_features; f++) {  
//                 variance += pow(input[channel_start + f] - mean, 2);  
//             }  
//             variance /= num_features;  
  
//             // 归一化  
//             for (int f = 0; f < num_features; f++) {  
//                 input[channel_start + f] = gamma[c] * (input[channel_start + f] - mean) / sqrt(variance + epsilon) + beta[c];  
//             }  
//         }  
//     }  
// }  
  
// int main() {  
//     // 示例输入数据：假设有 2 个样本，每个样本有 3 个通道，每个通道有 4 个特征  
//     float input[] = {  
//         1.0f, 2.0f, 3.0f, 4.0f,  // 样本 1，通道 1  
//         5.0f, 6.0f, 7.0f, 8.0f,  // 样本 1，通道 2  
//         9.0f, 10.0f, 11.0f, 12.0f, // 样本 1，通道 3  
//         13.0f, 14.0f, 15.0f, 16.0f, // 样本 2，通道 1  
//         17.0f, 18.0f, 19.0f, 20.0f, // 样本 2，通道 2  
//         21.0f, 22.0f, 23.0f, 24.0f  // 样本 2，通道 3  
//     };  
  
//     // 初始化 gamma 和 beta 为 1 和 0  
//     float gamma[] = {1.0f, 1.0f, 1.0f};  
//     float beta[] = {0.0f, 0.0f, 0.0f};  
  
//     int batch_size = 2;  
//     int num_channels = 3;  
//     int num_features = 4;  
//     float epsilon = 1e-5f;  
  
//     // 执行 Instance Normalization  
//     instance_normalization(input, gamma, beta, batch_size, num_channels, num_features, epsilon);  
  
//     // 打印归一化后的结果  
//     for (int i = 0; i < batch_size * num_channels * num_features; i++) {  
//         printf("%f ", input[i]);  
//     }  
//     printf("\n");  
  
//     return 0;  
// }