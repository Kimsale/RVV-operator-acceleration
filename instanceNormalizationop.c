#include <stdio.h>
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
            float final_sum = 0;
            for (int i = 0; i < num_features; i+= vl) {  
                vl = vsetvl_e32m1(len - i);
                vfloat32m1_t va = vle32_v_f32m1(&input1[channel_start +i], vl);
                vfloat32m1_t dest = vle32_v_f32m1(&input2[channel_start +i], vl);
                vfloat32m1_t scalar = vle32_v_f32m1(&input2[channel_start +i], vl);
                vfloat32m1_t sum = vfredosum_vs_f32m1_f32m1(dest, va, scalar, vl);
                float sum_2 =  vfmv_f_s_f32m1_f32(sum);
                final_sum = final_sum + sum_2;
            }  
            mean = final_sum/num_features;
            // 计算方差  
            float final_sum_var = 0;
            for (int f = 0; f < num_features; f+=v2) { 
                 v2 = vsetvl_e32m1(len - f);
                vfloat32m1_t va = vle32_v_f32m1(&input1[channel_start +f], v2);
                vfloat32m1_t op1 = vfsub_vf_f32m1 (va, mean, v2);
                vfloat32m1_t var = vfmul_vv_f32m1 (op1, op1, v2);
                vfloat32m1_t destv2 = vle32_v_f32m1(&input2[channel_start +f], v2);
                vfloat32m1_t scalarv2 = vle32_v_f32m1(&input2[channel_start +f], v2);
                vfloat32m1_t sumv2 = vfredosum_vs_f32m1_f32m1(destv2, var, scalarv2, v2);
                float sum_2v2 =  vfmv_f_s_f32m1_f32(sumv2);
                final_sum_var = final_sum_var + sum_2v2;
            }  
            variance = final_sum_var/num_features;
            // // 归一化  
            for (int j = 0; j < num_features; j+=v3) {  
                v3 = vsetvl_e32m1(len - j);
                float vva=sqrt(variance + epsilon);
                vfloat32m1_t vvva = vle32_v_f32m1(&input1[channel_start + j], v3);
                vfloat32m1_t vvb = vfsub_vf_f32m1 (vvva, mean, v3);
                vfloat32m1_t vvc = vfmul_vf_f32m1(vvb, gamma[c], v3);
                vfloat32m1_t vvd = vfdiv_vf_f32m1(vvc, vva, v3);
                vfloat32m1_t vec_output = vfadd_vf_f32m1(vvd, beta[c], v3);
                vse32_v_f32m1(&output_vec[channel_start+j], vec_output, v3);
            }  
        }  
    }  
}


void vectorop1(float *input1, float *input2, float *gamma,float *beta,int batch_size,int num_channels,int num_features, float epsilon, float *output_vec, size_t len) {
    size_t vl;  
    size_t v2;
    size_t v3;

    for (int b = 0; b < batch_size; b++) {  
        for (int c = 0; c < num_channels; c++) {  
            int channel_start = b * num_channels * num_features + c * num_features;  
           
            float mean = 0.0f;  
            float variance= 0.0f;  
      
            // 计算均值  
            float final_sum = 0;
            for (int i = 0; i < num_features; i+= vl) {  
                vl = vsetvl_e32m1(len - i);
                // printf("vl = %d\t", vl);
                vfloat32m1_t va = vle32_v_f32m1(&input1[channel_start + i], vl);
                // input1 += vl;
                // vfloat32m1_t va = vle32_v_f32m1(&input1[channel_start +i], vl);
                vfloat32m1_t dest = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t scalar = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t sum = vfredosum_vs_f32m1_f32m1(dest, va, scalar, vl);
                float sum_2 =  vfmv_f_s_f32m1_f32(sum);
                final_sum = final_sum + sum_2;
            }  
            // printf("\n");
            mean = final_sum/num_features;
            // printf("mean: %f\t", mean);

            // 计算方差  
            float final_sum_var = 0;
            for (int f = 0; f < num_features; f+=v2) { 
                 v2 = vsetvl_e32m1(len - f);
                vfloat32m1_t va = vle32_v_f32m1(&input1[channel_start +f], v2);

                // vfloat32m1_t vfmsub_vv_f32m1 (vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl);
                // vfloat32m1_t vfsub_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
                vfloat32m1_t op1 = vfsub_vf_f32m1 (va, mean, v2);
                vfloat32m1_t var = vfmul_vv_f32m1 (op1, op1, v2);

                vfloat32m1_t destv2 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t scalarv2 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t sumv2 = vfredosum_vs_f32m1_f32m1(destv2, var, scalarv2, v2);
                float sum_2v2 =  vfmv_f_s_f32m1_f32(sumv2);
                final_sum_var = final_sum_var + sum_2v2;
            }  
            variance = final_sum_var/num_features;
            // printf("variance: %f\n", variance);  
  
            // // 归一化  
            for (int j = 0; j < num_features; j+=v3) {  
                // vfloat32m1_t vva =  vfsqrt_v_f32m1(vfadd_vv_f32m1(variance , epsilon, v3), v3);
                v3 = vsetvl_e32m1(len - j);
                float vva=sqrt(variance + epsilon);
                //  printf("vva: %f\n", vva); 
                vfloat32m1_t vvva = vle32_v_f32m1(&input1[channel_start + j], v3);
                vfloat32m1_t vvb = vfsub_vf_f32m1 (vvva, mean, v3);
                vfloat32m1_t vvc = vfmul_vf_f32m1(vvb, gamma[c], v3);
                vfloat32m1_t vvd = vfdiv_vf_f32m1(vvc, vva, v3);

                vfloat32m1_t vec_output = vfadd_vf_f32m1(vvd, beta[c], v3);
                // input[channel_start + f] = gamma[c] * (input[channel_start + f] - mean) / sqrt(variance + epsilon) + beta[c];  
                vse32_v_f32m1(&output_vec[channel_start+j], vec_output, v3);
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
    int nums = 48;
//    float input1[48];
    float input1[48];

    float input2[48];
    float output_vec[48];
    float output_sca[48];  

    float gamma[] = {1.0f, 1.0f, 1.0f};  
    float beta[] = {0.0f, 0.0f, 0.0f};  
  
    int batch_size = 2;  
    int num_channels = 3;  
    int num_features = 8;  
    float epsilon = 1e-5f; 

    clock_t start, end;
    double cpu_time;

    for (int i = 0; i < nums; i++) {
        input1[i] = i;  // 示例输入数据，可以根据实际需求修改
    }
    for (int i = 0; i < nums; i++) {
            input2[i] = 0.0f;
    }

    start = clock();
	scalar(input1, gamma, beta, batch_size, num_channels, num_features, epsilon, output_sca);
	end = clock();
	cpu_time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("scalar execution time %f seconds\n",cpu_time);//输出运行时间  


    start = clock();
	vector(input1, input2,  gamma, beta, batch_size, num_channels, num_features, epsilon, output_vec, nums);
	end = clock();
	cpu_time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("vector execution time %f seconds\n",cpu_time);//输出运行时间

    start = clock();
	vectorop1(input1, input2,  gamma, beta, batch_size, num_channels, num_features, epsilon, output_vec, nums);
	end = clock();
	cpu_time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("vectorop1 execution time %f seconds\n",cpu_time);//输出运行时间


    int pass = 1;
    for (int i = 0; i < nums; i++) {
        if(output_vec[i]-output_sca[i] > 1e-6){
            printf("index %d failed, %f=!%f\n", i, output_sca[i], output_vec[i]);
		    pass = 0;
        }
    }
    if (pass)
		printf("passed\n");

    printf("\n");
    return 0;
}
