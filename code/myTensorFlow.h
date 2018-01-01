#ifndef _MYTF_H_
#define _MYTF_H_

#define SAME 1
#define VALID 0

#define TRUE 1
#define FALSE 0

#define NUM_CLASS 1001

/************************Layers*************************/
void conv_relu_layer (float* output, float* kernels, float* input, float* bias, 
			 int kernel_h,int kernel_w, int input_size, int input_channels, 
			 int output_channels, int stride,int is_same);
void conv_relu_BN_layer (float* output, float* kernels, float* input, float* beta, 
			 float* mean, float* var, int kernel_h,int kernel_w, int input_size, 
			 int input_channels, int output_channels, int stride,int is_same);
void affine_relu_layer(float* output, float* weights, float* input, float*bias,
			 int input_size, int output_size);
void affine_layer(float* output, float* weights, float* input, float*bias,
			 int input_size, int output_size);
void dropout (float* tensor, int input_size, float rate);

/************************Operations*********************/

void reorder (float* input, float* output, int size, int channels);
float sum (float* input, int size);
void matmul(float* weight, float* input, float* output, int input_size, int output_size);
void conv_2d(float* output, float* kernels, float* input, float* bias, 
			 int kernel_h,int kernel_w, int input_size, int input_channels, 
			 int output_channels, int stride,int is_same);
void max_pool(float* input, float*output,int input_size ,int size, int channles, int stride);
void avg_pool(float* input, float* output, int input_size ,int size, int channles, int stride);
void global_avg_pool(float* input, float* output, int size, int channels);
void relu(float* tensor, int size, int channels); //relu can be done in-place
void concat(float *input_a, float *input_b, float *input_c, float *input_d, float* output,
			int size, int channel_a, int channel_b, int channel_c, int channel_d);
void softmax(float *tensor, int length);
int argmax(float* tensor, int length);
void batch_norm(float* input, float* var, float* mean,
				float* beta, int size, int channel);
float* get_tensor(int size, int channels);
float* get_kernel(int size, int channels_in, int channels_out);
float* get_weight(int input, int output);
void destroy_tensor(float* tensor);

/***************************Utilities********************/
void show_tensor(float* mat, int size_z, int size_y, int size_x);
char* search_name(char* path, int index);
int top_5_hit (float* result, int label);

/**************************File Operations***************/
int read_data(char* path, float* array, int size);

#endif


