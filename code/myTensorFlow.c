#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "myTensorFlow.h"

int top_5_hit (float* result, int label){
	float top_5_prob[5] = {0};
	int top_5_label[5] = {0};

	for (int i = 0;i < NUM_CLASS;i ++){
		if (result[i] < top_5_prob[4]) continue;
		for(int j = 3;j >= 0;j --){
			if (result[i] < top_5_prob[j]){
				top_5_prob[j+1] = result[i];
				top_5_label[j+1] = i;
				break;
			} else if (j == 0) {
				top_5_prob[0] = result[i];
				top_5_label[0] = i;
			}
		}
	}
	for (int i=0;i<5;i++){
		printf("%d@",top_5_label[i]);
		printf("%f\n",top_5_prob[i]);
	}

	return 0;
}


void conv_relu_BN_layer (float* output, float* kernels, float* input, float* beta, 
			 float* mean, float* var, int kernel_h,int kernel_w, int input_size, 
			 int input_channels, int output_channels, int stride,int is_same){
	int output_size;
	if (is_same){
		output_size = input_size;
	} else {
		if (kernel_h!=kernel_w) {
			printf("%s\n", "ASDFASDFASFA!!!!");
		}
		output_size = (input_size-kernel_w)/stride+1;
	}
	conv_2d(output, kernels, input, NULL, 
			 kernel_h,kernel_w, input_size,input_channels, 
			 output_channels, stride,is_same);
	batch_norm(output, var, mean, beta, output_size, output_channels);
	relu(output,output_size,output_channels);
}


void batch_norm(float* input, float* var, float* mean,
				float* beta, int size, int channels) {
	int h,w,c;
	float temp;
	for(c=0;c<channels;c++){
		for(h=0;h<size;h++){
			for(w=0;w<size;w++){
				temp = *(input+c*(size*size)+h*size+w);
				temp = ((temp-*(mean+c))/sqrt(*(var+c)+0.001))+*(beta+c); //epsilon=0.001--tf
				*(input+c*(size*size)+h*size+w) = temp;
			}
		}
	}
}

//We assume that avgpool is always 'SAME' pooling
void global_avg_pool(float* input, float* output, int size, int channels){
	int c,h,w;
	float sum;
	for(c=0;c<channels;c++){ //fro each channel
		sum = 0; //set the max to 0
		for (h=0;h<size;h++){ //loop through all elements in the window
			for(w=0;w<size;w++){
					sum += *(input+c*(size*size)+h*size+w);
			}
		}
		*(output+c) = sum/(size*size);
	}
}


//We assume that avgpool is always 'SAME' pooling
void avg_pool(float* input, float* output, int input_size ,int size, int channels, int stride){
	int c,out_h,out_w;
	int h,w;
	int in_h,in_w;
	int output_size;
	float local_sum;
	int half;
	int num_count;

	output_size = input_size;
	half = (size-1)/2;
	for(c=0;c<channels;c++){ //fro each channel
		for(out_h=0;out_h<output_size;out_h++){ //for each column
			for(out_w=0;out_w<output_size;out_w++){ //for each row
				local_sum = 0; //set the max to 0
				num_count = 0; //number of present numbers within the window
				for (h=-half;h<half+1;h++){ //loop through all elements in the window
					for(w=-half;w<half+1;w++){
						in_h = stride*out_h+h;
						in_w = stride*out_w+w;
						if((in_h<0)||(in_h>=input_size)||(in_w<0)||(in_w>=input_size)){
							continue; //skip this loop, nothing to pick here
						} else {
							num_count ++;
							local_sum += *(input+c*(input_size*input_size)+in_h*input_size+in_w);
						}
					}
				}
				*(output+c*(output_size*output_size)+out_h*output_size+out_w) = local_sum/(num_count);
			}
		}
	}
}

void reorder (float* input, float* output, int size, int channels){
	int c,h,w;
	float temp;
	int count=0;
	for(h=0;h<size;h++){
		for(w=0;w<size;w++){
			for(c=0;c<channels;c++){
				temp = *(input+c*(size*size)+size*h+w);
				*(output+count)=temp;
				count++;
			}
		}
	}
}

void dropout (float* tensor, int input_size, float rate){
	int i;
	for(i=0;i<input_size;i++){
		*(tensor+i) *= rate;
	}
}

int argmax(float *tensor, int length){
	int i,max_idx = 0;
	float max = 0;
	for(i=0;i<length;i++){
		if(*(tensor+i) > max){
			max = *(tensor+i);
			max_idx = i;
		}
	}
	return max_idx;
}

void affine_layer(float* output, float* weights, float* input, float*bias,
			 int input_size, int output_size){
	matmul(weights, input, output, input_size, output_size);
	int i;
	for (i=0;i<output_size;i++){
		*(output+i) += *(bias+i);
	}
}

void affine_relu_layer(float* output, float* weights, float* input, float*bias,
			 int input_size, int output_size){
	matmul(weights, input, output, input_size, output_size);
	int i;
	for (i=0;i<output_size;i++){
		*(output+i) += *(bias+i);
		if(*(output+i) < 0){ //ReLU
			*(output+i) = 0;
		}
	}
}


void conv_relu_layer (float* output, float* kernels, float* input, float* bias, 
			 int kernel_h,int kernel_w, int input_size, int input_channels, 
			 int output_channels, int stride,int is_same){
	int output_size;
	if (!is_same){
		output_size = (int)(input_size-kernel_w)/stride+1;
		//TODO for now, ignore since there is no non-uniform filter like that
	} else {
		output_size = input_size;
	}
	conv_2d(output,kernels,input,bias,kernel_h,kernel_w,input_size,
			input_channels,output_channels,stride,is_same);
	relu(output,output_size,output_channels);
	return;
}


void softmax(float *tensor, int length){ //can also be done in-place
	float sum = 0;
	float temp = 0;
	int i;
	for(i=0;i<length;i++){	
		temp = exp(*(tensor+i));
		*(tensor+i) = temp;
		sum += temp;
	}
	for(i=0;i<length;i++){
		*(tensor+i) /= sum;
	}
}


void copy(float *input, float *output, int size, int channels){
	if(input == NULL) return;
	int c,h,w;
	for (c=0;c<channels;c++){
		for(h=0;h<size;h++){
			for(w=0;w<size;w++){
				*(output+c*(size*size)+h*size+w) = *(input+c*(size*size)+h*size+w);
			}
		}
	}
}


void concat(float *input_a, float *input_b, float *input_c, float *input_d, float* output,
			int size, int channel_a, int channel_b, int channel_c, int channel_d){
	copy(input_a,output,size,channel_a);
	output += channel_a*size*size;
	copy(input_b,output,size,channel_b);
	output += channel_b*size*size;
	copy(input_c,output,size,channel_c);
	output += channel_c*size*size;
	copy(input_d,output,size,channel_d);
}

void max_pool(float* input, float*output,int input_size ,int size, int channels, int stride){
	int c,out_h,out_w;
	int h,w;
	int in_h,in_w;
	int output_size;
	float local_max,temp;

	//Assume max_pool is always valid
	output_size = (input_size-size)/stride + 1; //Might need error handling
	for(c=0;c<channels;c++){ //fro each channel
		for(out_h=0;out_h<output_size;out_h++){ //for each column
			for(out_w=0;out_w<output_size;out_w++){ //for each row
				local_max = 0; //set the max to 0
				for (h=0;h<size;h++){ //loop through all elements in the window
					for(w=0;w<size;w++){
						in_h = stride*out_h+h;
						in_w = stride*out_w+w;
						if((in_h<0)||(in_h>=input_size)||(in_w<0)||(in_w>=input_size)){
							continue; //skip this loop, nothing to pick here
						} else {
							temp = *(input+c*(input_size*input_size)+in_h*input_size+in_w);
						}
						if (temp > local_max) {
							local_max = temp;
						}
					}
				}
				*(output+c*(output_size*output_size)+out_h*output_size+out_w) = local_max;
			}
		}
	}
}

void relu(float* tensor,int size, int channels){
	int h,w,c;
	float temp;
	for(c=0;c<channels;c++){
		for(h=0;h<size;h++){
			for(w=0;w<size;w++){
				temp =*(tensor+c*(size*size)+h*size+w);
				if (temp < 0){
					*(tensor+c*(size*size)+h*size+w) = 0;
				}
			}
		}
	}
}


float sum (float* input, int size) {
	int i;
	float ret = 0;
	for (i=0;i<size;i++) {
		ret += input[i];
	}
	return ret;
}

/* Matrix multiplcation experiment
 * input should be size of W
 * output should be size of H
 */
void matmul(float* weight, float* input, float* output, int input_size, int output_size){
	int i,j;
	for (i = 0;i < output_size;i ++){ //row
		for (j = 0;j < input_size;j ++){ //column
			*(output+i) += (*(input+j)) * (*(weight+input_size*i+j));
		}
	}
}

/* Convolution
 *
 *
 */

void conv_2d(float* output, float* kernels, float* input, float* bias, 
			 int kernel_h,int kernel_w, int input_size, int input_channels, 
			 int output_channels, int stride,int is_same){
	float temp,local_sum;
	int out_h,out_w,out_c;
	int in_h,in_w,in_c;
	int k_h,k_w;
	int output_size;
	int half_h,half_w;
	int init_h,init_w = 0; //the position of the upper-left-most patch
					  //depending on whether it is VALID or SAME
	float* kernel;

	/************************calculate utility variables********/
	half_h = (kernel_h-1)/2;
	half_w = (kernel_w-1)/2;
	if(is_same){
		output_size = input_size;
		init_h = 0;
		init_w = 0;
	} else {
		if (kernel_w != kernel_h) {
			printf("Cannot do VALID with asym filter\n");
			return;
		}
		output_size = (int)(input_size-kernel_h)/stride+1;
		init_h = half_h;
		init_w = half_w;
	}

	if (bias == NULL){
		bias = get_tensor(1,output_channels);
	}

	/**************main loop body****************************/
	for(out_c=0;out_c<output_channels;out_c++){ //for each output channel
		kernel = kernels+out_c*(kernel_h*kernel_w*input_channels);
		for(out_h=0;out_h<output_size;out_h++){ //for each row
			for(out_w=0;out_w<output_size;out_w++){ //for each element
				local_sum = 0;
				for(in_c=0;in_c<input_channels;in_c++){ //for each input channel
					for(k_h=-half_h;k_h<half_h+1;k_h++){ //e.g. for kernel size 7, go from -3 to 3
						for(k_w=-half_w;k_w<half_w+1;k_w++){
							in_h = init_h+out_h*stride+k_h;
							in_w = init_w+out_w*stride+k_w;
							if((in_h<0)||(in_h>=input_size)||(in_w<0)||(in_w>=input_size)){
								temp = 0;
							} else {
								temp = *(input+in_c*(input_size*input_size)+in_h*input_size+in_w);
							}
							local_sum += (*(kernel+in_c*(kernel_h*kernel_w)+(k_h+half_h)*kernel_w+(k_w+half_w)))*temp;
						}
					}
				}
				*(output+out_c*(output_size*output_size)+out_h*output_size+out_w) = local_sum + *(bias + out_c);
			}
		}
	}
}



/* Convolution_Original
 *	Doesn't support asymmetric filter
 *
 */
/*
void conv_2d(float* output, float* kernels, float* input, float* bias, 
			 int kernel_size, int input_size, int input_channels, 
			 int output_channels, int stride,int is_same){
	float temp,local_sum;
	int out_h,out_w,out_c;
	int in_h,in_w,in_c;
	int k_h,k_w;
	int output_size;
	int half_kernel;
	int init_pos = 0; //the position of the upper-left-most patch
					  //depending on whether it is VALID or SAME
	float* kernel;

	if(kernel_size%2 != 1){
		printf("invalid kernel size!\n");
		return;
	}

	half_kernel = (kernel_size-1)/2;
	if(is_same){
		output_size = input_size;
		init_pos = 0;
	} else {
		output_size = (int)(input_size-kernel_size)/stride+1;
		init_pos = half_kernel;
	}

	for(out_c=0;out_c<output_channels;out_c++){ //for each output channel
		kernel = kernels+out_c*(kernel_size*kernel_size*input_channels);
		for(out_h=0;out_h<output_size;out_h++){ //for each row
			for(out_w=0;out_w<output_size;out_w++){ //for each element
				local_sum = 0;
				for(in_c=0;in_c<input_channels;in_c++){ //for each input channel
					for(k_h=-half_kernel;k_h<half_kernel+1;k_h++){ //e.g. for kernel size 7, go from -3 to 3
						for(k_w=-half_kernel;k_w<half_kernel+1;k_w++){
							in_h = init_pos+out_h*stride+k_h;
							in_w = init_pos+out_w*stride+k_w;
							if((in_h<0)||(in_h>=input_size)||(in_w<0)||(in_w>=input_size)){
								temp = 0;
							} else {
								temp = *(input+in_c*(input_size*input_size)+in_h*input_size+in_w);
							}
							local_sum += (*(kernel+in_c*(kernel_size*kernel_size)+(k_h+half_kernel)*kernel_size+(k_w+half_kernel)))*temp;
						}
					}
				}
				*(output+out_c*(output_size*output_size)+out_h*output_size+out_w) = local_sum + *(bias + out_c);
			}
		}
	}
}
*/


void show_tensor(float* mat,int size_z, int size_y, int size_x){
	int x,y,z;
	for(z=0;z<size_z;z++){
		printf("channel #%d\n", z);
		for(y=0;y<size_y;y++){
			for(x=0;x<size_x;x++){
				printf("%0.8f ",*(mat+z*(size_y*size_x)+y*size_x+x));
			}
			printf("\n");
		}
		printf("\n");
	}
}

float* get_tensor(int size, int channels){
	return (float*)calloc(size*size*channels,sizeof(float));
}

float* get_kernel(int size, int channels_in, int channels_out){
	return (float*)calloc(size*size*channels_in*channels_out,sizeof(float));
}

float* get_weight(int input, int output){
	return (float*)calloc(input*output,sizeof(float));
}

void destroy_tensor(float* tensor){
	return free(tensor);
}

int read_data(char* path, float* array, int size){
	FILE *fp;
	fp = fopen(path,"rb");
	//printf("read: %s\n", path);
	if(fp == NULL) printf("FAIL!!!!\n");
	int ret;
	ret = fread(array,sizeof(float),size,fp);
	//printf("read in %d floats\n", ret);
	fclose(fp);
	return ret;
}


char* search_name(char* path, int index){
	FILE *fp;
	fp = fopen(path,"rb");
	if(fp == NULL) printf("FAIL!!!!\n");
	int i;
	char* ret;

	size_t max_length = 64;
	char* buffer = (char*) malloc(sizeof(char)*(64+1));
	for(i=0;i<index;i++){ //since index start from 0
		ret = fgets(buffer,max_length,fp);
		if(ret == NULL){
			printf("index out of bound!\n");
			break;
		}
	}
	fclose(fp);
	return buffer;
}