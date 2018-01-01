#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "myTensorFlow.h"

void unit_test(void);
//input size expected to be 35*35*384
void block_inception_a(char* path, float* input, float* output);
void block_reduction_a(char* path, float* input, float* output);
void block_inception_b(char* path, float* input, float* output);
void block_reduction_b(char* path, float* input, float* output);
void block_inception_c(char* path, float* input, float* output);
void block_stem_a(char* path, float* input, float* output);
void block_stem_b(char* path, float* input, float* output);
void block_stem_c(char* path, float* input, float* output);

//run once
void* workerThreadStart(void* threadArgs);

typedef struct {
	int start_index;
	int end_index;
    int threadId;
    int numThreads;
} WorkerArgs;

int main (int argc, char* argv[]){
	const static int MAX_THREADS = 32;
	int numThreads = 1;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        args[i].numThreads = numThreads;
        args[i].threadId = i;
    }

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}

void* workerThreadStart(void* threadArgs) {

	char path[128];
	strcpy(path,"./image.bin");
	float* img = get_tensor(299,3);
	read_data(path,img,299*299*3);

	//simply take the largest needed
	float* inter_a = get_tensor(147,64);
	float* inter_b = get_tensor(147,64);

	//largest size is 
	float* kernel = get_kernel(3,32,64); //the size is actually larger than needed
	float* var = get_tensor(1,64);
	float* mean = get_tensor(1,64);
	float* beta = get_tensor(1,64);

	int root_len;
	float* input;
	float* output;
	float* swp;
	int i = 0;

	//299*299*3
	strcpy(path,"../model/InceptionV4_Conv2d_1a_3x3_weights.bin");
	read_data(path,kernel,3*3*3*32);
	strcpy(path,"../model/InceptionV4_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path,mean,32);
	strcpy(path,"../model/InceptionV4_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path,var,32);
	strcpy(path,"../model/InceptionV4_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path,beta,32);
	conv_relu_BN_layer (inter_a, kernel, img, beta, mean, var, 
			 3, 3, 299, 3, 32, 2, FALSE);

	//149*149*32
	strcpy(path,"../model/InceptionV4_Conv2d_2a_3x3_weights.bin");
	read_data(path,kernel,3*3*32*32);
	strcpy(path,"../model/InceptionV4_Conv2d_2a_3x3_BatchNorm_moving_mean.bin");
	read_data(path,mean,32);
	strcpy(path,"../model/InceptionV4_Conv2d_2a_3x3_BatchNorm_moving_variance.bin");
	read_data(path,var,32);
	strcpy(path,"../model/InceptionV4_Conv2d_2a_3x3_BatchNorm_beta.bin");
	read_data(path,beta,32);
	conv_relu_BN_layer (inter_b, kernel, inter_a, beta, mean, var, 
			 3, 3, 149, 32, 32, 1, FALSE);

	//147*147*32
	strcpy(path,"../model/InceptionV4_Conv2d_2b_3x3_weights.bin");
	read_data(path,kernel,3*3*32*64);
	strcpy(path,"../model/InceptionV4_Conv2d_2b_3x3_BatchNorm_moving_mean.bin");
	read_data(path,mean,64);
	strcpy(path,"../model/InceptionV4_Conv2d_2b_3x3_BatchNorm_moving_variance.bin");
	read_data(path,var,64);
	strcpy(path,"../model/InceptionV4_Conv2d_2b_3x3_BatchNorm_beta.bin");
	read_data(path,beta,64);
	conv_relu_BN_layer (inter_a, kernel, inter_b, beta, mean, var, 
			 3, 3, 147, 32, 64, 1, TRUE);

	strcpy(path,"../model/InceptionV4_Mixed_3a");
	block_stem_a(path, inter_a, inter_b);
	strcpy(path,"../model/InceptionV4_Mixed_4a");
	block_stem_b(path, inter_b, inter_a);
	strcpy(path,"../model/InceptionV4_Mixed_5a");
	block_stem_c(path, inter_a, inter_b);
	
	//4x Inception Block A 
	strcpy(path,"../model/InceptionV4_Mixed_5b");
	input = inter_b;
	output = inter_a;
	root_len = strlen(path);
	for(i = 0;i<4;i++){
		block_inception_a(path,(float*)input,(float*)output);
		swp = input;
		input = output;
		output = swp;
		path[root_len-1] ++;
	}

	//It turns out that the final output is stored in inter_b

	strcpy(path,"../model/InceptionV4_Mixed_6a");
	block_reduction_a(path,inter_b,inter_a);
	
	strcpy(path,"../model/InceptionV4_Mixed_6b");
	input = inter_a;
	output = inter_b;
	root_len = strlen(path);
	for(i = 0;i<7;i++){
		block_inception_b(path,(float*)input,(float*)output);
		swp = input;
		input = output;
		output = swp;
		path[root_len-1] ++;
	}

	//it turns out that the result is stored in inter_b

	strcpy(path,"../model/InceptionV4_Mixed_7a");
	block_reduction_b(path,inter_b,inter_a);

	strcpy(path,"../model/InceptionV4_Mixed_7b");
	input = inter_a;
	output = inter_b;
	root_len = strlen(path);
	for(i = 0;i<3;i++){
		block_inception_c(path,(float*)input,(float*)output);
		swp = input;
		input = output;
		output = swp;
		path[root_len-1] ++;
	}

	float* pre_fc = get_tensor(1,1536);
	global_avg_pool(input, pre_fc, 8, 1536);

	float* fc_w = get_weight(1001,1536);
	float* fc_b = get_tensor(1,1001);
	float* fc_o = get_tensor(1,1001);
	read_data("../model/InceptionV4_Logits_Logits_weights.bin",fc_w,1001*1536);
	read_data("../model/InceptionV4_Logits_Logits_biases.bin",fc_b,1001);
	affine_layer(fc_o, fc_w, pre_fc, fc_b, 1536, 1001);
	
	softmax(fc_o, 1001);
	top_5_hit (fc_o, 0);
	int class = argmax(fc_o, 1001);
	char* final = search_name("./imagenet_classes.txt", class);
	printf("done!\n");
	printf("classified as %s\n", final);

	return EXIT_SUCCESS;
}

void unit_test(void){
	//int i,j,k; //can be used anywhere
/*

	float input[2][5][5] = {{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}},{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}}};
	float output[2][5][5] = {0};

	show_tensor((float*)input,2,5,5);

	avg_pool((float*)input, (float*)output, 5 ,3, 2, 1);

	show_tensor((float*)output,2,5,5);

	char path[128];
	strcpy(path,"../model/InceptionV4_Mixed_5b");
	int root_len = strlen(path);
	float buf_a[384][35][35] = {0};
	float buf_b[384][35][35] = {0};
	
	float* input;
	float* output;
	float* swp;
	read_data("./Mixed_5a.bin",(float*)buf_a,35*35*384);
	input = (float*)buf_a;
	output = (float*)buf_b;
	for(int i;i<4;i++){
		block_inception_a(path,(float*)input,(float*)output);
		swp = input;
		input = output;
		output = swp;
		path[root_len-1] ++;
		printf("%s\n", path);
		show_tensor((float*)input,1,1,35);
	}*/
	char path[128];
	strcpy(path,"../model/InceptionV4_Mixed_7b");
	float buf_a[1536][8][8] = {0};
	float buf_b[1536][8][8] = {0};
	read_data("./Mixed_7a.bin",(float*)buf_a,8*8*1536);
	block_inception_c(path,(float*)buf_a,(float*)buf_b);
	show_tensor((float*)buf_b+8*8*512,1,8,8);
	show_tensor((float*)buf_b+8*8*1024,1,8,8);
	show_tensor((float*)buf_b+8*8*1500,1,8,8);
	//strcpy(path,"../model/InceptionV4_Mixed_5c");
	//block_inception_a(path,(float*)buf_b,(float*)buf_a);
	
	//block_inception_a(path,(float*)buf_a,(float*)buf_b);
	//block_inception_a(path,(float*)buf_a,(float*)buf_b);
	//show_tensor((float*)input,1,1,35);

}

void block_stem_a(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(3,64,96); //the size is actually larger than needed
	float* var = get_tensor(1,96);
	float* mean = get_tensor(1,96);
	float* beta = get_tensor(1,96);
	float* branch_0 = get_tensor(73,64);
	float* branch_1 = get_tensor(73,96);

	//Branch 0
	max_pool(input, branch_0, 147 ,3, 64, 2);

	//Branch 1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*320*320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,320);
	conv_relu_BN_layer (branch_1, kernel, input, beta, mean, var, 
			 3, 3, 147, 64, 96, 2, FALSE);

	concat(branch_0,branch_1,NULL,NULL,output,73,64,96,0,0);

	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1);
}


void block_stem_b(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(3,64,96); //the size is actually larger than needed
	float* var = get_tensor(1,96);
	float* mean = get_tensor(1,96);
	float* beta = get_tensor(1,96);
	float* branch_0a = get_tensor(73,96);
	float* branch_0b = get_tensor(73,96);
	float* branch_1a = get_tensor(73,96);
	float* branch_1b = get_tensor(73,96);

	//Branch 0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*160*64);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_0a, kernel, input, beta, mean, var, 
			 1,1, 73, 160, 64, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*64*96);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_0b, kernel, branch_0a, beta, mean, var, 
			 3, 3, 73, 64, 96, 1,FALSE);

	//branch 1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*160*64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_1a, kernel, input, beta, mean, var, 
			 1,1, 73, 160, 64, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_weights.bin");
	read_data(path_buf,kernel,1*7*64*64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 1, 7, 73, 64, 64, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_weights.bin");
	read_data(path_buf,kernel,7*1*64*64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_1a, kernel, branch_1b, beta, mean, var, 
			 7, 1, 73, 64, 64, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*320*320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,320);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 3, 3, 73, 64, 96, 1, FALSE);

	concat(branch_0b,branch_1b,NULL,NULL,output,71,96,96,0,0);

	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0a);
	destroy_tensor(branch_0b);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
}


void block_stem_c(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(3,192,192); //the size is actually larger than needed
	float* var = get_tensor(1,192);
	float* mean = get_tensor(1,192);
	float* beta = get_tensor(1,192);
	float* branch_0 = get_tensor(35,192);
	float* branch_1 = get_tensor(35,192);

	//Branch 0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*192*192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_0, kernel, input, beta, mean, var, 
			 3, 3, 71, 192, 192, 2, FALSE);

	//Branch 1
	max_pool(input, branch_1, 71, 3, 192, 2);

	concat(branch_0,branch_1,NULL,NULL,output,35,192,192,0,0);

	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1);
}

void block_inception_c(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(2,448,512); //the size is actually larger than needed
	float* var = get_tensor(1,512);
	float* mean = get_tensor(1,512);
	float* beta = get_tensor(1,512);
	float* branch_0 = get_tensor(8,256);
	float* branch_1 = get_tensor(8,512);
	float* branch_1a = get_tensor(8,256);
	float* branch_1b = get_tensor(8,256);
	float* branch_2 = get_tensor(8,512);
	float* branch_2a = get_tensor(8,512);
	float* branch_2b = get_tensor(8,256);
	float* branch_3a = get_tensor(8,1536);
	float* branch_3b = get_tensor(8,256);

	//Branch0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*1536*256);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_0, kernel, input, beta, mean, var, 
			 1,1, 8, 1536, 256, 1,TRUE);

	//Branch1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*1536*384);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,384);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,384);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,384);
	conv_relu_BN_layer (branch_1, kernel, input, beta, mean, var, 
			 1,1, 8, 1536, 384, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x3_weights.bin");
	read_data(path_buf,kernel,1*3*384*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1a, kernel, branch_1, beta, mean, var, 
			 1, 3, 8, 384, 256, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_3x1_weights.bin");
	read_data(path_buf,kernel,3*1*384*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_3x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1b, kernel, branch_1, beta, mean, var, 
			 3, 1, 8, 384, 256, 1, TRUE);

	concat(branch_1a,branch_1b,NULL,NULL,branch_1,8,256,256,0,0);

	//Branch2
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*1536*384);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,384);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,384);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,384);
	conv_relu_BN_layer (branch_2, kernel, input, beta, mean, var, 
			 1,1, 8, 1536, 384, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x1_weights.bin");
	read_data(path_buf,kernel,3*1*384*448);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,448);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,448);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,448);
	conv_relu_BN_layer (branch_2a, kernel, branch_2, beta, mean, var, 
			 3, 1, 8, 384, 448, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x3_weights.bin");
	read_data(path_buf,kernel,3*1*448*512);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,512);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,512);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,512);
	conv_relu_BN_layer (branch_2, kernel, branch_2a, beta, mean, var, 
			 1, 3, 8, 448, 512, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_1x3_weights.bin");
	read_data(path_buf,kernel,1*3*512*256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_1x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_1x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_1x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_2a, kernel, branch_2, beta, mean, var, 
			 1, 3, 8, 512, 256, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_3x1_weights.bin");
	read_data(path_buf,kernel,3*1*512*256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_3x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_3x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_3x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_2b, kernel, branch_2, beta, mean, var, 
			 3, 1, 8, 512, 256, 1, TRUE);

	concat(branch_2a,branch_2b,NULL,NULL,branch_2,8,256,256,0,0);

	//Branch_3
	avg_pool(input, branch_3a, 8 ,3, 1536, 1);

	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_weights.bin");
	read_data(path_buf,kernel,1536*256);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_3b, kernel, branch_3a, beta, mean, var,  
			 1,1, 8, 1536, 256, 1, TRUE);

	//concatenation
	concat(branch_0, branch_1, branch_2, branch_3b, output,
			8, 256, 512, 512, 256);

	//clean up
	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1);
	destroy_tensor(branch_2);
	destroy_tensor(branch_2a);
	destroy_tensor(branch_2b);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
	destroy_tensor(branch_3a);
	destroy_tensor(branch_3b);
}

void block_reduction_b(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(3,320,320); //the size is actually larger than needed
	float* var = get_tensor(1,320);
	float* mean = get_tensor(1,320);
	float* beta = get_tensor(1,320);
	float* branch_0a = get_tensor(17,192);
	float* branch_0b = get_tensor(8,192);
	float* branch_1a = get_tensor(17,320);
	float* branch_1b = get_tensor(17,320);
	float* branch_2 = get_tensor(8,1024);

	//Branch 0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*192*1024);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_0a, kernel, input, beta, mean, var, 
			 1,1, 17, 1024, 192, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*192*192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_0b, kernel, branch_0a, beta, mean, var, 
			 3, 3, 17, 192, 192, 2, FALSE);
	
	//Branch 1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*1024*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1a, kernel, input, beta, mean, var, 
			 1,1, 17, 1024, 256, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_weights.bin");
	read_data(path_buf,kernel,1*7*256*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 1, 7, 17, 256, 256, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_weights.bin");
	read_data(path_buf,kernel,7*1*256*320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,320);
	conv_relu_BN_layer (branch_1a, kernel, branch_1b, beta, mean, var, 
			 7, 1, 17, 256, 320, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*320*320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,320);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,320);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 3,3, 17, 320, 320, 2, FALSE);

	//Branch 2
	max_pool(input, branch_2, 17 ,3, 1024, 2);

	//concatenation
	concat(branch_0b, branch_1b, branch_2, NULL, output,
			8, 192, 320, 1024, 0);

	//clean up
	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0a);
	destroy_tensor(branch_0b);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
	destroy_tensor(branch_2);
}

void block_inception_b(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 
	float* kernel = get_kernel(3,224,256); //the size is actually larger than needed
	float* var = get_tensor(1,384);
	float* mean = get_tensor(1,384);
	float* beta = get_tensor(1,384);
	float* branch_0 = get_tensor(17,384);
	float* branch_1a = get_tensor(17,256);
	float* branch_1b = get_tensor(17,256);
	float* branch_2a = get_tensor(17,256);
	float* branch_2b = get_tensor(17,256);
	float* branch_3a = get_tensor(17,1024);
	float* branch_3b = get_tensor(17,128);

	//branch 0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1024*384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,384);
	conv_relu_BN_layer (branch_0, kernel, input, beta, mean, var, 
			 1,1, 17, 1024, 384, 1,TRUE);

	//branch 1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1024*192);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_1a, kernel, input, beta, mean, var, 
			 1,1, 17, 1024, 192, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_weights.bin");
	read_data(path_buf,kernel,1*7*192*224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_1x7_BatchNorm_beta.bin");
	read_data(path_buf,beta,224);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 1, 7, 17, 192, 224, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_weights.bin");
	read_data(path_buf,kernel,7*1*224*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0c_7x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1a, kernel, branch_1b, beta, mean, var, 
			 7, 1, 17, 224, 256, 1, TRUE);

	//branch 2
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1024*192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_2a, kernel, input, beta, mean, var, 
			 1,1, 17, 1024, 192, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_7x1_weights.bin");
	read_data(path_buf,kernel,7*1*192*192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_7x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_2b, kernel, branch_2a, beta, mean, var, 
			 7, 1, 17, 192, 192, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x7_weights.bin");
	read_data(path_buf,kernel,1*7*192*224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_1x7_BatchNorm_beta.bin");
	read_data(path_buf,beta,224);
	conv_relu_BN_layer (branch_2a, kernel, branch_2b, beta, mean, var, 
			 1, 7, 17, 192, 224, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_7x1_weights.bin");
	read_data(path_buf,kernel,7*1*224*224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,224);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0d_7x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,224);
	conv_relu_BN_layer (branch_2b, kernel, branch_2a, beta, mean, var, 
			 7, 1, 17, 224, 224, 1, TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_1x7_weights.bin");
	read_data(path_buf,kernel,1*7*224*256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0e_1x7_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_2a, kernel, branch_2b, beta, mean, var, 
			 1, 7, 17, 224, 256, 1, TRUE);

	//branch 3
	avg_pool(input, branch_3a, 17 ,3, 1024, 1);

	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_weights.bin");
	read_data(path_buf,kernel,1024*128);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,128);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,128);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,128);
	conv_relu_BN_layer (branch_3b, kernel, branch_3a, beta, mean, var,  
			 1,1, 17, 1024, 128, 1, TRUE);


	//concatenation
	concat(branch_0, branch_1a, branch_2a, branch_3b, output,
			17, 384, 256, 256, 128);

	//clean up
	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
	destroy_tensor(branch_2a);
	destroy_tensor(branch_2b);
	destroy_tensor(branch_3a);
	destroy_tensor(branch_3b);
}

void block_reduction_a(char* path, float* input, float* output){
	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);
	
	//largest size is 96*384*3*3
	float* kernel = get_kernel(3,384,384);
	float* var = get_tensor(1,384);
	float* mean = get_tensor(1,384);
	float* beta = get_tensor(1,384);
	float* branch_0 = get_tensor(17,384);
	float* branch_1a = get_tensor(35,224);
	float* branch_1b = get_tensor(35,224);
	float* branch_2 = get_tensor(17,384);

	//Branch 0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*384*384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,384);
	conv_relu_BN_layer (branch_0, kernel, input, beta, mean, var, 
			 3,3, 35, 384, 384, 2,FALSE);
	
	//Branch 1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,1*1*192*384);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,192);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,192);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,192);
	conv_relu_BN_layer (branch_1a, kernel, input, beta, mean, var, 
			 1,1, 35, 384, 192, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*192*224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,224);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,224);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var, 
			 3,3, 35, 192, 224, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*224*256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,256);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,256);
	conv_relu_BN_layer (branch_1a, kernel, branch_1b, beta, mean, var, 
			 3,3, 35, 224, 256, 2, FALSE);

	//Branch 2
	max_pool(input, branch_2, 35 ,3, 384, 2);

		//concatenation
	concat(branch_0, branch_1a, branch_2, NULL, output,
			17, 384, 256, 384, 0);

	//clean up
	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
	destroy_tensor(branch_2);
}


//input size expected to be 35*35*384
void block_inception_a(char* path, float* input, float* output){

	char path_buf[256] = {0};
	strcpy(path_buf,path);
	int root_len = strlen(path);

	//largest size is 96*384*3*3
	float* kernel = get_kernel(3,384,96);
	float* var = get_tensor(1,96);
	float* mean = get_tensor(1,96);
	float* beta = get_tensor(1,96);
	float* branch_0 = get_tensor(35,96);
	float* branch_1a = get_tensor(35,96);
	float* branch_1b = get_tensor(35,96);
	float* branch_2a = get_tensor(35,96);
	float* branch_2b = get_tensor(35,96);
	float* branch_3a = get_tensor(35,384);
	float* branch_3b = get_tensor(35,96);

	//branch0
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,96*384);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_0, kernel, input, beta, mean, var, 
			 1,1, 35, 384, 96, 1,TRUE);

	//branch1
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,64*384);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_1a, kernel, input, beta, mean, var, 
			 1,1, 35, 384, 64, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*64*96);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_1_Conv2d_0b_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_1b, kernel, branch_1a, beta, mean, var,  
			 3,3, 35, 64, 96, 1,TRUE);

	//branch2
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_weights.bin");
	read_data(path_buf,kernel,64*384);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,64);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,64);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,64);
	conv_relu_BN_layer (branch_2a, kernel, input, beta, mean, var,  
			 1,1, 35, 384, 64, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*64*96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0b_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_2b, kernel, branch_2a, beta, mean, var,  
			 3,3, 35, 64, 96, 1,TRUE);

	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_3x3_weights.bin");
	read_data(path_buf,kernel,3*3*96*96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_2_Conv2d_0c_3x3_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_2a, kernel, branch_2b, beta, mean, var,  
			 3,3, 35, 96, 96, 1,TRUE);

	//branch_3
	avg_pool(input, branch_3a, 35 ,3, 384, 1);

	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_weights.bin");
	read_data(path_buf,kernel,96*384);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.bin");
	read_data(path_buf,mean,96);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.bin");
	read_data(path_buf,var,96);
	strcpy(path_buf+root_len,"_Branch_3_Conv2d_0b_1x1_BatchNorm_beta.bin");
	read_data(path_buf,beta,96);
	conv_relu_BN_layer (branch_3b, kernel, branch_3a, beta, mean, var,  
			 1,1, 35, 384, 96, 1, TRUE);

	//concatenation
	concat(branch_0, branch_1b, branch_2a, branch_3b, output,
			35, 96, 96, 96, 96);

	//clean up
	destroy_tensor(kernel);
	destroy_tensor(var);
	destroy_tensor(mean);
	destroy_tensor(beta);
	destroy_tensor(branch_0);
	destroy_tensor(branch_1a);
	destroy_tensor(branch_1b);
	destroy_tensor(branch_2a);
	destroy_tensor(branch_2b);
	destroy_tensor(branch_3a);
	destroy_tensor(branch_3b);
}