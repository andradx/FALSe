/*
 * ldpc.cpp
 *
 *  Created on: Oct 17, 2014
 *      Author: andrade
 */

#include "ldpc.h"


namespace ecc {

ldpc::ldpc() {
	N = 0;
	M = 0;
	K = N - M;
	BNW = NULL;
	CNW = NULL;
	BNAdjacency = NULL;
	CNAdjacency = NULL;
	h_ligx = d_ligx = NULL;
	h_ligf = d_ligf =NULL;
	h_gamma = d_gamma = NULL;
	h_beta = d_beta = NULL;
	h_alpha = d_alpha = NULL;
	h_word = d_word = NULL;
	h = NULL;
	rng_state = NULL;

}

ldpc::~ldpc() {

	// host cleanup
	if(BNW != NULL) cudaFreeHost(BNW);
	if(CNW != NULL) cudaFreeHost(CNW);
	if(BNAdjacency != NULL) cudaFreeHost(BNAdjacency);
	if(CNAdjacency != NULL) cudaFreeHost(CNAdjacency);
	if(h_ligx != NULL) cudaFreeHost(h_ligx);
	if(h_ligf != NULL) cudaFreeHost(h_ligf);
	if(h_gamma != NULL) cudaFreeHost(h_ligf);
	if(h_beta != NULL) cudaFreeHost(h_ligf);
	if(h_alpha != NULL) cudaFreeHost(h_ligf);
	checkCUDAError("Error freeing buffers on host");

	// device cleanup
	if(d_ligx != NULL) 	cudaFree(d_ligx);
	if(d_ligf != NULL) 	cudaFree(d_ligf);
	if(d_gamma != NULL) cudaFree(d_gamma);
	if(d_beta != NULL) 	cudaFree(d_beta);
	if(d_alpha != NULL) cudaFree(d_alpha);
	if(rng_state != NULL) cudaFree(rng_state);
	checkCUDAError("Error freeing buffers on device");

	cudaDeviceReset();
}


ldpc::ldpc(const char * alist_file){

	FILE *fd;
	int i, j;
	if((fd= fopen(alist_file,"r")) == NULL)
		printf("Can't open this ldpc file\n");

	fscanf(fd,"%d %d\n",&N,&M);																// read N and M
	fscanf(fd,"%d %d\n",&MaxBNW,&MaxCNW);													// read MaxBNW and MaxCNW
	K = N - M;																				// compute K
	h = NULL;																				// NULL to h
	rng_state = NULL;																		// NULL to rng_state

	cudaHostAlloc((void**)&BNW,sizeof(int)*N,cudaHostAllocWriteCombined);					// allocate space for BNW
	cudaHostAlloc((void**)&CNW,sizeof(int)*M,cudaHostAllocWriteCombined);					// allocate space for CNW

	cudaHostAlloc((void**)&BNAdjacency,sizeof(int)*N*MaxBNW,cudaHostAllocWriteCombined);	// allocate space for BNAdjacency
	cudaHostAlloc((void**)&CNAdjacency,sizeof(int)*M*MaxCNW,cudaHostAllocWriteCombined);	// allocate space for CNAdjacency

	cudaHostAlloc((void**)&h_ligx,sizeof(int)*N*MaxBNW,cudaHostAllocWriteCombined);			// allocate space for h_ligx
	cudaHostAlloc((void**)&h_ligf,sizeof(int)*M*MaxCNW,cudaHostAllocWriteCombined);			// allocate space for h_ligf


	memset((void*)BNW, -1, N*sizeof(int));
	memset((void*)CNW, -1, M*sizeof(int));
	memset((void*)BNAdjacency, -1, sizeof(int)*N*MaxBNW);
	memset((void*)CNAdjacency, -1, sizeof(int)*M*MaxCNW);
	memset((void*)h_ligx, -1, sizeof(int)*N*MaxBNW);
	memset((void*)h_ligf, -1, sizeof(int)*M*MaxCNW);


	for(i = 0; i < N; i++)																	// read BNW
		fscanf(fd,"%d",&BNW[i]);

	for(i = 0; i < M; i++)																	// read CNW
		fscanf(fd,"%d",&CNW[i]);

	for(i = 0; i < N; i++)																	// read BNAdjacency padded to the MaxBNW
		for(j = 0; j < BNW[i]; j++){
			fscanf(fd,"%d",&BNAdjacency[MaxBNW*i+j]);
			//printf("BNW: %d %d\n",i,j);
		}

	for(i = 0; i < M; i++)																	// read CNAdjacency padded to the MaxBNW
		for(j = 0; j < CNW[i]; j++){
			fscanf(fd,"%d",&CNAdjacency[MaxCNW*i+j]);
			//printf("CNW: %d %d\n",i,j);
		}

	fclose(fd);
	generate_ligVectors();
}

void ldpc::write_alistFile(const char *alist_file){

	FILE *fd;
	int i, j;
	if((fd= fopen(alist_file,"w")) == NULL)
		printf("Can't open this ldpc file for writing\n");

	fprintf(fd,"%d %d\n",N,M);					// read N and M
	fprintf(fd,"%d %d\n",MaxBNW,MaxCNW);		// read MaxBNW and MaxCNW

	for(i = 0; i < N; i++)											// read BNW
		fprintf(fd,"%d ",BNW[i]);
	fprintf(fd,"\n");

	for(i = 0; i < M; i++)											// read CNW
		fprintf(fd,"%d ",CNW[i]);
	fprintf(fd,"\n");


	for(i = 0; i < N; i++){											// read BNAdjacency padded to the MaxBNW
		for(j = 0; j < BNW[i]; j++){
			fprintf(fd,"%d ",BNAdjacency[MaxBNW*i+j]);
			//printf("BNW: %d %d\n",i,j);
		}
		fprintf(fd,"\n");
	}

	for(i = 0; i < M; i++){											// read CNAdjacency padded to the MaxBNW
		for(j = 0; j < CNW[i]; j++){
			fprintf(fd,"%d ",CNAdjacency[MaxCNW*i+j]);
			//printf("CNW: %d %d\n",i,j);
		}
		fprintf(fd,"\n");
	}

	fclose(fd);

}


void ldpc::generate_PCM(){

	int i, j;

	cudaHostAlloc((void**)&h,sizeof(int*)*M*N,cudaHostAllocWriteCombined);						// allocate space for parity-check matrix
	memset((void*)h, 0, M*N*sizeof(int));

	for(i = 0; i < M; i++){
		for(j = 0; j < CNW[i]; j++){
			h[M*i+CNAdjacency[MaxCNW*i+j]-1] = 1;
		}
	}
}

void ldpc::generate_ligVectors(){

	int i, j;
	int idx = 0;

	for (i = 1; i <= M; i++)			//sweep the matrix row-wise to compute ligx
		for(j = 0; j < N*MaxBNW; j++)
			if(BNAdjacency[j] == i)
				h_ligx[j] = idx++;

	idx = 0;

	for (i = 1; i <= N; i++)			//sweep the matrix column-wise to compute ligx
		for(j = 0; j < M*MaxCNW; j++)
			if(CNAdjacency[j] == i)
				h_ligf[j] = idx++;

}

void ldpc::print_PCM(FILE *fd){

	int i, j;

	if(h == NULL) generate_PCM();

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++){
			fprintf(fd,"%d ",h[i*M+j]);
		}
		fprintf(fd,"\n");
	}
}


void ldpc::print_ligVectors(FILE *fd){

	int i;

	fprintf(fd,"ligx: ");
	for(i = 0; i < N*MaxBNW; i++){
		fprintf(fd,"%d ",h_ligx[i]);
	}
	fprintf(fd,"\n");
	fprintf(fd,"ligf: ");
	for(i = 0; i < N*MaxBNW; i++){
		fprintf(fd,"%d ",h_ligf[i]);
	}
	fprintf(fd,"\n");
}

void ldpc::init_device(int deviceID){

	cudaSetDevice(deviceID);
	checkCUDAError("Can't grab such device");
	allocate_decoderMemory();
}

void ldpc::allocate_decoderMemory(){

	cudaMalloc((void**)&d_gamma,N*sizeof(char4));
	cudaMalloc((void**)&d_alpha,MaxBNW*N*sizeof(char4));
	cudaMalloc((void**)&d_beta,MaxCNW*M*sizeof(char4));
	cudaMalloc((void**)&d_word,N*sizeof(char4));
	cudaMalloc((void**)&d_ligx,N*MaxBNW*sizeof(int));
	cudaMalloc((void**)&d_ligf,M*MaxCNW*sizeof(int));
	checkCUDAError("Allocating decoder memory on the device");

	cudaHostAlloc((void**)&h_gamma,N*sizeof(char4),cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&h_alpha,MaxBNW*N*sizeof(char4),cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&h_beta,MaxCNW*M*sizeof(char4),cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&h_word,N*sizeof(char4),cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&h_gamma,N*sizeof(char4),cudaHostAllocWriteCombined);
	checkCUDAError("Allocating decoder memory on the host");

}

void ldpc::copy_decoderMemory(cudaMemcpyKind kind){

	if(kind == cudaMemcpyHostToDevice){
		cudaMemcpy((void*)d_gamma,(void*)h_gamma,N*sizeof(char4),kind);
		cudaMemcpy((void*)d_alpha,(void*)h_alpha,MaxBNW*N*sizeof(char4),kind);
		cudaMemcpy((void*)d_beta,(void*)h_beta,MaxCNW*M*sizeof(char4),kind);
		cudaMemcpy((void*)d_word,(void*)h_word,N*sizeof(char),kind);
		cudaMemcpy((void*)d_ligx,(void*)h_ligx,N*MaxBNW*sizeof(int),kind);
		cudaMemcpy((void*)d_ligf,(void*)h_ligf,M*MaxCNW*sizeof(int),kind);
		checkCUDAError("Copying data cudaMemcpyHostToDevice");
	}
	else if(kind == cudaMemcpyDeviceToHost){
		cudaMemcpy((void*)h_gamma,(void*)d_gamma,N*sizeof(char4),kind);
		cudaMemcpy((void*)h_alpha,(void*)d_alpha,MaxBNW*N*sizeof(char4),kind);
		cudaMemcpy((void*)h_beta,(void*)d_beta,MaxCNW*M*sizeof(char4),kind);
		cudaMemcpy((void*)h_word,(void*)d_word,N*sizeof(char),kind);
		cudaMemcpy((void*)h_ligx,(void*)d_ligx,N*MaxBNW*sizeof(int),kind);
		cudaMemcpy((void*)h_ligf,(void*)d_ligf,M*MaxCNW*sizeof(int),kind);
		checkCUDAError("Copying data cudaMemcpyDeviceToHost");
	}

}

void ldpc::copy_data(cudaMemcpyKind kind){
	copy_decoderMemory(kind);
}

void ldpc::setup_rng(unsigned int seed){

	if(rng_state != NULL) return;		// if rng_state is defined no action is required

	int threads = 128;
	int blocks = roundUp(threads,N)/threads;

	cudaMalloc((void**)&rng_state,N*sizeof(curandState));
	setup_kernel<<<blocks,threads>>>(rng_state,seed,N);
	checkCUDAError("Can't set up the PRNG");

}



__global__ void setup_kernel(curandState *state,unsigned int seed,int Nlim)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	if(id<Nlim)
		curand_init(seed, id, 0, &state[id]);

}

__global__ void generate_noise(curandState *state,char4 *gamma,char4 *alpha, float Noise_stddev,char *tx_codeword,int Nlim)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float noise;
	char4 pack_gamma;
	char tx;
	char llr;

	if(id<Nlim){

		tx  = tx_codeword[id];

		curandState localState = state[id];



		for(int n = 0; n < WORDS; n++) {

			noise = curand_normal(&localState);

			noise *= Noise_stddev;

			noise += !(tx  & (1 << (WORDS-n-1))) ? 1 : -1;


			llr_bpsk(noise,Noise_stddev);

			llr = clip(noise,SCALE,OFFSET);

			if(n<0)
				pack_gamma.x = llr;
			else if(n<1)
				pack_gamma.y = llr;
			else if(n<2)
				pack_gamma.z = llr;
			else
				pack_gamma.w = llr;



			state[id]=localState;

			BroadCastBits(pack_gamma,gamma,alpha,id);


		}
	}
}



__device__ void llr_bpsk(float &bit,float Noise_stddev)
{

	bit= 2 * bit / (Noise_stddev*Noise_stddev);
}



__device__ void llr_qpsk(float &bit,float Noise_stddev)
{

	bit= _2SQRT2 * bit / (2*Noise_stddev*Noise_stddev);
}


__device__ char clip(float &bit,float scale,float offset){
	bit = bit * scale;
	if(bit < -offset)
		return (char) -offset;
	else if(bit > (offset - 1.0))
		return (char) (offset - 1.0);
	else
		return (char) (bit);
}

size_t roundUp(int group_size, int global_size)
{
	int r = global_size % group_size;
	if(r == 0)
	{
		return global_size;
	} else
	{
		return global_size + group_size - r;
	}
}

} /* namespace ecc */


