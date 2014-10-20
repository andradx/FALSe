/*
 * ldpc.cpp
 *
 *  Created on: Oct 17, 2014
 *      Author: andrade
 */

#include "ldpc.h"
#include "vector_operations.h"

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
	if(cumBNW != NULL) cudaFreeHost(cumBNW);
	if(cumCNW != NULL) cudaFreeHost(cumCNW);
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

	cudaHostAlloc((void**)&cumBNW,sizeof(int)*N,cudaHostAllocWriteCombined);					// allocate space for cumBNW
	cudaHostAlloc((void**)&cumCNW,sizeof(int)*M,cudaHostAllocWriteCombined);					// allocate space for cumCNW

	cudaHostAlloc((void**)&BNAdjacency,sizeof(int)*edges,cudaHostAllocWriteCombined);	// allocate space for BNAdjacency
	cudaHostAlloc((void**)&CNAdjacency,sizeof(int)*edges,cudaHostAllocWriteCombined);	// allocate space for CNAdjacency

	cudaHostAlloc((void**)&h_ligx,sizeof(int)*edges,cudaHostAllocWriteCombined);			// allocate space for h_ligx
	cudaHostAlloc((void**)&h_ligf,sizeof(int)*edges,cudaHostAllocWriteCombined);			// allocate space for h_ligf


	memset((void*)BNW, -1, N*sizeof(int));
	memset((void*)CNW, -1, M*sizeof(int));
	memset((void*)BNAdjacency, -1, sizeof(int)*edges);
	memset((void*)CNAdjacency, -1, sizeof(int)*edges);
	memset((void*)h_ligx, -1, sizeof(int)*edges);
	memset((void*)h_ligf, -1, sizeof(int)*edges);


	for(i = 0; i < N; i++)																	// read BNW
		fscanf(fd,"%d",&BNW[i]);

	for(i = 0; i < M; i++)																	// read CNW
		fscanf(fd,"%d",&CNW[i]);

	cumBNW[0] = 0;
	for(i = 1; i < N; i++)																	// compute cumBNW
		cumBNW[i] = cumBNW[i-1] + BNW[i];

	cumCNW[0] = 0;
	for(i = 1; i < M; i++)																	// compute cumCNW
		cumCNW[i] = cumCNW[i-1] + CNW[i];

	edges = cumBNW[N-1] + BNW[0];

	if(cumCNW[M-1] + CNW[0] != edges)
		printf("Number of edges does not add right\n");

	for(i = 0; i < N; i++)																	// read BNAdjacency padded to the MaxBNW
		for(j = 0; j < BNW[i]; j++){
			fscanf(fd,"%d",&BNAdjacency[cumBNW[i]+j]);
			//printf("BNW: %d %d\n",i,j);
		}

	for(i = 0; i < M; i++)																	// read CNAdjacency padded to the MaxBNW
		for(j = 0; j < CNW[i]; j++){
			fscanf(fd,"%d",&CNAdjacency[cumCNW[i]+j]);
			//printf("CNW: %d %d\n",i,j);
		}

	fclose(fd);

	generate_ligVectors();																	// generate the indexing tables for decoding
	test_regularity();																		// set regularity flag

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
			fprintf(fd,"%d ",BNAdjacency[cumBNW[i]+j]);
			//printf("BNW: %d %d\n",i,j);
		}
		fprintf(fd,"\n");
	}

	for(i = 0; i < M; i++){											// read CNAdjacency padded to the MaxBNW
		for(j = 0; j < CNW[i]; j++){
			fprintf(fd,"%d ",CNAdjacency[cumCNW[i]+j]);
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

	for(i = 0; i < M; i++)
		for(j = 0; j < CNW[i]; j++){
			//printf("h(%2d,%2d) = 1\n",i+1,(CNAdjacency[cumCNW[i]+j]));
			h[i*N+(CNAdjacency[cumCNW[i]+j]-1)] = 1;
		}
}

void ldpc::print_PCM(FILE *fd){

	int i, j;

	if(h == NULL) generate_PCM();

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++){
			fprintf(fd,"%d ",h[i*N+j]);
		}
		fprintf(fd,"\n");
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

void ldpc::print_weights(FILE *fd){

	int i;

	fprintf(fd,"BNW: ");
	for(i = 0; i < N; i++){
		fprintf(fd,"%d ",BNW[i]);
	}
	fprintf(fd,"\n");
	fprintf(fd,"CNW: ");
	for(i = 0; i < M; i++){
		fprintf(fd,"%d ",CNW[i]);
	}
	fprintf(fd,"\n");

	fprintf(fd,"cumBNW: ");
	for(i = 0; i < N; i++){
		fprintf(fd,"%d ",cumBNW[i]);
	}
	fprintf(fd,"\n");

	fprintf(fd,"cumCNW: ");
	for(i = 0; i < M; i++){
		fprintf(fd,"%d ",cumCNW[i]);
	}
	fprintf(fd,"\n");
}

void ldpc::print_ligVectors(FILE *fd){

	int i;

	fprintf(fd,"ligx: ");
	for(i = 0; i < edges; i++){
		fprintf(fd,"%d ",h_ligx[i]);
	}
	fprintf(fd,"\n");
	fprintf(fd,"ligf: ");
	for(i = 0; i < edges; i++){
		fprintf(fd,"%d ",h_ligf[i]);
	}
	fprintf(fd,"\n");
}

void ldpc::print_adjacencies(FILE *fd){

	int i;

	fprintf(fd,"BNAdjacency: ");
	for(i = 0; i < edges; i++){
		fprintf(fd,"%d ",BNAdjacency[i]);
	}
	fprintf(fd,"\n");
	fprintf(fd,"CNAdjacency: ");
	for(i = 0; i < edges; i++){
		fprintf(fd,"%d ",CNAdjacency[i]);
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

void ldpc::test_regularity(){

	if(edges != M*MaxCNW)
		h_regular = 0;
	else
		h_regular = 1;
}


__global__ void setup_kernel(curandState *state,unsigned int seed,int Nlim)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	if(id<Nlim)
		curand_init(seed, id, 0, &state[id]);

}

__global__ void generate_noise(curandState *state,char4 *gamma,char4 *alpha, float Noise_stddev,char *tx_codeword,int *BNW,int *cumBNW,int N,int Nlim)
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

			broadcast_bits(pack_gamma,gamma,alpha,BNW,cumBNW,N,id);


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


__device__ void broadcast_bits(char4 pack_gamma,char4 *gamma,char4 *alpha,int *BNW,int *cumBNW,int N,unsigned int tid)
{


	if(tid<N){

		gamma[tid] = pack_gamma;

#pragma unroll
		for(int i = 0; i < BNW[i]; i++)
			alpha[cumBNW[tid]+i] = pack_gamma;
	}
}

template <int maxBNW> __global__ void bitnode_processing(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder)
{
	register int tid = threadIdx.x + blockIdx.x*blockDim.x;

	register int write_idx[maxBNW];
	register int4 ld_llr[maxBNW];
	register int4 st_llr[maxBNW];
	register int4 sum;

	register int i;

	if(tid<N){

		sum = load_char4_to_int4(gamma[tid]);

		for(i = 0; i < BNW[i]; i++){
			ld_llr[i] = load_char4_to_int4(beta[cumBNW[tid]+i]);			// load the beta messages
			write_idx[i] = ligx[cumBNW[tid]+i];								// load the writing indexes
			sum += ld_llr[i];
		}

		for(i = 0; i < BNW[i]; i++){
			st_llr[i] = sum - ld_llr[i];
			alpha[cumBNW[tid]+i] = store_int4_to_char4(st_llr[write_idx[i]],128);					// store the alpha messages
		}
	}
}

template __global__ void bitnode_processing<1>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<2>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<3>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<4>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<5>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<6>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);
template __global__ void bitnode_processing<7>(char4 *gamma,char4 *alpha,char4 *beta,int *BNW,int *cumBNW,int *ligx,int N,decoding_type decoder);

template <int maxCNW> __global__ void checknode_processing(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder)
{
	register int tid = threadIdx.x + blockIdx.x*blockDim.x;

	register int write_idx[maxCNW];
	register int4 ld_llr[maxCNW];
	register int4 st_llr[maxCNW];
	register int4 sig_llr[maxCNW];
	register int4 mag_llr[maxCNW];
	register int4 min_idx;

	register int4 min1,min2;

	register int i;

	if(tid<M){

		ld_llr[0] = load_char4_to_int4(beta[cumCNW[tid]+0]);									// load the first beta message
		write_idx[0] = ligf[cumCNW[tid]+0];														// load the first writing index
		min1 = min2 = mag_llr[0] = abs(ld_llr[0]);
		min_idx = make_int4(0,0,0,0);

		for(i = 1; i < CNW[i]; i++){
			ld_llr[i] = load_char4_to_int4(beta[cumCNW[tid]+i]);								// load the beta messages
			mag_llr[i] = abs(ld_llr[i]);														// compute the magnitude of beta messages
			write_idx[i] = ligf[cumCNW[tid]+i];													// load the writing indexes

			min_idx = min(mag_llr[i],min1,min2,i);												// find minimums and store the indexes of absolute min1
		}


###TERMINATE MINIMUM PROPAGATION###


		for(i = 0; i < CNW[i]; i++){
			st_llr[i] = mag_llr[i] * sig_llr[i];
			alpha[cumCNW[tid]+i] = store_int4_to_char4(st_llr[write_idx[i]],128);					// store the alpha messages
		}
	}
}

template __global__ void checknode_processing<1>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);
template __global__ void checknode_processing<2>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);
template __global__ void checknode_processing<3>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);
template __global__ void checknode_processing<4>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);
template __global__ void checknode_processing<5>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);
template __global__ void checknode_processing<6>(char4 *alpha,char4 *beta,int *CNW,int *cumCNW,int *ligf,int M,decoding_type decoder);

//
//__device__ void msa(int gamma,int *ld_llr,int *st_llr,int *BNW)
//{
//
//	register int i;
//	register int sum = 0;
//
//	for(i = 0; i < BNW[i]; i++){
//		sum += ld_llr[i];
//	}
//	sum += gamma;
//	for(i = 0; i < BNW[i]; i++){
//		st_llr[i] = sum - ld_llr[i];
//	}
//
//
//}
} /* namespace ecc */


