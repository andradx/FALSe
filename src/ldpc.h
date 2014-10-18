/*
 * ldpc.h
 *
 *  Created on: Oct 17, 2014
 *      Author: andrade
 */

#ifndef LDPC_H_
#define LDPC_H_

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector_types.h>


#define WORDS 4
#define _2SQRT2 2.828427125f
#define _SQRT2_2 0.707107678f
#define _SQRT2 1.414213562f
#define SCALE 2.0
#define OFFSET 128.0

void checkCUDAError(const char *msg);

namespace ecc {



__global__ void setup_kernel(curandState *state,unsigned int seed,int Nlim);
__device__ void llr_bpsk(float &bit,float Noise_stddev);
__device__ void llr_qpsk(float &bit,float Noise_stddev);
__device__ char clip(float &bit,float scale,float offset);
__device__ void BroadCastBits(pack_gamma,gamma,alpha,id);


size_t roundUp(int group_size, int global_size);


class ldpc {
private:

	// code properties
	std::string name;						// LDPC code name
	int N;									// Codeblock size
	int M;									// No. of parity-check restrictions
	int K;									// Information length
	int MaxBNW;								// Maximum BN weight
	int MaxCNW;								// Maximum CN weight
	int *BNW;								// Array of size N containing the weights of each BN
	int *CNW;								// Array of size M containing the weights of each CN
	int *BNAdjacency;						// Array storing the indexes of the CNs adjacent to each BN strided to MaxBNW
	int *CNAdjacency;						// Array storing the indexes of the BNs adjacent to each CN strided to MaxCNW
	int *h_ligx, *d_ligx;					// Indexing for BNs strided to MaxBNW
	int *h_ligf, *d_ligf;					// Indexing for CNs strided to MaxCNW
	int *h;									// Parity-check matrix



	void generate_ligVectors();						// Generate ligx and ligf vectors for indexing of \alpha and \beta messages
	void allocate_decoderMemory();					// Allocate the required memory buffers for decoding on the host and device
	void copy_decoderMemory(cudaMemcpyKind kind);	// Copy data between device and host

	// transmission methods
	void setup_rng(unsigned int seed);								// Initialize data structures required for PRNG


	// transmission properties
	char4 *h_gamma, *d_gamma;				// Array of size N containing \gamma LLRs
	char4 *h_beta, *d_beta;					// Array of size MaxBNW*N containing \beta LLRs
	char4 *h_alpha, *d_alpha;				// Array of size MaxCNW*M containing \alpha LLRs
	char *h_word, *d_word;					// Array of size N containing decoded bits
	curandState *rng_state;					// Array of size N containing the state of the PRNG

public:
	ldpc();									// not really useful
	ldpc(const char *alist_file);			// load code from alist file
	virtual ~ldpc();
	void copy_ligVectors();					// Copy ligx and ligf vectors to the device
	void init_device(int deviceID);			// Initialize the device
	void copy_data(cudaMemcpyKind kind);	// Public caller to copy_decoderMemory



	// decoding methods
	void msa_decoding();					// Min-Sum Decoder
	void scmsa_decoding();					// Self-Corrected Min-Sum Decoder

	//other methods
	void write_alistFile(const char *alist_file);	// Writes loaded code contents to alist file
	void print_PCM(FILE *fd);						// Prints parity-check matrix to file descriptor
	void print_ligVectors(FILE *fd);				// Prints ligx and ligf vectors to file descriptor
	void generate_PCM();							// Build parity-check matrix from the adjacency vectors
};



} /* namespace ecc */
#endif /* LDPC_H_ */
