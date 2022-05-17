#include<iostream>
#include<iomanip>
#include<math.h> 
#include<chrono>
#include<float.h>

/*
 * Parallel Cyclic Reduction on GPU 
 * for solving Ax = b where A is tridiagonal and 
 * only the non-zero elements of A are stored. 
 * Compile using 
 * nvcc -g -G -c "%f"
 * build using
 * nvcc -g -G  -o "%e" "%f"
 *
 */

using namespace std;
using namespace std::chrono;
const int N = pow(2, 12);
const int threads_per_block = 256;

__global__ void kernel(double *d_F, double *d_D, double *d_Dsub, double *d_Dsup, \\
					   double *d_Ft, double *d_Dt, double *d_Dsubt, double *d_Dsupt,
					   double *d_y, int N, int i, int stride)
{
	int id 	   = blockIdx.x*blockDim.x + threadIdx.x;
	int index1 = id - stride;
	int index2 = id + stride;
	double gamma = 0.0, alpha = 0.0;
	
	if(id < N){
		if (index1 < 0){
			gamma   		= -d_Dsup[id]/d_D[index2];
			d_Dt[id]		= d_D[id] + d_Dsub[index2]*gamma;
			d_Dsupt[id]    	= d_Dsup[index2]*gamma;
			d_Ft[id]    	= d_F[id] + gamma*d_F[index2];
		}
			
		else if (index2 > N - 1 ){
			alpha   		= -d_Dsub[id]/d_D[index1];
			d_Dsubt[id]	    = d_Dsub[index1]*(alpha);
			d_Dt[id] 	    = d_D[id] + d_Dsup[index1]*alpha;
			d_Ft[id]   	    = d_F[id] + alpha*d_F[index1];
		}
			
		else{
			alpha   = -d_Dsub[id]/d_D[index1];
			gamma   = -d_Dsup[id]/d_D[index2];
			
			d_Dsubt[id]	    = d_Dsub[index1]*(alpha);
			d_Dt[id]    	= d_D[id] + (d_Dsup[index1]*alpha + d_Dsub[index2]*gamma);
			d_Dsupt[id]     = d_Dsup[index2]*gamma;
			d_Ft[id]        = d_F[id] + (alpha*d_F[index1] + gamma*d_F[index2]);
		}
		
		d_y[id] = d_Ft[id]/d_Dt[id];
		
	}
	
}


int main()
{
	/* Memory Allocation */
	double *y 		= (double*) malloc(N*sizeof(double));
	double *ytrue 	= (double*) malloc(N*sizeof(double));
	double dx 		= 1.0/(N + 1);

	for(int i = 0;i<N;i++){
		y[i] 		= 0.0;
		ytrue[i] 	= double(-100.0*(i+1)*dx + 373.15);
	}	

	double *F    = (double*) malloc( N* sizeof(double));
	double *D    = (double*) malloc( N* sizeof(double));
	double *Dsub = (double*) malloc( N* sizeof(double));
	double *Dsup = (double*) malloc( N* sizeof(double));
	
	
	double *d_F, *d_D, *d_Dsub, *d_Dsup, \\
			*d_Ft, *d_Dt, *d_Dsubt, *d_Dsupt, *d_y;
	
	for(int i=0;i<N;i++){
		D[i]    = -2.0;
		Dsub[i] = 1.0;
		Dsup[i] = 1.0;
		F[i]    = 0.0;
	}
	F[0]      = -373.15;
	F[N - 1]  = -273.15;
	

	Dsub[0]   = 0.0;
	Dsup[N-1] = 0.0;
	
	
	int num_blocks 			= max(N/threads_per_block, 1);
	cout<<"num_blocks = "<<num_blocks<<"\n"<<endl;
	dim3 grid_size (num_blocks);
	
	auto start_mem_transfer = high_resolution_clock::now();
	
	cudaMalloc(&d_F , N*sizeof(double));
	cudaMalloc(&d_D , N*sizeof(double));
	cudaMalloc(&d_Dsub , N*sizeof(double));
	cudaMalloc(&d_Dsup, N*sizeof(double));
	
	cudaMalloc(&d_Ft , N*sizeof(double));
	cudaMalloc(&d_Dt , N*sizeof(double));
	cudaMalloc(&d_Dsubt , N*sizeof(double));
	cudaMalloc(&d_Dsupt, N*sizeof(double));
	cudaMalloc(&d_y, N*sizeof(double));
	
	cudaMemcpy(d_F, F, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, D, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dsub, Dsub, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dsup, Dsup, N*sizeof(double), cudaMemcpyHostToDevice);
	
	auto stop_mem_transfer = high_resolution_clock::now();
	auto durationt1 = duration_cast<microseconds>(stop_mem_transfer - start_mem_transfer);
	
	
	auto start = high_resolution_clock::now();
	
	for(int i = 0; i < log2(N); i++){
		int stride = pow(2, i);
		kernel<<<grid_size , threads_per_block>>>\\
		(d_F, d_D, d_Dsub, d_Dsup, d_Ft, d_Dt, d_Dsubt, d_Dsupt, \\
											d_y,  N, i, stride);
											
		cudaDeviceSynchronize();
		double* t1 = d_Ft;
		double* t2 = d_Dt;
		double* t3 = d_Dsubt;
		double* t4 = d_Dsupt;
		
		d_Ft 	= d_F;
		d_Dt 	= d_D;
		d_Dsubt = d_Dsub;
		d_Dsupt = d_Dsup;
		
		d_F 	= t1;
		d_D 	= t2;
		d_Dsub  = t3;
		d_Dsup  = t4;
	}
	//cout<<"first trans :"<< durationt1.count() <<endl;
	
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "kernel compution :"<<duration.count() << endl;
	cout<< "---------------------------------------------" << endl;
	
	auto start_mem_transfer2 = high_resolution_clock::now();
	cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);
	auto stop_mem_transfer2 = high_resolution_clock::now();
	auto durationt2 = duration_cast<microseconds>(stop_mem_transfer2 - start_mem_transfer2);
	auto total_mem_trans = durationt1 + durationt2;
	
	cout << "Mem transfer and allocation :"<<total_mem_trans.count() << endl;
	cout<< "---------------------------------------------" << endl;
	
	//cudaMemcpy(Dsup, d_Dsup, N*sizeof(double), cudaMemcpyDeviceToHost);
	double err = DBL_MIN; 
	double temp;
	for(int i=0;i<N;i++){
		temp = abs( y[i] - ytrue[i]);
		if (temp > err) err  = temp;
	}
	cout<<"Max_error  :"<< err << endl;
	
	//~ for(int i=0;i<N;i++){
		//~ cout <<"i : "<< i << "--  " << y[i] << " true :" << ytrue[i] << endl;
	//~ }	
	
	free(Dsub);
	free(D);
	free(Dsup);
	free(F);
	return 0;
}
