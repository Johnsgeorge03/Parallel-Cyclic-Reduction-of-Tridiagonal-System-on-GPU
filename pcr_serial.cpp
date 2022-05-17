#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <float.h>
using namespace std;
using namespace std::chrono;
const int N = pow(2, 12);
int main(){
int index1,index2,offset;
double alpha,gamma;

/*
 * Serial version of Parallel Cyclic Reduction on CPU 
 * for solving Ax = b where A is tridiagonal and 
 * only the non-zero elements of A are stored
 *
 * /

/* Memory Allocation */
double *x = (double*) malloc(N*sizeof(double));
double *xtrue = (double*) malloc(N*sizeof(double));
double dx = 1.0/(N + 1);

for(int i = 0;i<N;i++){
	x[i] = 0.0;
	xtrue[i] = double(-100.0*(i+1)*dx + 373.15);
}	

double *F    = (double*) malloc( N* sizeof(double));
double *D    = (double*) malloc( N* sizeof(double));
double *Dsub = (double*) malloc( N* sizeof(double));
double *Dsup = (double*) malloc( N* sizeof(double));

double *Ft    = (double*) malloc( N* sizeof(double));
double *Dt    = (double*) malloc( N* sizeof(double));
double *Dsubt = (double*) malloc( N* sizeof(double));
double *Dsupt = (double*) malloc( N* sizeof(double));

for(int i=0;i<N;i++){
	D[i]    = -2.0;
    Dsub[i] = 1.0;
    Dsup[i] = 1.0;
	F[i]    = 0.0;
	Dt[i]    = -2.0;
    Dsubt[i] = 1.0;
    Dsupt[i] = 1.0;
	Ft[i]    = 0.0;
}
F[0]      = -373.15;
F[N - 1]  = -273.15;
Ft[0]     = -373.15;
Ft[N - 1] = -273.15;

Dsub[0]   = 0.0;
Dsup[N-1] = 0.0;
Dsubt[0]   = 0.0;
Dsupt[N-1] = 0.0;


auto start = high_resolution_clock::now();

/* Parallel Cyclic Reduction */
for(int i = 0; i < log2(N); i++){ // reduction stages
	for(int j = 0; j < N; j++){ // matrix row iteration
		offset = int(pow(2,i));
		index1 = j - offset;
		index2 = j + offset;
		if (index1 < 0){
			gamma    = -Dsup[j]/D[index2];
			Dt[j] 	 = D[j] + Dsub[index2]*gamma;
			Dsupt[j] = Dsup[index2]*gamma;
			Ft[j]    = F[j] + gamma*F[index2];
		}
		
		else if (index2 > N - 1){
			alpha    = -Dsub[j]/D[index1];
			Dsubt[j] = Dsub[index1]*(alpha);
			Dt[j] 	 = D[j] + (Dsup[index1]*alpha);
			Ft[j]    = F[j] + alpha*F[index1];
		}
		
		else{
			
			alpha = -Dsub[j]/D[index1];
			gamma = -Dsup[j]/D[index2];
			
			Dsubt[j]  = Dsub[index1]*(alpha);
			Dt[j] 	  = D[j] + (Dsup[index1]*alpha + Dsub[index2]*gamma);
			Dsupt[j]  = Dsup[index2]*(gamma);
			Ft[j]     = F[j] + (alpha*F[index1] + gamma*F[index2]);
		}
		
	}
	
	
	double* t1 = Ft;
	double* t2 = Dt;
	double* t3 = Dsubt;
	double* t4 = Dsupt;
	
	Ft 	= F;
	Dt 	= D;
	Dsubt = Dsub;
	Dsupt = Dsup;
	
	F 	= t1;
	D 	= t2;
	Dsub  = t3;
	Dsup  = t4;
	
	
	if( i == log2(N) - 1){
		for(int k = 0; k < N; k+= 1){
			x[k] = F[k]/D[k];
		}
	}
}



auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << duration.count() << endl;
cout<< "---------------------------------------------" << endl;

double err = DBL_MIN; 
double temp;
for(int i=0;i<N;i++){
	temp = abs( x[i] - xtrue[i]);
	if (temp > err) err  = temp;
}
cout<<"Max_error  :"<< err << endl;

//~ for(int i=0;i<N;i++){
	//~ cout <<"i : "<< i << "--  " << x[i] << " true :" << xtrue[i] << endl;
//~ }

free(Dsub);
free(D);
free(Dsup);
free(F);

return 0;
}
