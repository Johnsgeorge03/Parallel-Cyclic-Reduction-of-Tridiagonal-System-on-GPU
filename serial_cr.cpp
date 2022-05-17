#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
const int N = 4095;
int main(){
int i,j,k;
int index1,index2,offset;
double alpha,gamma;

/*
 * Serial Cyclic Reduction on CPU for solving Ax = b
 * where A is tridiagonal and the entire matrix is stored
 * including zeroes.
 * 
 * /


/* Memory Allocation */
double * x = new double[N];
for(i=0;i<N;i++)
	x[i] = 0.0;
	
double * F = new double[N];
double ** A = new double*[N];

for(i=0;i<N;i++){
	A[i] = new double[N];
	for(j=0;j<N;j++)
		A[i][j] = 0.;
	//F[i] = (double)i;
}
F[0] = -373.15;
F[N - 1] = -273.15;

A[0][0] = -2.0; A[0][1] = 1.0;
A[N-1][N-2] = 1.0; A[N-1][N-1] = -2.0;


for(i=1;i<N-1;i++){
	A[i][i] = -2.0;
	A[i][i-1] = 1.0;
	A[i][i+1] = 1.0;
}
auto start = high_resolution_clock::now();

/* Cyclic Reduction */
for(i=0;i<log2(N+1)-1;i++){
	//#pragma omp parallel for private(offset, alpha, gamma, index1, index2, k)
	for(j=int(pow(2,i+1))-1;j<N;j+= int(pow(2,i+1))){
		offset = pow(2,i);
		index1 = j - offset;
		index2 = j + offset;
		alpha = A[j][index1]/A[index1][index1];
		gamma = A[j][index2]/A[index2][index2];
		for(k=0;k<N;k++){
			A[j][k] -= (alpha*A[index1][k] + gamma*A[index2][k]);
		}
		F[j] -= (alpha*F[index1] + gamma*F[index2]);
	}
}

/* Back Substitution */
int index = (N-1)/2;
x[index] = F[index]/A[index][index];
for(i=log2(N+1)-2;i>=0;i--){
	//#pragma omp parallel for private(j, index1, index2, offset, k)
	for(j = int(pow(2,i+1))-1;j<N; j += int(pow(2,i+1))){
		offset = pow(2,i);
		index1 = j - offset;
		index2 = j + offset;
		x[index1] = F[index1];
		x[index2] = F[index2];
		for(k=0;k<N;k++){
			if(k!= index1)
			x[index1] -= A[index1][k]*x[k];
			if(k!= index2)
			x[index2] -= A[index2][k]*x[k];
		}
		x[index1] = x[index1]/A[index1][index1];
		x[index2] = x[index2]/A[index2][index2];
	}
}

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << duration.count() << endl;
cout<< "---------------------------------------------" << endl;

//for(i=0;i<N;i++){
	//cout <<"i : "<<i << "--  " << x[i] << endl;
//}

delete[] x;
delete[] F;
for(i=0;i<N;i++)
	delete[] A[i];
	
delete[] A;

return 0;
}
