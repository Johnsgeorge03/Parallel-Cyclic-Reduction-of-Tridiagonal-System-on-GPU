#include <iostream>
#include <math.h>
#include <float.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
const int N = pow(2, 15);

/*Thomas algorithm to solve Ax = b, 
 * where A is tridiagonal matrix.
 * Only non zero entries of A is stored
 */

void solve(double* a, double* b, double* c, double* d, int n) {
    n--;
    c[0] /= b[0];
    d[0] /= b[0];

    for (int i = 1; i < n; i++) {
        c[i] /= b[i] - a[i]*c[i-1];
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
    }

    d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

    for (int i = n; i-- > 0;) {
        d[i] -= c[i]*d[i+1];
    }
}

int main() {
	/* Memory allocation */
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
	
	auto start = high_resolution_clock::now();
	solve(Dsub,D,Dsup,F,N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << duration.count() << endl;
	cout<< "---------------------------------------------" << endl;
	
	double err = DBL_MIN; 
	double temp;
	for(int i=0;i<N;i++){
		temp = abs( F[i] - ytrue[i]);
		if (temp > err) err  = temp;
	}
	cout<<"Max_error  :"<< err << endl;
	
	//~ for(int i=0;i<N;i++){
		//~ cout <<"i : "<< i << "--  " << F[i] << " true :" << ytrue[i] << endl;
	//~ }
	
	free(Dsub);
	free(D);
	free(Dsup);
	free(F);
	return 0;
	
}
