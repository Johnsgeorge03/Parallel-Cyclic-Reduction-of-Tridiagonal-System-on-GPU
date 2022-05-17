#include <Eigen/Core>
#include <iostream>
#include <math.h>
#include <float.h>
#include <chrono>
#include <Eigen/Dense>
using Eigen::internal::BandMatrix;
#include <Eigen/SparseCholesky>
using namespace Eigen;
using namespace std;
using namespace std::chrono;
int main()
/*
 * Eigen library has to be installed for this code to work.
 * The entire runtime could be longer compared to the actual
 * solve due to conversion of matrices into different formats.
 * 
 * The code can be run by using the command 
 * g++ -I ../../../usr/local/include/eigen-3.4.0/ eigentest.cpp -o eigentest
 * */

{
	int N = pow(2, 15);
    int rows = N;
    int cols = N;
    int sups = 1;
    int subs = 1;
    double dx 		= 1.0/(N + 1);
	double *ytrue 	= (double*) malloc(N*sizeof(double));
	for(int i = 0;i<N;i++){
		ytrue[i] 	= double(-100.0*(i+1)*dx + 373.15);
	}
    BandMatrix<double> A(rows,cols,sups,subs);
	Eigen::VectorXd b(cols);
	b(0) = -373.15;
	b(cols-1) = -273.15;
    A.diagonal().setConstant(-2);
   

    for(int i = 1; i <= A.supers(); ++i)
    {
        A.diagonal(i).setConstant(i);
    }

    for(int i = 1; i <= A.subs(); ++i)
    {
        A.diagonal(-i).setConstant(i);
    }
    
    Eigen::MatrixXd Ad = A.toDenseMatrix();
    Eigen::SparseMatrix<double>  As = Ad.sparseView();

	VectorXd x;
    SimplicialLDLT<SparseMatrix<double> > solver;
    
	auto start = high_resolution_clock::now();
    x = solver.compute(As).solve(b);
    auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << duration.count() << endl;
	cout<< "---------------------------------------------" << endl;
	
    double err = DBL_MIN; 
	double temp;
	for(int i=0;i<N;i++){
		temp = abs( x[i] - ytrue[i]);
		if (temp > err) err  = temp;
	}
	cout<<"Max_error  :"<< err << endl;
	//std::cout << x <<std::endl;
    //std::cout << Ad << "\n\n";
    //std::cout << b << std::endl;
    
    //~ for(int i=0;i<N;i++){
		//~ cout <<"i : "<< i <<"---true :" << ytrue[i] << endl;
	//~ }
}
