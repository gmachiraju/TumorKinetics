#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
using namespace std;
using std::cout;
using std::endl;

/*------------------------------------------------------------------------------------
Written by Gautam Machiraju, Mallick Lab

Global Variables
----------------
N : maxPrint
    Max number of iterations/data points for vector construction.
N_T0 : double
    Initial number of tumor cells.
N_H0 : double
    Inital number of healthy cells.
u_H : double
    Influx (rate of entry) of biomarker mass from HEALTHY cells with respect to time.
k_GR : double
    Fractional growth rate of tumor population.
k_decay : double
    Decaying rate of tumor growth.
k_EL : double
    Rate of plasma biomarker elimination.
q_0 : double
    Initial biomarker mass.

Note: all values are derived from Supplementary Material by Hori et al.
      Initial hardcoded biomarker: CA-125
      Will implement so user input is taken
------------------------------------------------------------------------------------*/

const int maxPrint = 100;        // Max Iterations
const double N_T0 = 100;        // N_T0 > 1
const double N_H0 = 100;        // ?
const double k_GR = 5.78e-3;    // k_GR = ln(2)/t_DT, where t_DT (tumor doubling time) = 120 days
const double k_decay = 1e-4; 
const double k_EL = 0.11;       // k_EL = ln(2)/t_1/2, where t_1/2 (half life for CA-125) = 6.4 days
const double q_0 = 150;         // ?
int N = 2048 * 2048;
double *N_T, *u_T, *C, *q_PL;            // host copies of N_T, u_T, q_PL
double *dN_T, *du_T, *dC, *dq_PL;         // device copies of N_T, u_T, q_PL


__global__ void tumorGrowth(double *N_T) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    N_T[t] = N_T0 * exp((k_GR * (1 - exp(-k_decay*t)))/k_decay);
}

__global__ void tumorMarkerInflux(double *N_T, double *u_T) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    double f_PLT = 0.1;
    double R_T = 4.5e-5;

    u_T[t] = f_PLT * R_T * N_T[t];
}

__global__ void markerMass(double *u_T, double *C, double *q_PL) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    double R_H = 4.5e-5;
    double f_PLH = 0.1;
    double u_H = f_PLH * R_H * N_H0;
    
    double A = u_H/k_EL;                                     // q_ss (steady state)
    double B = u_T[0]/(k_EL + k_GR);
    C[t] = q_0 - (u_T[t]/(k_EL + k_GR)) - (u_H/k_EL);        // growth factor
    q_PL[t] = A + (B * exp(k_GR*t)) + (C[t] * exp(-k_EL*t));    
}

#define THREADS_PER_BLOCK 512
int main() {
	int size = N * sizeof(int);

	// allocate space for device copies of N_T, u_T, q_PL
	cudaMalloc((void **)&dN_T, size);
	cudaMalloc((void **)&du_T, size);
	cudaMalloc((void **)&dC, size);
	cudaMalloc((void **)&dq_PL, size);

	// allocate space for host copies of N_T, u_T, q_PL and set up input values
	N_T = (double *)malloc(size); 
	u_T = (double *)malloc(size); 
	C = (double *)malloc(size);
	q_PL = (double *)malloc(size);

	// copy of inputs on the device
	cudaMemcpy(dN_T, N_T, size, cudaMemcpyHostToDevice);
	cudaMemcpy(du_T, u_T, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dq_PL, q_PL, size, cudaMemcpyHostToDevice);

	ofstream outfile("data/tumorSimData.csv");


	// launch tumorGrowth() kernel on GPU 
	//------------------------------------
	tumorGrowth<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dN_T);

	// copy result back to host
	cudaMemcpy(N_T, dN_T, size, cudaMemcpyDeviceToHost);


	// launch tumorMarkerInflux() on the device
	//------------------------------------------
	tumorMarkerInflux<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dN_T, du_T);

	// copy result back to host
	cudaMemcpy(u_T, du_T, size, cudaMemcpyDeviceToHost);


	// launch markerMass() on the device
	//-----------------------------------
	markerMass<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(du_T, dC, dq_PL);

	// copy result back to host
	cudaMemcpy(q_PL, dq_PL, size, cudaMemcpyDeviceToHost);

	printf("Data Preview:");
	printf("\n--------------");
    cout << "\nN_T" << "\t\t\t" << "u_T" << "\t\t\t\t" << "q_PL" << endl;
    cout << "[tumor cell count]" << "\t" << "[influx of biomarker mass]" << "\t" << "[plasma biomarker mass]" << endl;
    cout << "---------------------------------------------------------------------------" << endl;
  	

  	// Printing maxPrint points
  	for (int i = 0; i <= maxPrint; i++) {
		cout << N_T[i] << "\t\t\t" << u_T[i] << "\t\t\t" << q_PL[i] << endl;
  	}
  	printf("\n\t...full dataset available in data folder.\n");


	// Generate CSV data file
	for (int i = 0; i <= N; i++) {
	  	outfile << N_T[i] << "," << u_T[i] << "," << q_PL[i] << "\n";
	  	if (N_T[i] > 10e10) {
	  		break;                //Friberg et al -- visible tumor of ~1cm^3 only needs 10^9 cells (assuming 25 um cell size)
	  	}
	}

	// cleanup
	free(N_T); free(u_T); free(C); free(q_PL);
	cudaFree(dN_T); cudaFree(du_T); cudaFree(dC); cudaFree(dq_PL);
}

