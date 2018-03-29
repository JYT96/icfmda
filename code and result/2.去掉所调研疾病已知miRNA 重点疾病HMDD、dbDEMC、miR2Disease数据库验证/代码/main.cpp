#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#define Nr 495
#define Nd 383

void read_matrix_sparse_d(const char *filename, MatrixXd &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	while (fscanf(fp, "%d %d\n", &j, &i) != EOF)
		M((i - 1), (j - 1)) = 1.0;
	fclose(fp);
}

void read_matrix_sparse_i(const char *filename, MatrixXi &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	while (fscanf(fp, "%d %d\n", &j, &i) != EOF)
		M((i - 1), (j - 1)) = 1;
	fclose(fp);
}

void read_matrix_dense_d(const char *filename, MatrixXd &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	double val;
	for (i = 0; i<M.rows(); i++)
		for (j = 0; j<M.cols(); j++) {
			fscanf(fp, "%lf", &val);
			M(i, j) = val;
		}
	fclose(fp);
}

void read_Names(const char *filename, string *A) {
	ifstream inFile;
	inFile.open(filename, ios::in);
	int idx = 0;
	string tmpStr("");
	while (getline(inFile, tmpStr)) A[idx++] = tmpStr;
	inFile.close();
}

void read_Cared(const char *filename, int *A) {
	ifstream inFile;
	inFile.open(filename, ios::in);
	int idx = 0;
	while (inFile >> A[idx]) A[idx++]--;
	inFile.close();
}

void CalcF(const MatrixXd &A, const MatrixXd &SIM_DIS, const MatrixXd &SIM_MIR, MatrixXd &F) {
	int i, j, k;

	int XDim = A.rows();
	MatrixXd X(XDim, XDim); X = A*A.transpose();
	VectorXd degX(XDim); degX = X.diagonal();
	VectorXd degX_inv(XDim);
	for (k = 0; k < XDim; k++) {
		if (abs(degX(k)) < 0.1) degX(k) = 0.1;
		degX_inv(k) = 1.0 / degX(k);
	}

	int YDim = A.cols();
	MatrixXd Y(YDim, YDim); Y = A.transpose()*A;
	VectorXd degY(YDim); degY = Y.diagonal();
	VectorXd degY_inv(YDim);
	for (k = 0; k < YDim; k++) {
		if (abs(degY(k)) < 0.1) degY(k) = 0.1;
		degY_inv(k) = 1.0 / degY(k);
	}

	MatrixXd VCorrX(XDim, XDim); VCorrX = MatrixXd::Zero(XDim, XDim);
	MatrixXd VCorrY(YDim, YDim); VCorrY = MatrixXd::Zero(YDim, YDim);

	for (i = 0; i<XDim; i++)
		for (j = 0; j<XDim; j++) {
			if (i == j) continue;
			VCorrX(i, j) = (1.0 / degX(i))*
				(1.0 / (degX(j) - X(i, j) + 1))*
				(((A.row(i).array()*A.row(j).array()).matrix()*degY_inv).sum());
		}
	VCorrX = VCorrX / VCorrX.sum();
	for (i = 0; i<YDim; i++)
		for (j = 0; j<YDim; j++) {
			if (i == j) continue;
			VCorrY(i, j) = (1.0 / degY(i))*
				(1.0 / (degY(j) - Y(i, j) + 1))*
				(((A.transpose().row(i).array()*A.transpose().row(j).array()).matrix()*degX_inv).sum());
		}
	VCorrY = VCorrY / VCorrY.sum();
	MatrixXd F0(Nd, Nr);
	MatrixXd F1(Nd, Nr);
	F0 = (SIM_DIS + VCorrX)*A;
	F1 = A*(SIM_MIR.transpose() + VCorrY.transpose());
	F = F0 + F1;
}

int main(){
	ifstream inFile;
	int index;
	string tmpStr("");
	int i, j, k;
	ofstream outFile;
	//get names for miRNA and disease of HMDD2, check the cared disease number
	string *mir2_name = new string[495];
	read_Names("datasets/HMDD2/miRNA_Name.txt", mir2_name);
	string *dis2_name = new string[383];
	read_Names("datasets/HMDD2/Disease_Name.txt", dis2_name);
	int *cared = new int[14];
	read_Cared("datasets/HMDD2/cared.txt", cared);
	//get and prepare datasets
	MatrixXd A(Nd, Nr);
	read_matrix_sparse_d("datasets/HMDD2/HMDD2.txt", A);
	MatrixXi AI(Nd, Nr);
	read_matrix_sparse_i("datasets/HMDD2/HMDD2.txt", AI);
	MatrixXd SIM_DIS0(Nd, Nd);
	read_matrix_dense_d("datasets/HMDD2/DisSim0.txt", SIM_DIS0);
	MatrixXd SIM_DIS1(Nd, Nd);
	read_matrix_dense_d("datasets/HMDD2/DisSim1.txt", SIM_DIS1);
	MatrixXd SIM_DIS(Nd, Nd);
	SIM_DIS = (SIM_DIS0 + SIM_DIS1) / 2;
	SIM_DIS = SIM_DIS / SIM_DIS.sum();
	MatrixXd SIM_MIR(Nr, Nr);
	read_matrix_dense_d("datasets/HMDD2/miRSim.txt", SIM_MIR);
	SIM_MIR = SIM_MIR / SIM_MIR.sum();
	//calculate the scores
	MatrixXd F(Nd, Nr);
	string path("results/");
	string suffix(".txt");
	//output the result
	MatrixXi HMDD2(Nd, Nr);
	read_matrix_sparse_i("datasets/HMDD2/HMDD2.txt", HMDD2);
	MatrixXi dbDEMC(Nd, Nr);
	read_matrix_sparse_i("datasets/dbDEMC.txt", dbDEMC);
	MatrixXi miR2Disease(Nd, Nr);
	read_matrix_sparse_i("datasets/miR2Disease.txt", miR2Disease);
	for (index = 0; index<14; index++) {
		i = cared[index];
		for (k = 0; k<A.rows(); k++) {
			A(i, k) = 0;
			AI(i, k) = 0;
		}
		CalcF(A, SIM_DIS, SIM_MIR, F);
		outFile.open((path + dis2_name[i] + suffix).c_str(), ios::out);
		for (j = 0; j<Nr; j++) {
			outFile << dis2_name[i] << "&" << mir2_name[j] << "&" << F(i, j) << "&";
			if (dbDEMC(i, j) == 1) outFile << "dbDEMC;";
			if (miR2Disease(i, j) == 1) outFile << "miR2Disease;";
			if (HMDD2(i, j) == 1) outFile << "HMDD2;";
			outFile << endl;
		}
		outFile.close();
		for (k = 0; k<A.rows(); k++) {
			A(i, k) = HMDD2(i, k);
			AI(i, k) = HMDD2(i, k);
		}
	}
	delete[]mir2_name;
	delete[]dis2_name;
	delete[]cared;
}
