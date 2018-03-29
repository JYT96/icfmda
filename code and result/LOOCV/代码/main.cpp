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

void CalcSIG(const MatrixXd &A, MatrixXd &SIG) {
	int XDim = A.rows();
	int YDim = A.cols();
	MatrixXd X(XDim, XDim); X = A*A.transpose();
	MatrixXd Y(YDim, YDim); Y = A.transpose()*A;
	VectorXd degX(XDim); degX = X.diagonal();
	VectorXd degY(YDim); degY = Y.diagonal();
	for (int k = 0; k < XDim; k++) if (abs(degX(k)) < 0.1) degX(k) = 0.1;
	for (int k = 0; k < YDim; k++) if (abs(degY(k)) < 0.1) degY(k) = 0.1;
	int i, j;
	for (i = 0; i<YDim; i++) degY(i) = 1.0 / degY(i);
	for (i = 0; i<XDim; i++)
		for (j = 0; j<XDim; j++) {
			if (i == j) continue;
			SIG(i, j) = (1.0 / degX(i))*
				(1.0 / (degX(j) - X(i, j) + 1))*
				(((A.row(i).array()*A.row(j).array()).matrix()*degY).sum());
		}
}

void CalcF(const MatrixXd &A, const MatrixXd &SIM_DIS, MatrixXd &SIG_DIS, const MatrixXd &SIM_MIR, MatrixXd &SIG_MIR, MatrixXd &F, int updateX, int updateY) {
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

	MatrixXd VCorrX(XDim, XDim); VCorrX = SIG_DIS;
	MatrixXd VCorrY(YDim, YDim); VCorrY = SIG_MIR;

	for (i = 0; i<XDim; i++)
		for (j = 0; j<XDim; j++) {
			if (i == j || (i != updateX&&j != updateX)) continue;
			VCorrX(i, j) = (1.0 / degX(i))*
				(1.0 / (degX(j) - X(i, j) + 1))*
				(((A.row(i).array()*A.row(j).array()).matrix()*degY_inv).sum());
		}
	VCorrX = VCorrX / VCorrX.sum();
	for (i = 0; i<YDim; i++)
		for (j = 0; j<YDim; j++) {
			if (i == j || (i != updateY&&j != updateY)) continue;
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

void LOOCV(MatrixXd &A, const MatrixXi &AI, const MatrixXd &SIM_DIS, const MatrixXd &SIM_MIR) {
	int i, j, p, q;
	MatrixXd F(Nd, Nr);
	MatrixXf GRes(Nd, Nr);
	MatrixXf LRes(Nd, Nr);
	float cnt_GG = 0, cnt_GE = 0, cnt_GL = 0;
	float cnt_LG = 0, cnt_LE = 0, cnt_LL = 0;
	MatrixXd SIG_DIS(Nd, Nd);
	MatrixXd SIG_MIR(Nr, Nr);
	SIG_DIS = MatrixXd::Zero(Nd, Nd);
	SIG_MIR = MatrixXd::Zero(Nr, Nr);
	CalcSIG(A, SIG_DIS);
	CalcSIG(A.transpose(), SIG_MIR);
	for (i = 0; i<AI.rows(); i++)
		for (j = 0; j<AI.cols(); j++)
			if (AI(i, j) == 1) {
				//init counters
				cnt_GG = 0;  cnt_GE = 0;  cnt_GL = 0;
				cnt_LG = 0;  cnt_LE = 0;  cnt_LL = 0;
				//mask one positive example
				A(i, j) = 0;
				//calculate the score matrix
				CalcF(A, SIM_DIS, SIG_DIS, SIM_MIR, SIG_MIR, F, i, j);
				//statistics
				for (p = 0; p<F.rows(); p++) {
					if (p == i) {
						for (q = 0; q<F.cols(); q++)
							if (AI(p, q) == 0) {
								if (abs(F(i, j) - F(p, q))<0.000000000000001) {
									cnt_GE++;
									cnt_LE++;
								}
								else if (F(i, j)>F(p, q)) {
									cnt_GG++;
									cnt_LG++;
								}
								else {
									cnt_GL++;
									cnt_LL++;
								}
							}
					}
					else {
						for (q = 0; q<F.cols(); q++)
							if (AI(p, q) == 0) {
								if (abs(F(i, j) - F(p, q))<0.000000000000001) cnt_GE++;
								else if (F(i, j)>F(p, q)) cnt_GG++;
								else cnt_GL++;
							}
					}
				}
				//result rank
				GRes(i, j) = cnt_GL + cnt_GE / 2;
				LRes(i, j) = cnt_LL + cnt_LE / 2;
				printf("D(%3d)xR(%3d): GRank %8.1f   LRank%8.1f\n", i, j, GRes(i, j), LRes(i, j));
				//recover the positive example
				A(i, j) = 1;
			}
	FILE *tup = fopen("datasets/HMDD2.txt", "r");
	FILE *gfp = fopen("results/gloocv.txt", "w");
	FILE *lfp = fopen("results/lloocv.txt", "w");
	while (fscanf(tup, "%d %d\n", &j, &i) != EOF) {
		fprintf(gfp, "%.1f ", GRes(i - 1, j - 1) + 1);
		fprintf(lfp, "%.1f ", LRes(i - 1, j - 1) + 1);
	}
	fclose(tup);
	fclose(gfp);
	fclose(lfp);
}

int main() {
	MatrixXd A(Nd, Nr);
	read_matrix_sparse_d("datasets/HMDD2.txt", A);
	MatrixXi AI(Nd, Nr);
	read_matrix_sparse_i("datasets/HMDD2.txt", AI);
	MatrixXd SIM_DIS0(Nd, Nd);
	read_matrix_dense_d("datasets/DisSim0.txt", SIM_DIS0);
	MatrixXd SIM_DIS1(Nd, Nd);
	read_matrix_dense_d("datasets/DisSim1.txt", SIM_DIS1);
	MatrixXd SIM_DIS(Nd, Nd);
	SIM_DIS = (SIM_DIS0 + SIM_DIS1) / 2;
	SIM_DIS = SIM_DIS / SIM_DIS.sum();
	MatrixXd SIM_MIR(Nr, Nr);
	read_matrix_dense_d("datasets/miRSim.txt", SIM_MIR);
	SIM_MIR = SIM_MIR / SIM_MIR.sum();
	LOOCV(A, AI, SIM_DIS, SIM_MIR);
	return 0;
}
