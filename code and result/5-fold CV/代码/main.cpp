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

void read_matrix_sparse_d(const char *filename, MatrixXd &M){
  FILE *fp = fopen(filename,"r");
  int i,j;
  while(fscanf(fp,"%d %d\n",&j,&i)!=EOF)
    M((i-1),(j-1)) = 1.0;
  fclose(fp);
}

void read_matrix_sparse_i(const char *filename, MatrixXi &M){
  FILE *fp = fopen(filename,"r");
  int i,j;
  while(fscanf(fp,"%d %d\n",&j,&i)!=EOF)
    M((i-1),(j-1)) = 1;
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

void CalcF(const MatrixXd &A, const MatrixXd &SIM_DIS, const MatrixXd &SIM_MIR, MatrixXd &F){
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
			if (i == j ) continue;
			VCorrX(i, j) = (1.0 / degX(i))*
				(1.0 / (degX(j) - X(i, j) + 1))*
				(((A.row(i).array()*A.row(j).array()).matrix()*degY_inv).sum());
		}
	VCorrX = VCorrX / VCorrX.sum();
	for (i = 0; i<YDim; i++)
		for (j = 0; j<YDim; j++) {
			if (i == j ) continue;
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

double find_rank(const ArrayXd &V,double val){
  int lowerbound=0,upperbound=V.size()-1;
  int middle;
  double ret;
  while(1){
    if(upperbound-lowerbound == 1) break;
    middle = (lowerbound+upperbound)/2;
    if(abs(V(middle)-val)<0.00000001){
      if(abs(V(lowerbound)-val)>0.00000001) lowerbound++;
      else if(abs(V(upperbound)-val)>0.00000001) upperbound--;
      else{
        ret = ((double)lowerbound+(double)upperbound)/2;
        break;
      }
    }else if(V(middle)>val) lowerbound = middle;
    else upperbound = middle;
  }
  if(upperbound-lowerbound==1){
    if(V(upperbound)<val) ret = (double)upperbound;
    else ret = (double)lowerbound;
  }
  return ret;
}

void FFCV(MatrixXd &A,const MatrixXi &AI, const MatrixXd &SIM_DIS, const MatrixXd &SIM_MIR){
  int rep,i,j,p,n;
  MatrixXd F(Nd,Nr);
  ArrayXd V(Nd*Nr-5430);
  //prepare index matrix
  MatrixXi idx(5430,3); //disease,mirna,set_id
  ArrayXd score_rank(5430); //score, rank
  FILE *tup = fopen("datasets/HMDD2.txt","r");
  n = 0;
  while(fscanf(tup,"%d %d\n",&j,&i)!=EOF){
    idx.row(n) << i-1,j-1,n%5;
    n++;
  }
  fclose(tup);
  FILE *ff_fp = fopen("results/ffcv.txt","w");
  for(rep=0;rep<100;rep++){
    //split the seeds to 5 random sets
    for(i=0;i<5430;i++){
      j = rand()%5430;
      p = idx(i,2);
      idx(i,2) = idx(j,2);
      idx(j,2) = p;
    }
    for(n=0;n<5;n++){
      //prepare matrix
		for (i = 0; i < 5430; i++)
			if (idx(i, 2) == n) A(idx(i, 0), idx(i, 1)) = 0;
      //calculate the score matrix
		CalcF(A, SIM_DIS, SIM_MIR, F);
      //gather and rank the scores
      p = 0;
      for(i=0;i<AI.rows();i++)
        for(j=0;j<AI.cols();j++)
          if(AI(i,j)==0) V(p++) = F(i,j);
      std::sort(V.data(),V.data()+V.size(),std::greater<double>());
      for(i=0;i<5430;i++) 
        if(idx(i,2) == n) score_rank(i) = find_rank(V,F(idx(i,0),idx(i,1)));
      //recover matrix
      for(i=0;i<5430;i++)
        if(idx(i,2) == n) A(idx(i,0),idx(i,1)) = 1;
    }
    for(i=0;i<5430;i++) fprintf(ff_fp,"%.1f ",score_rank(i)+1);
    fprintf(ff_fp,"\n");
    printf("finished %3d FFCV\n",rep+1);
  }
  fclose(ff_fp);
}



int main(){
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
	FFCV(A, AI, SIM_DIS, SIM_MIR);
	return 0;
}
