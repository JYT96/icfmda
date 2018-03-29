#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
using namespace std;
using namespace Eigen;
#include "global.h"
#include "CalcF.h"
#include "helper.h"

int main(){
  ifstream inFile;
  int index;
  string tmpStr("");
  int i,j,k;
  ofstream outFile;
  //get names for miRNA and disease of HMDD2, check the cared disease number
  string *mir2_name = new string[495];
  read_Names("datasets/HMDD2/miRNA_Name.txt", mir2_name);
  string *dis2_name = new string[383];
  read_Names("datasets/HMDD2/Disease_Name.txt", dis2_name);
  int *cared = new int[14];
  read_Cared("datasets/HMDD2/cared.txt", cared);
  //get and prepare datasets
  MatrixXd A(Nd,Nr);
  read_matrix_sparse_d("datasets/HMDD2/HMDD2.txt",A);
  MatrixXi AI(Nd,Nr);
  read_matrix_sparse_i("datasets/HMDD2/HMDD2.txt",AI);
  MatrixXd DS0(Nd,Nd);
  read_matrix_dense_d("datasets/HMDD2/DisSim0.txt",DS0);
  MatrixXd DS1(Nd,Nd);
  read_matrix_dense_d("datasets/HMDD2/DisSim1.txt",DS1);
  MatrixXi DW(Nd,Nd);
  read_matrix_dense_i("datasets/HMDD2/DisWgt.txt",DW);
  MatrixXd RS(Nr,Nr);
  read_matrix_dense_d("datasets/HMDD2/miRSim.txt",RS);
  MatrixXi RW(Nr,Nr);
  read_matrix_dense_i("datasets/HMDD2/miRWgt.txt",RW);
  MatrixXd DS(Nd,Nd);
  DS = (DS0+DS1)/2;
  //calculate the scores
  MatrixXd DIS(Nd,Nd);
  MatrixXd RIS(Nr,Nr);
  MatrixXd F(Nd,Nr);
  string path("results/");
  string suffix(".txt");
  //output the result
  MatrixXi HMDD2(Nd,Nr);
  read_matrix_sparse_i("datasets/HMDD2/HMDD2.txt",HMDD2);
  MatrixXi dbDEMC(Nd,Nr);
  read_matrix_sparse_i("datasets/dbDEMC.txt",dbDEMC);
  MatrixXi miR2Disease(Nd,Nr);
  read_matrix_sparse_i("datasets/miR2Disease.txt",miR2Disease);
  for(index=0;index<14;index++){
    i = cared[index];
    for(k=0;k<A.rows();k++){
      A(i,k) = 0;
      AI(i,k) = 0;
    }
    compute_IntSim_Wgt(DW,A,DS,DIS);
    compute_IntSim_Wgt(RW,A.transpose(),RS,RIS);
    CalcF(A,DIS,RIS,F);
    outFile.open((path+dis2_name[i]+suffix).c_str(),ios::out);
    for(j=0;j<Nr;j++){
      outFile<<dis2_name[i]<<"&"<<mir2_name[j]<<"&"<<F(i,j)<<"&";
      if(dbDEMC(i,j)==1) outFile<<"dbDEMC;";
      if(miR2Disease(i,j)==1) outFile<<"miR2Disease;";
      if(HMDD2(i,j)==1) outFile<<"HMDD2;";
      outFile<<endl; 
    }
    outFile.close();
    for(k=0;k<A.rows();k++){
      A(i,k) = HMDD2(i,k);
      AI(i,k) = HMDD2(i,k);
    }  
  }
  delete []mir2_name;
  delete []dis2_name;
  delete []cared;
}
