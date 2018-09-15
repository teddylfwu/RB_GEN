#include "rb_gen.h"

void exit_with_help()
{
  printf(
      "Usage: rb_gen [options] dataFileName featureFileName\n"
      "options:\n"
      "-R number_grids : number of grids needed\n"
      "-S sigma: bandwith of the laplacian kernel\n"
      "-O saveFileName: save grids parameters and feature code-index table to saveFileName\n"
      "-I restoreFileName: restore grids parameters and feature code-index table from restoreFileName\n"
      );
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]){

  int i;
  size_t R = 0; // number of grids drawed
  double sigma = 0; // kernel parameter in Laplacian
  size_t D = 0; //the row number of the codeTable (only need for test data)
  char *saveFileName = NULL; 
  char *restoreFileName = NULL;
  char *datafilename = NULL; // input train or test file name
  char *featurefilename = NULL; // output train or test feature file name
  KernelType kernel = Laplace; // Kernel type

  if(argc == 1){
    exit_with_help();
  }

  for(i=1;i<argc;i++)
  {
    if(argv[i][0] != '-') break;
    ++i;
    switch(argv[i-1][1])
    {
      case 'R': R = atoi(argv[i]); break;
      case 'S': sigma = atof(argv[i]); break;
      case 'O': saveFileName = argv[i]; break;
      case 'I': restoreFileName = argv[i]; break;
      default:
                fprintf(stderr,"unknown option\n\n");
                exit_with_help();
    }
  }
  datafilename = argv[i]; // get input train or test file name
  i++;
  featurefilename = argv[i]; // get output train or test feature file name

  if(R <= 0 && (restoreFileName==NULL))
  {
    fprintf(stderr,"Number of grids needed must be greater than 0!\n\n");
    exit_with_help();
  }

  if(sigma <= 0 && (restoreFileName==NULL))
  {
    fprintf(stderr,"Bandwith of the Laplacian Kernel must be greater than 0!\n\n");
    exit_with_help();
  }

  if(saveFileName && restoreFileName)
  {
    fprintf(stderr,"Can not use -O and -I simultaneously!\n\n");
    exit_with_help();
  }

  if(!(datafilename && featurefilename))
  {
    fprintf(stderr,"Must provide both data and feature file name!\n\n");
    exit_with_help();
  }

  if(argc != i+1) 
    exit_with_help();
  
  double start, end;
  double totaltElapsed, readtElapsed, writetElapsed, feagentElapsed;
  start = CLOCK;
  /* pass 1: read data matrix X (n*d), y (n*1) in LibSVM format */
  Mat X;
  double *y = NULL; 
  if (readSVMfile(datafilename, &X, &y) == -1){
    return -1;
  }
  end = CLOCK;
  readtElapsed = end - start;

  /* pass 1.5: generate R of grid parameters with dimension d:
     (delta_i, mu_i), i=1, ..., d.  
     delta = random_distribution(d*R)/sigma
     mu = rand(1,d*R).*delta */
  double *delta = (double *)malloc(X.d*R*sizeof(double));
  assert(delta != NULL);
  double *mu = (double *)malloc(X.d*R*sizeof(double));
  assert(mu != NULL);
  int *codeTable = NULL;
  size_t *offset = NULL;
  start = CLOCK;
  if(restoreFileName == NULL){ // for training
    printf("X.d=%zu, R=%zu, sigma=%g\n", X.d, R, sigma);
    if (GenerateGridParas(delta, mu, X.d, R, sigma, kernel, 0) == -1) {
      return -1;
    }
  }
  else{ // for testing
    if (readGRIDfile(restoreFileName, &delta, &mu, &offset, &codeTable, &R, X.d, &D) == -1){
      return -1;    
    }
  }
  double gridtElapsed = CLOCK - start;

//  PrintFPMatrix("delta", delta, R, X.d);
//  writeDEN2SVMfile("delta", delta, y, R, X.d);
//  PrintFPMatrix("mu", mu, R, X.d);
//  writeDEN2SVMfile("mu", mu, y, R, X.d);
//  PrintIntMatrix("codeTable", codeTable, D, X.d);
  
  /* pass 2: generate feature code-index table and sparse feature 
     matrix Phi based on R grid parameters (delta, mu) */
  Mat Phi = {Mat_type_CSR, X.n, 0, X.n*R, NULL, NULL, NULL}; 
  if(restoreFileName != NULL){ // for testing
    Phi.d = D;
  }
  Phi.val = (double *)malloc(X.n*R*sizeof(double));
  Phi.rowIndex = (size_t *)malloc((X.n+1)*sizeof(size_t));
  Phi.colIndex = (size_t *)malloc(X.n*R*sizeof(size_t));
  if (!Phi.val || !Phi.rowIndex || !Phi.colIndex) {
    printf("Not enough memory!\n");
    return -1;
  }
  printf("Approx required space: min %.2f GiB spec %.2f GiB\n", (1.0*sizeof(double)*X.n*R + sizeof(size_t)*X.n*R)/1024/1024/1024, 1.0*sizeof(int)*X.n*X.d*R/1024/1024/1024);
  if(restoreFileName == NULL){ // for training
    offset = (size_t*)malloc(sizeof(size_t)*(R+1)); 
    ComputeTrainFeatureMatrix(R, X, delta, mu, offset, &codeTable, &Phi);
  }
  else{ // testing
    ComputeTestFeatureMatrix(R, X, delta, mu, offset, codeTable, &Phi);
  }
  end = CLOCK;
  feagentElapsed = end - start;

  /* pass 3: firstly write random binning feature matrix in LibSVM 
     format and then write grid parameters and feature code-index 
     table into saveFileName for testing                          */
  start = CLOCK;
  if(restoreFileName == NULL){ // for training
    if (writeCSR2SVMfile(featurefilename, &Phi, y) == -1){
      return -1;
    }
    if (writeGRID2file(saveFileName, delta, mu, offset, codeTable, R, X.d, Phi.d) == -1){
      return -1;
    }
  }
  else{ // for testing
    if (writeCSR2SVMfile(featurefilename, &Phi, y) == -1){
      return -1;
    }
  }
  end = CLOCK;
  writetElapsed = end - start;
  totaltElapsed = readtElapsed + writetElapsed + feagentElapsed;
  printf("Read: %g Write: %g FeaGen: %g Total: %g\n", readtElapsed, writetElapsed, feagentElapsed, totaltElapsed);

  if(X.val != NULL){
    free(X.val);
    free(X.rowIndex);
    free(X.colIndex);
  }
  if(Phi.val != NULL){
    free(Phi.val);
    free(Phi.rowIndex);
    free(Phi.colIndex);
  }
  free(y);
  free(delta);
  free(mu);
  free(codeTable);
  free(offset);

  return 0;

}
