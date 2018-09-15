#include "writefile.h"

/*write dense matrix to LibSVM format in row major*/
int writeDEN2SVMfile(const char *filename, double *X, double *y, size_t n, size_t d) {

  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("WriteData. Error: Cannot open file %s. \n", filename);
    return -1;
  }

  /* Note: LibSVM index starts from 1, use size_t prevent overflow */
  size_t i, j;
  for (i=0; i<n; i++){
    fprintf(fp, "%g", y[i]);
    for (j=0; j<d; j++){
      if (X[d*i+j] != 0){
        fprintf(fp, " %zu:%.16g", j+1, X[d*i+j]);
      }
    }
    fprintf(fp, " \n");
  }

  fclose(fp);
  return 0;

}


/*write sparse matrix with CSR format to LibSVM format*/
int writeCSR2SVMfile(const char *filename, Mat *X, double *y) {

  assert(X->type == Mat_type_CSR);
  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("WriteData. Error: Cannot open file %s. \n", filename);
    return -1;
  }

  /* Note: LibSVM index starts from 1, use size_t prevent overflow */
  size_t i, j;
  for (i=0; i<X->n; i++){
    fprintf(fp, "%g", y[i]);
    for (j=X->rowIndex[i]; j<X->rowIndex[i+1]; j++){
      fprintf(fp, " %zu:%.16g", X->colIndex[j]+1, X->val[j]);
    }
    fprintf(fp, " \n");
  }

  fclose(fp);
  return 0;

}


int writeGRID2file(const char *filename, double *delta, double *mu, size_t *offset, int *codeTable, size_t R, size_t d, size_t D){

  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("WriteData. Error: Cannot open file %s, \n", filename);
    return -1;
  }
  
  fprintf(fp, "R:%zu\n", R);
  fprintf(fp, "d:%zu\n", d);
  fprintf(fp, "D:%zu\n", D);
  size_t i, j;
  fprintf(fp, "delta:\n");
  for(i=0; i<R; i++){
    for(j=0; j<d; j++){
      fprintf(fp, "%.16g ", delta[i*d+j]);
    }
    fprintf(fp, "\n");
  }
  
  fprintf(fp, "mu:\n");
  for(i=0; i<R; i++){
    for(j=0; j<d; j++){
      fprintf(fp, "%.16g ", mu[i*d+j]);
    }
    fprintf(fp, "\n");
  }

  fprintf(fp, "offset:\n");
  for(i=0; i<R+1; i++){
      fprintf(fp, "%zu ", offset[i]);
  }
  fprintf(fp, "\n");
  
  fprintf(fp, "codeTable:\n");
  for(i=0; i<D; i++){
    for(j=0; j<d; j++){
      fprintf(fp, "%d ", codeTable[i*d+j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  return 0;
}

