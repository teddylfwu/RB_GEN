#include "readfile.h"

int readSVMfile(const char *filename, Mat *X, double **y){

  // use size_t to prevent overflow when (*n)*d > int_max
  if (VerifyFileFormat(filename, &X->d, &X->n, &X->nnz) == -1) {
    return -1;
  }
  printf("d = %zu, n = %zu, nnz = %zu\n", X->d, X->n, X->nnz);

  FILE *fp = NULL;
  fp = fopen(filename, "r");

  X->type = Mat_type_CSR;

  // Allocate memory for X and y
  X->val = (double *)malloc(X->nnz*sizeof(double));
  assert(X->val != NULL);
  X->rowIndex = (size_t *)malloc(X->nnz*sizeof(size_t));
  assert(X->rowIndex != NULL);
  X->colIndex = (size_t *)malloc(X->nnz*sizeof(size_t));
  assert(X->colIndex != NULL);
  (*y) = (double *)malloc(X->n*sizeof(double));
  assert((*y) != NULL);

  // Read the file again and populate X and y
  size_t i = 0;
  size_t numi = 0;
  double numf = 0.0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  int read = 0;
  size_t len = 0;
  size_t cnl = 0; // count number of line
  size_t cnz = 0; // count number of non-zeros 
  X->rowIndex[0] = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    cnl++;
    size_t cnt = 0; // count number of tokens in one line
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL || !strcmp(subtoken,"\n")) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt == 1) {
        (*y)[i] = atof(subtoken);
      }
      else if (cnt%2 == 0) {
        numi = atoi(subtoken);
      }
      else {
        numf = atof(subtoken);
        X->val[cnz] = numf; 
        X->colIndex[cnz] = numi-1; //LibSVM index starts from 1
        cnz++;
      }
    }
    X->rowIndex[cnl] = cnz;
    i++;
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  return 0;

}


int VerifyFileFormat(const char *filename, size_t *d, size_t *n, size_t *nnz){

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("VerifyFileFormat. Error: Cannot open file %s.\n", filename);
    return -1;
  }

  *n = 0;
  *d = 0;
  *nnz = 0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  int read = 0;
  size_t len = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    (*n)++;
    size_t cnt = 0, maxdim = 0;
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL || !strcmp(subtoken,"\n")) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt%2 == 0) { // Verify format
        (*nnz)++;
        size_t num = (size_t)atoi(subtoken);
        if (maxdim > num) {
          printf("VerifyFileFormat. Error: Indices in line %zu are \
              not in the ascending order. Stop reading data.\n", *n);
          fclose(fp);
          return -1;
        }
        else {
          maxdim = num;
          if (maxdim > *d)
            *d = maxdim;
        }
      }
    }
    if (cnt%2 != 1) { // Verify format
      printf("VerifyFileFormat. Error: Line %zu does not conform with \
          a LibSVM format. Stop reading data.\n", *n);
      fclose(fp);
      return -1;
    }
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  if (*n == 0) {
    printf("VerifyFileFormat. Error: Empty file!\n");
    return -1;
  }

  return 0;

}


int readGRIDfile(const char *filename, double **delta, double **mu, size_t **offset, int **codeTable, size_t *R, size_t d, size_t *D){

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: Cannot open file %s.\n", filename);
    fclose(fp);
    return -1;
  }
  
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  int read = 0;
  size_t len = 0;
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    if(strcmp(subtoken, "R") == 0){
      subtoken = strtok_r(NULL, ":", &saveptr);
      *R = atoi(subtoken);
      printf("%zu\n", *R);
    }
    else{
      printf("Error: R info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  size_t d_test = 0;
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    if(strcmp(subtoken, "d") == 0){
      subtoken = strtok_r(NULL, ":", &saveptr);
      d_test = atoi(subtoken);
      printf("%zu\n", d_test);
      if(d_test != d){
        printf("Error: test dimension is not consistent with train dimension %s.\n", filename);
      }
    }
    else{
      printf("Error: d info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    if(strcmp(subtoken, "D") == 0){
      subtoken = strtok_r(NULL, ":", &saveptr);
      *D = atoi(subtoken);
      printf("%zu\n", *D);
    }
    else{
      printf("Error: D info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  (*delta) = (double *)malloc(d*(*R)*sizeof(double));
  assert((*delta) != NULL);
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    if(strcmp(subtoken, "delta") == 0){
      size_t i = 0;
      while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
        for(str = line; ; str = NULL){
          subtoken = strtok_r(str, " ", &saveptr);
          if (subtoken == NULL || !strcmp(subtoken,"\n")) {
            break;
          }
          (*delta)[i] = atof(subtoken);
          i++;
        }
        if(i == d*(*R)) break;
      }
    }
    else{
      printf("Error: delta info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  (*mu) = (double *)malloc(d*(*R)*sizeof(double));
  assert((*mu) != NULL);
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    printf("%s\n", subtoken);
    if(strcmp(subtoken, "mu") == 0){
      size_t i = 0;
      while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
        for(str = line; ; str = NULL){
          subtoken = strtok_r(str, " ", &saveptr);
          if (subtoken == NULL || !strcmp(subtoken,"\n")) {
            break;
          }
          (*mu)[i] = atof(subtoken);
          i++;
        }
        if(i == d*(*R)) break;
      }
    }
    else{
      printf("Error: mu info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  (*offset) = (size_t *)malloc(((*R)+1)*sizeof(size_t));
  assert((*offset) != NULL);
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    printf("%s\n", subtoken);
    if(strcmp(subtoken, "offset") == 0){
      size_t i = 0;
      while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
        for(str = line; ; str = NULL){
          subtoken = strtok_r(str, " ", &saveptr);
          if (subtoken == NULL || !strcmp(subtoken,"\n")) {
            break;
          }
          (*offset)[i] = atof(subtoken);
          i++;
        }
        if(i == (*R)+1) break;
      }
    }
    else{
      printf("Error: offset info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }
  (*codeTable) = (int *)malloc(d*(*D)*sizeof(int));
  assert((*codeTable) != NULL);
  if((read = getline(&line, &len, fp)) != -1){
    str = line;
    subtoken = strtok_r(str, ":", &saveptr);
    printf("%s\n", subtoken);
    if(strcmp(subtoken, "codeTable") == 0){
      size_t i = 0;
      while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
        for(str = line; ; str = NULL){
          subtoken = strtok_r(str, " ", &saveptr);
          if (subtoken == NULL || !strcmp(subtoken,"\n")) {
            break;
          }
          (*codeTable)[i] = atof(subtoken);
          i++;
        }
        if(i == d*(*D)) break;
      }
    }
    else{
      printf("Error: codeTable info is missing in file %s.\n", filename);
      fclose(fp);
      return -1;
    }
  }

  fclose(fp);
   
  return 0; 

}

