#include "feature_matrix.h"
#include <assert.h>
#include <unistd.h>

typedef struct List { void *p; size_t size; struct List *next; } List;

/* Provided two implementation of hash table. One is from glib and
   the other is from STL. By default, it is used glib if the code
   is compiled with a C compiler. Otherwise STL is used. If the
   macro RB_USE_GLIB is defined, glib implementation is used
   regardless of the compiler.
*/

#if defined(RB_USE_GLIB) || !defined(__cplusplus)
#include <glib.h>

guint my_hash_vector_int(gconstpointer v_) {
  int *v = (int*)v_;
  int len = v[0];
  v++;
  int i;
  guint h = 5381;
  for (i=0; i<len; i++) h = h*33 + v[i];
  return h;
}

gboolean my_equal_vector_int(gconstpointer v_, gconstpointer w_) {
  int *v = (int*)v_, *w = (int*)w_;
  int len = v[0];
  int i;
  for (i=0; i<=len; i++)
    if (v[i] != w[i]) return FALSE;
  return TRUE;
}

typedef GHashTable *RB_HASH;
#define RB_HASH_NEW g_hash_table_new(my_hash_vector_int, my_equal_vector_int)
#define RB_HASH_DELETE(HT) g_hash_table_destroy((HT))
#define RB_HASH_LOOKUP(HT, CODE) \
  (int*)g_hash_table_lookup((HT), (CODE))
#define RB_HASH_INSERT(HT, CODE, VALUE) \
    g_hash_table_insert((HT), (CODE), (VALUE))

#else /* RB_USE_GLIB */

struct my_hash_vector_int {
  size_t operator()(const int * v) const  {
    int len = v[0];
    int i;
    size_t h = 5381;
    for (i=0; i<len; i++) h = h*33 + v[i+1];
    return h;
  }
};

struct my_equal_vector_int {
  bool operator()(const int *v, const int *w) const {
    int len = v[0];
    int i;
    for (i=0; i<=len; i++)
      if (v[i] != w[i]) return false;
    return true;
  }
};

/* If the compiler don't support C++ 2011, then it is
   included the TR1 headers. */

#if __cplusplus < 201103L
# include <tr1/unordered_map>
typedef std::tr1::unordered_map<int*, int*, my_hash_vector_int, my_equal_vector_int> Hash;
#else
# include <unordered_map>
typedef std::unordered_map<int*, int*, my_hash_vector_int, my_equal_vector_int> Hash;
#endif
typedef Hash* RB_HASH;
#define RB_HASH_NEW new Hash(128)
#define RB_HASH_DELETE(HT) delete (HT)
inline int* RB_HASH_LOOKUP(Hash* hash, int *code) {
  Hash::iterator s = hash->find(code);
  return s != hash->end() ? s->second : (int*)NULL;
}
#define RB_HASH_INSERT(HT, CODE, VALUE) \
    (HT)->insert(std::make_pair((CODE), (VALUE)))

#endif

int compareCodeHash(const int *code, RB_HASH hashTable, size_t d, int *codeKeyNew, int *valueNew);

/* we use some reasonably large number to estimate practical upper  *
 * bound of expected number of non-empty bins per grid. However,    *
 * the theoretical upper bound is n, which however is rarely to be  *
 * true since it is largely overfitting. One simple technique to    *
 * estimate the number of nonempty bins per grid is to dynamically  *
 * average the number of nonempty bins produced by a small portion  *
 * of grids. Realloc() function is called to adjust the memory size */
int ComputeTrainFeatureMatrix(size_t R, Mat X, double *delta, 
    double *mu, SIZET *offset, int **codeTable, Mat *Phi){

  double kSqrtInvR = sqrt(1.0/R);
  size_t numNEBin = 128; // empirical estimation of num of non-empty bins per grid 

  List *listOfCodeTableLocal = (List*)malloc(sizeof(List)*R);

  RB_INDEX *rowStart = Phi ? Phi->rowIndex : NULL;
  RB_INDEX *feaIndex = Phi ? Phi->colIndex : NULL;
  RB_VAL *feaVal = Phi ? Phi->val : NULL;

  /* Set rowStart */

  if (rowStart) {
    size_t i;
    for (i=0; i<=X.n; i++)
      rowStart[i] = i*R;
  }

  size_t i;
#ifdef _OPENMP
# pragma omp parallel for firstprivate(numNEBin)
#endif
  for(i=0; i<R; i++){
    RB_HASH hashTable = RB_HASH_NEW;
    int *code = (int *)calloc(X.d, sizeof(int));
    assert(code != NULL);
    int *nextCode = (int *)calloc((X.d+1)*numNEBin, sizeof(int));
    assert(nextCode !=NULL);
    List l0 = {nextCode, (size_t)numNEBin, NULL};
    listOfCodeTableLocal[i] = l0;
    List *lastCodeTableLocal = &listOfCodeTableLocal[i];
    int *hashValues = (int *)calloc(X.n+1, sizeof(int));
    assert(hashValues !=NULL);

    /* Compute code for a row full of zeros */
    int *code0 = (int*)malloc(sizeof(int)*X.d);
    size_t j;
    for (j=0; j<X.d; j++) code0[j] = floor(-mu[X.d*i+j]/delta[X.d*i+j]);

    size_t nonEmptyBinsCount = 0; //count number of non-empty bins per grid
    for(j=0; j<X.n; j++){
      if (X.type == Mat_type_CSR) {
        size_t k;
        for (k=0; k<X.d; k++) code[k] = code0[k];
      }
      ComputeBinNum(X, j, code, delta+X.d*i, mu+X.d*i);
      int ret = compareCodeHash(code, hashTable, X.d, nextCode, hashValues+nonEmptyBinsCount);
      if(ret == -1) {
        if (feaIndex) feaIndex[i+j*R] = nonEmptyBinsCount;
        nonEmptyBinsCount++;
        hashValues[nonEmptyBinsCount] = nonEmptyBinsCount;
        nextCode += X.d+1;
        if((size_t)(nextCode-(int*)lastCodeTableLocal->p) > (size_t)(lastCodeTableLocal->size-1)*(size_t)(X.d+1)){
          numNEBin = MIN(numNEBin*2, X.n); //double meomory need
          int codeTableLocalSize = MIN(lastCodeTableLocal->size*2, X.n);
          List *p = (List*)malloc(sizeof(List));
          lastCodeTableLocal->next = p; lastCodeTableLocal = p;
          p->next = NULL;
          p->size = codeTableLocalSize;
          p->p = nextCode = (int *)malloc(sizeof(int)*(X.d+1)*codeTableLocalSize);
          assert(nextCode !=NULL);
        }
      }
      else{
        if (feaIndex) feaIndex[i+j*R] = ret; //index of same code in codeTable
      }
      if (feaVal) feaVal[i+j*R] = 1.0*kSqrtInvR; // all feature value is always 1/sqrt(R)
    }

    offset[i+1] = nonEmptyBinsCount;
    RB_HASH_DELETE(hashTable);
    free(hashValues);
    free(code);
    free(code0);
  }

  /* Compute offset */
  offset[0] = 0;
  for(i=0; i<R; i++) offset[i+1] += offset[i];

  /* Update feaIndex */

  if (feaIndex) {
#ifdef _OPENMP
# pragma omp parallel for
#endif
    for(i=0; i<R; i++) {
      const size_t offseti = offset[i];
      size_t j;
      for(j=0; j<X.n; j++){
          feaIndex[i+j*R] += offseti;
      }
    }
  }

  /* Create codeTable */

  *codeTable = (int *)MALLOC(X.d*offset[R]*sizeof(int));
  assert(*codeTable !=NULL);

#ifdef _OPENMP
# pragma omp parallel for
#endif
  for(i=0; i<R; i++) {
    /* Copy the local code tables back into the big code table */
    size_t j=0;
    List *p = &listOfCodeTableLocal[i];
    size_t nonEmptyBinsCount = offset[i+1] - offset[i];
    while (p != NULL) {
      copycodeTableLocalTocodeTable((int*)p->p, *codeTable+(offset[i]+j)*X.d, MIN(p->size, nonEmptyBinsCount-j), X.d);
      j += p->size;
      p = p->next;
    }

    /* Destroy the local code tables */
    p = listOfCodeTableLocal[i].next;
    free(listOfCodeTableLocal[i].p);
    while(p) {free(p->p); p = p->next;}
  }
  free(listOfCodeTableLocal);

  if (Phi) Phi->d = offset[R];

  //printf("compu: %g compa: %g copycode: %g copytable: %g\n", compBinTime, compareBinTime, copyBinTime, copyBinTableTime);

//  printf("phi.n=%zu, phi.d=%zu, phi.nnz=%zu\n", Phi->n, Phi->d, Phi->nnz);
//  PrintSizetMatrix("offset", (*offset), 1, R+1);
//  PrintSizetMatrix("Phi->rowStart", Phi->rowStart, 1, X.n+1);
//  PrintSizetMatrix("Phi->colIndex", Phi->colIndex, X.n, R);
//  PrintFPMatrix("Phi->val", Phi->val, X.n, R);
//  PrintIntMatrix("codeTable", *codeTable, Phi->d, X.d);

  return 0;

}


int ComputeTestFeatureMatrix(size_t R, Mat X, double *delta, double *mu, SIZET *offset, int *codeTable, Mat *Phi){

  size_t i;
  double kSqrtInvR = sqrt(1.0/R);
  RB_INDEX *rowStart = Phi->rowIndex;
  RB_INDEX *feaIndex = Phi->colIndex;
  RB_VAL *feaVal = Phi->val;
  rowStart[0] = 0; // row index starts from 0 by convention 

#ifdef _OPENMP
# pragma omp parallel for
#endif
  for(i=0; i<R; i++){
    RB_HASH hashTable = RB_HASH_NEW;
    // rebuild train hash table for test in each grid
    int *keysHash = (int*)malloc(sizeof(int)*(X.d+1)*(offset[i+1]-offset[i]));
    int *valuesHash = (int*)malloc(sizeof(int)*(offset[i+1]-offset[i]));
    size_t j;
    for (j=0; j<offset[i+1]-offset[i]; j++) {
      valuesHash[j] = offset[i]+j;
      compareCodeHash(codeTable+(offset[i]+j)*X.d, hashTable, X.d, keysHash + j*(X.d+1), valuesHash + j);
    }
    int *code = (int *)calloc(X.d, sizeof(int));
    assert(code != NULL);

    /* Compute code for a row full of zeros */
    int *code0 = (int*)malloc(sizeof(int)*X.d);
    assert(code0 != NULL);
    for (j=0; j<X.d; j++) code0[j] = floor(-mu[X.d*i+j]/delta[X.d*i+j]);

    for(j=0; j<X.n; j++){
      if (X.type == Mat_type_CSR) {
        size_t k;
        for (k=0; k<X.d; k++) code[k] = code0[k];
      }
      ComputeBinNum(X, j, code, delta+X.d*i, mu+X.d*i);
      int ret = compareCodeHash(code, hashTable, X.d, NULL, NULL);
      if(ret == -1){
        // new feature in testing is meaningless therefore set to 0
        feaIndex[i+j*R] = offset[i]+1;; // index must be (offset[i] offset[i+1])
        feaVal[i+j*R] = 0.0; //must be set to 0
      }
      else{
        feaIndex[i+j*R] = ret; //index of same code in codeTable
        feaVal[i+j*R] = 1.0*kSqrtInvR;
      }
      rowStart[j+1] = (j+1)*R;
    }
    free(keysHash);
    free(valuesHash);
    free(code);
    free(code0);
    RB_HASH_DELETE(hashTable);
  }
  Phi->d = offset[R];
//  printf("phi.n=%zu, phi.d=%zu, phi.nnz=%zu\n", Phi->n, Phi->d, Phi->nnz);
//  PrintSizetMatrix("Phi->rowStart", Phi->rowStart, 1, X.n+1);
//  PrintSizetMatrix("Phi->colIndex", Phi->colIndex, X.n, R);
//  PrintFPMatrix("Phi->val", Phi->val, X.n, R);
//  PrintIntMatrix("codeTable", codeTable, Phi->d, X.d);

  return 0;

}


/* Since we don't know how many non-zero features in a sample, we 
   need keep tracking the starting non-zero features in X with cnz. 
   To compute the code for a sample, we need consider three cases: 
   1) missing features in the middle of a sample 
   2) missing features in the end of a sample
   3) full features in a sample                                     */
void ComputeBinNum(Mat X, size_t row, int *code, double *delta, double *mu){
  if (X.type == Mat_type_dense) {
    int i;
    int d = X.d;
    RB_VAL *X_row = &X.val[d*row];
    for(i=0; i<d; i++){
        code[i] = floor((X_row[i]-mu[i])/delta[i]);
    }
  } 
  else if (X.type == Mat_type_CSR) {
    int nnz_in_row = (int)(X.rowIndex[row+1] - X.rowIndex[row]);
    RB_INDEX *col = &X.colIndex[X.rowIndex[row]];
    RB_VAL *val = &X.val[X.rowIndex[row]];
    int i;
    for(i=0; i<nnz_in_row; i++) {
      code[col[i]] = floor((val[i]-mu[col[i]])/delta[col[i]]);
    }
  }
}

/* If new code is not in corresponding segment of codeTable, return -1; 
   Otherwise, return index of the same code in codeTableLocal. */
int compareCode(int *code, int *codeTable, size_t start, size_t end, size_t d, size_t *nonEmptyBinsCount){
    
  int ret = -1; // assume code is not in codeTableLocal
  size_t i, j;
  for(i=start; i<end; i++){
    for(j=0; j<d; j++){
      if(code[j] != codeTable[i*d + j]){
        /* if last indicator is different, reset j to avoid confusion*/
        if(j == d-1){
          j = 0; 
        }
        break;
      }
    }
//    printf("j = %zu\n", j);
    if(j == d){
      ret = i; // find code in corresponding segment of codeTable!
      break;
    }
  }
  if(ret == -1){
    (*nonEmptyBinsCount)++;
  }

  return ret;

}

/* If new code is not in hashTable, return -1 and insert code into hashTable 
   for training stage; Otherwise, return index of the same code in hashTable. 
   Note: trainFlag == 1 indicating training stage, otherwise testing stage */
int compareCodeHash(const int *code, RB_HASH hashTable, size_t d, int *codeKeyNew, int *valueNew) {
  
  int codeKey[d+1];
  int *codeKeyFound;
  size_t i;
  codeKey[0] = d;
  for (i=0; i<d; i++)
    codeKey[i+1] = code[i];
  codeKeyFound = RB_HASH_LOOKUP(hashTable, codeKey);

  /* If found, return the index */
  if (codeKeyFound){
    return *codeKeyFound;
  }
  
  /* If not, insert the code into hashTable in training and return -1 */
  if (codeKeyNew && valueNew){
    memcpy(codeKeyNew, codeKey, sizeof(int)*(d+1));
    RB_HASH_INSERT(hashTable, codeKeyNew, valueNew); 
  }
  
  return -1;

}
 
void copyCodeTocodeTable(int *code, int *codeTableAppend, size_t d){

  size_t i;
  for(i=0; i<d; i++){
    codeTableAppend[i] = code[i];
  }

}

void copycodeTableLocalTocodeTable(int *codeTableLocal, int *codeTable, size_t end, size_t d){

  size_t i, j;
  for(i=0; i<end; i++){
    for(j=0; j<d; j++){
      codeTable[i*d+j] = codeTableLocal[i*(d+1)+j+1];
    }
  }

}
