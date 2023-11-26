#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

// float d[] = {0, 1, 0, 1};
// Mat m = {.rows = 4, .cols = 1, .es = d};

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride; 
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
 //used to pass values in the objects of struct
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a); //42:44
void mat_sig(Mat m);
void mat_fill(Mat m, float x);
void mat_print(Mat m,const char *name);

#define MAT_PRINT(m) mat_print(m, #m) // 1:19:45

#endif // NN_H_

#ifdef NN_IMPLEMENTATION /*/*//*/*////////////////////

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}
void mat_dot(Mat dst, Mat a, Mat b)
{
    
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);
    
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j< dst.cols; ++j){
               MAT_AT(dst, i, j) = 0;
               for(size_t k = 0; k<n; ++k){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k)* MAT_AT(b, k, j);
               } 
            }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j< dst.cols; ++j){
                MAT_AT(dst, i, j) += MAT_AT(a, i, j);
            }//1:02:07
    }
}

Mat mat_row(Mat m, size_t row)
{
     return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0), //*d*b*t 1:43:28
    };
}

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i<dst.rows; ++i){
        for(size_t j = 0; j<dst.cols; ++j){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sig(Mat m)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j<m.cols; ++j){
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));  
        }
    }
}

void mat_print(Mat m, const char *name)
{
    printf("%s = [\n", name);
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j<m.cols; ++j){
            printf("    %f", MAT_AT(m, i, j)); //53:25
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_fill(Mat m, float x){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j<m.cols; ++j){
                MAT_AT(m, i, j) = x;
        }//1:03:28
    }
}

void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j<m.cols; ++j){
                MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }//1:00:55
    }
}

#endif // NN_IMPLEMENTATION
