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
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride; 
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
 //used to pass values in the objects of struct
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a); //42:44
void mat_sig(Mat m);
void mat_fill(Mat m, float x);
void mat_print(Mat m,const char *name, size_t padding);

#define MAT_PRINT(m) mat_print(m, #m, 0) // 1:19:45

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);

void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);


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

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name); //creates empty string of size=padding hence resulting space.
    for(size_t i = 0; i < m.rows; ++i){
        printf("%*s    ", (int) padding, "");
        for(size_t j = 0; j<m.cols; ++j){
            printf("%f ", MAT_AT(m, i, j)); //53:25
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
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
//size_t arch[] = {2, 2, 1};
//NN nn = nn_alloc(arch, ARRAY_LEN(arch));

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;
    
    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);

    nn.as = NN_MALLOC(sizeof(*nn.as)*nn.count + 1);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for(size_t i = 1; i < arch_count; ++i){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]); //****2~22~14
        nn.as[i] = mat_alloc(1, arch[i]);
    }
    return nn;   
}

void nn_print(NN nn, const char *name)
{   
    char buf[256];
    printf("%s = [\n", name);
    for(size_t i = 0; i< nn.count; ++i){
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for(size_t i = 0; i<nn.count; ++i){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}
void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}
void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
   // NN_ASSERT(NN_OUTPUT(n).cols == to.cols);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for(size_t i = 0; i<n; ++i){
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);
        //NN_OUTPUT(nn) == mat_row(to, i);
        for(size_t j = 0; j<to.cols; ++j){
            MAT_AT(NN_OUTPUT(g), 0, j) = 
            MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }
        for(size_t l = nn.count; l>0; --l){
           for(size_t j = 0; j<nn.as[l].cols; ++j){
            float a = MAT_AT(nn.as[l], 0, j);
            float da = MAT_AT(g.as[l], 0, j);

            MAT_AT(g.bs[l-1],0, j) += 2*da*a*(1-a);

            for(size_t k = 0; k < nn.as[l-1].cols; ++k){
                // j = weight matrix col
                // k = weight matrix row
                float pa = MAT_AT(nn.as[l-1], 0, k);
                float w = MAT_AT(nn.as[l-1], k, j);
                MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1 - a)*w;
             }
           }        
        }
    }
    for(size_t i = 0; i < g.count; ++i){
        for(size_t j = 0; j < g.ws[i].rows; ++j){
            for(size_t k = 0; k < g.ws[i].cols; ++k){
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for(size_t j = 0; j < g.count; ++j){
            for(size_t k = 0; k < g.ws[i].rows; ++k){
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}
#endif // NN_IMPLEMENTATION
