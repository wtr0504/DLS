#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void matmul(const float* A,const float *B,float *C, size_t m,size_t n,size_t k,
            bool A_T,bool B_T){
  /**
  A shape : 
  A_T false : m * k
  A_T true : k * m
  B shape : 
  B_T false : k * n
  B_T true : n * k
  */
  std::memset(C, 0, sizeof(float) * m * n);

  for(size_t i = 0;i < m;i++){
    for(size_t j = 0;j < n;j++){
      float sum = 0.;
      for(size_t p = 0;p < k;p++){
        size_t a_idx = A_T ? (p * m + i) : (i * k + p);
        size_t b_idx = B_T ? (j * k + p) : (p * n + j);
        sum += A[a_idx] * B[b_idx];
      }
      C[i * n + j] = sum;
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

     /**
      for i in range(0,X.shape[0],batch):
        X_batch = X[i : (i + batch),:]
        y_batch = y[i : (i + batch)]
        tmp = np.exp(X_batch @ theta)
        Z = tmp / np.sum(tmp,1,keepdims=True)
        I_y = np.zeros((batch,theta.shape[1]))
        I_y[np.arange(batch),y_batch] = 1
        grad = X_batch.T @ (Z - I_y) / batch
        theta -= grad * lr
     */

    /// BEGIN YOUR CODE
    float *tmp = (float*)malloc(sizeof(float) * batch * k); // batch * k
    float *tmp_sum = (float*)malloc(sizeof(float) * batch);
    float *grad = (float*)malloc(sizeof(float) * n * k);
    for(size_t i = 0;i < m;i += batch){
      size_t real_batch = std::min(m - i,batch);
      const float *X_batch = X + i * n;
      const unsigned char* y_batch = y + i;
      matmul(X_batch,theta,tmp,real_batch,k,n,false,false);

      for(size_t j = 0;j < real_batch;j++){
        tmp_sum[j] = 0.;
        for(size_t p = 0;p < k;p++){
          size_t idx = j * k + p;
          tmp[idx] = exp(tmp[idx]);
          tmp_sum[j] += tmp[idx];
        }
      }
      for(size_t j = 0;j < real_batch;j++){
        for(size_t p = 0;p < k;p++){
          size_t idx = j * k + p;
          tmp[idx] = tmp[idx] / tmp_sum[j];
        }
        tmp[j * k + y_batch[j]] -= 1;
      }
      matmul(X_batch,tmp,grad,n,k,real_batch,true,false);


      for(size_t j = 0;j < n;j++){
        for(size_t p = 0; p < k;p++){
          theta[j * k + p] -= grad[j * k + p] / real_batch * lr;
        }
      }
    }
    free(tmp);
    free(tmp_sum);
    free(grad);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
