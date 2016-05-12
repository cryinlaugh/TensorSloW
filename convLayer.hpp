#ifndef CNNCONV_H_
#define CNNCONV_H_

extern "C"{
  #include "blob.h"
  #include "cnnConvolutionImp.h"
}
#include <cstdlib>
#include <math.h>


class convLayer{
public:
  Weight Wold;
  float* bold;
  Features convolvedFeatures;

  Tensor* input, * output;
  Tensor W, dW, b, db;
  Tensor col_data, col_data_full;

  void setUp(Tensor* const in, Tensor* const out, int K1, int K2){
      W.K1 = K1;
      W.K2 = K2;
      W.Ni = in->N;
      W.No = out->N;
      W.size = W.K1*W.K2*W.Ni*W.No;

      dW.K1 = K1;
      dW.K2 = K2;
      dW.Ni = in->N;
      dW.No = out->N;
      dW.size = dW.K1*dW.K2*dW.Ni*dW.No;
      
      b.N = out->N; //?
      b.size = b.N; //unexpanded
      db.N = b.N;
      db.size = db.N;

      W.data = (real*)malloc(REALSIZE*W.size);
      dW.data = (real*)malloc(REALSIZE*dW.size);
      b.data = (real*)malloc(REALSIZE*b.size);
      db.data = (real*)malloc(REALSIZE*db.size);

      input = in;
      output = out;


  }
  void setWeightFromFile(char* const Wfile, char * const bfile){
      __TensorLoadWeight(&W, Wfile);
      __TensorLoadB(&b, bfile);   
  }

    //init weight's randomly between [-sqrt(6/(fanout+fanin)), sqrt(6/(fanout+fanin))]
    //init bias to 0
    void initWeightRandom(){
        int fan_out = W.K1*W.K2*W.No;
        int fan_in = W.K1*W.K2*W.Ni;
        real bound = sqrt(6/double(fan_out+fan_in));
        __TensorDataInitRandom(&W, -bound, bound);
        memset(b.data, 0, REALSIZE*b.size);
    }
    
  void setDown(){
    free(W.data);
    free(b.data);
    free(dW.data);
    free(db.data);
  }

  void forward(int profile){
      //__convForward(&col_data, &W, &b, output);
      if(true){
        __forward_im2col(input, &W, &col_data);
        __convForward(&col_data, &W, &b, output);
      }else{
        __convForward2(input, &W, &b, output);
      }
      printf("forward is ok\n");
      //__TensorPrint(output, "./log/ftr.txt");
      //__TensorCheckRes("./log/ftr.txt", "../testdata/convfp/convolvedFeatures.txt");
  }

  void backward(){
      __backward_im2col(input, &W, &col_data_full);
      printf("conv bp_im2col is ok\n");
      __TensorPrint(&col_data_full, "./log/col_data_full.txt");
      __TensorCheckRes("./log/col_data_full.txt", "../testdata/convbp/col_data.txt");
      __convBackward(&col_data_full, &W, &b, output);
      __TensorPrint(output, "./log/bpconveddata.txt");
      __TensorCheckRes("./log/bpconveddata.txt", "../testdata/convbp/res.txt");
  }

};


#endif
