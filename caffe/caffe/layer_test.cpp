//
// Created by bright on 2021/10/6.
//

/*  单层 layer 测试 */

#include "../caffe_helper.h"
#include <caffe/layers/relu_layer.hpp>

int main()
{
//    ReLUParameter relu_p;
    LayerParameter relu_p;

    ReLUParameter relu_ps = relu_p.relu_param();
    float ori_slope = relu_ps.negative_slope();
    relu_ps.set_negative_slope(0.1);
    float new_slope = relu_ps.negative_slope();
    ReLULayer<float> relu_layer(relu_p);

    // forward 函数无法调用， 所以转为父类
    Layer<float>* layer = static_cast<Layer<float>*>(&relu_layer);
    vector<int> shape{1,2,3};
    Blob<float> bottom(shape);
    Blob<float> top(shape);

    CaffeHelper::generate_random_blob(bottom);
    const float* input_ptr = bottom.cpu_data();
    float* output_ptr = top.mutable_cpu_data();

    vector<Blob<float>*> bts;
    vector<Blob<float>*> tps;

    bts.push_back(&bottom);
    tps.push_back(&top);

    layer->Forward(bts, tps);

    return 0;
}