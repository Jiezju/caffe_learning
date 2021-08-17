#include "caffe_helper.h"


int main()
{
	string net_file = "slice.prototxt";      //prototxtÎÄ¼þ
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));

	string scale_layer_name = "layer2-scale";

	int layer_idx = CaffeHelper::get_layer_index(net, scale_layer_name);

	//boost::shared_ptr<Layer<float>> scale_layer = net->layers()[layer_idx];
	//vector<boost::shared_ptr<Blob<float>>> scale_blobs = scale_layer->blobs();

	// init input
	float* input_ptr = CaffeHelper::_generate_random_input(net->input_blobs()[0]->count());

	//// init weight
	//float* weight_ptr = scale_blobs[0]->mutable_cpu_data();
	//float* bias_ptr = scale_blobs[1]->mutable_cpu_data();

	//float* muta_weight = CaffeHelper::_generate_random_input(scale_blobs[0]->count());
	//float* muta_bias = CaffeHelper::_generate_random_input(scale_blobs[1]->count());

	//memcpy(weight_ptr, muta_weight, scale_blobs[0]->count()*sizeof(float));
	//memcpy(bias_ptr, muta_bias, scale_blobs[1]->count()*sizeof(float));

	CaffeHelper::caffe_forward(net, input_ptr);

	vector<Blob<float>*> output_blobs = net->output_blobs();

	
	delete[] input_ptr;

	return 0;
}