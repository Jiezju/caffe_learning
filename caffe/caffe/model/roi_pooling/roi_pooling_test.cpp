#include "caffe_helper.h"


int main()
{
	string net_file = "roi_pooling.prototxt";      //prototxt文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));
	// int arr = _generate_random_input();

	Blob<float>* input_rois = net->input_blobs()[0];
	Blob<float>* input_conv = net->input_blobs()[1];

	vector<double> conv_vector;
	vector<double> rois_vector;
	CaffeHelper::load_data_from_txt(conv_vector, "D:\\caffe\\caffe\\caffe\\caffe\\conv.txt");
	CaffeHelper::load_data_from_txt(rois_vector, "D:\\caffe\\caffe\\caffe\\caffe\\rois.txt");

	float* inputConv_ptr = input_conv->mutable_cpu_data();//data_ptr指向网络的输入blob
	float* inputRois_ptr = input_rois->mutable_cpu_data();
	//int num_ele = input_rois->count();
	//float* data_ptr = CaffeHelper::_generate_random_input(num_ele);
	//float* input_ptr = input_blobs->mutable_cpu_data();//data_ptr指向网络的输入blob
	CaffeHelper::memcpy_from_vector(inputConv_ptr, conv_vector);
	CaffeHelper::memcpy_from_vector(inputRois_ptr, rois_vector);

	net->Reshape();
	net->Forward();

	vector<Blob<float>*> output_blobs = net->output_blobs();
	for (int i = 0; i < output_blobs.size(); ++i)
	{
		std::cout << "Output " << i << ": " << output_blobs[i]->shape_string();
	}

	system("pause");

	return 0;
}