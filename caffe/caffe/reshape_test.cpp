#include "caffe_helper.h"


int main()
{
	string net_file = "reshape.prototxt";      //prototxt文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));
	// int arr = _generate_random_input();

	Blob<float>* input_blobs = net->input_blobs()[0];
	int num_ele = input_blobs->count();
	float* data_ptr = CaffeHelper::_generate_random_input(num_ele);
	float* input_ptr = input_blobs->mutable_cpu_data();//data_ptr指向网络的输入blob

	memcpy(input_ptr, data_ptr, sizeof(float)* num_ele);

	net->Reshape();
	net->Forward();

	vector<Blob<float>*> output_blobs = net->output_blobs();
	for (int i = 0; i < output_blobs.size(); ++i)
	{
		std::cout << "Output " << i << ": " << output_blobs[i]->shape_string();
	}

	system("pause");

	delete[] data_ptr;
	return 0;
}