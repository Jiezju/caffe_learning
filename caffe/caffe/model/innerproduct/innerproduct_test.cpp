#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

using namespace caffe;

float* _generate_random_input(int num_size)
{
	srand(time(NULL));
	float* input = new float[num_size];
	for (int i = 0; i < num_size; i++)
	{
		input[i] = float(rand() % 10);
	}

	return input;
}

int main() 
{
	string net_file = "fc.prototxt";      //prototxt文件
	string weight_file = "fc.caffemodel";     //caffemodel文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));
	net->CopyTrainedLayersFrom(weight_file);
	// int arr = _generate_random_input();

	Blob<float>* input_blobs = net->input_blobs()[0];
	int num_ele = input_blobs->count();
	float* data_ptr = _generate_random_input(num_ele*4);
	float* input_ptr = input_blobs->mutable_cpu_data();//data_ptr指向网络的输入blob

	memcpy(input_ptr, data_ptr, sizeof(float)* num_ele);

	net->Reshape();
	net->Forward();

	Blob<float>* output_layer = net->output_blobs()[0];//output_layer指向网络输出的数据，存储网络输出数据的blob的规格是(1,c,1,1)
	const float* begin = output_layer->cpu_data();//begin指向输入数据对应的第一类的概率
	const float* end = begin + output_layer->channels();

	vector<float> ans(begin, end);

	system("pause");

	delete[] data_ptr;
	return 0;
}