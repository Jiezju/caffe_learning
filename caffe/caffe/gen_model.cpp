#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

float* __generate_random_data(int n)
{
	float* fdata_ptr = new float[n];
	srand(time(NULL));

	for (int i = 0; i < n; i++)
	{
		fdata_ptr[i] = (rand() % 2000) / 1000.f - 1.0;  // 生成数据为[-1,1]区间内的值
	}
	return fdata_ptr;
}

void __generate_random_learnable_blob(boost::shared_ptr<Layer<float>> layer)
{
	string op = layer->type();
	if (!(op == string("InnerProduct")))
	{
		cout << layer->type() << endl;
		return;
	}
		

	vector<boost::shared_ptr<Blob<float>>> layer_learnable_blob = layer->blobs();
	int n_elem;
	for (int i = 0; i < layer_learnable_blob.size(); ++i)
	{
		n_elem = layer_learnable_blob[i]->count();
		float* data_ptr = __generate_random_data(n_elem);
		float* weight_ptr = layer_learnable_blob[i]->mutable_cpu_data();
		memcpy(weight_ptr, data_ptr, sizeof(float)*n_elem);
		delete[] data_ptr;
	}
}

int main()
{
	string net_file = "fc.prototxt";      //prototxt文件
	string weight_file = "fc_gen.caffemodel";     //caffemodel文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));

	vector<boost::shared_ptr<Layer<float>>> vlayers = net->layers();
	const char* layer_type;
	for (int i = 0; i < vlayers.size(); i++)
	{
		__generate_random_learnable_blob(vlayers[i]);
	}

	// 读取模型的权重参数，并写成二进制文件
	caffe::NetParameter net_param;
	net->ToProto(&net_param);
	caffe::WriteProtoToBinaryFile(net_param, weight_file);

	return 0;
}