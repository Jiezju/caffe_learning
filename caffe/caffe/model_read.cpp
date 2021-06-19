#include<caffe.hpp>
#include <string>
#include <vector>
using namespace caffe;
using namespace std;

void print(vector<boost::shared_ptr<Blob<float>>>& blobs)
{
	for (unsigned int i = 0; i < blobs.size(); ++i)
	{
		vector<int> shape = blobs[i]->shape();
		unsigned int size = blobs[i]->count();
		float* bpt = blobs[i]->mutable_cpu_data();
	}
}

int main() {
	string net_file = "D:\\visual_studio\\caffe\\caffe\\caffe\\model-zoo\\SqueezeNet\\deploy.prototxt";      //prototxt文件
	string weight_file = "D:\\visual_studio\\caffe\\caffe\\caffe\\model-zoo\\SqueezeNet\\squeezenet_v1.1.caffemodel";     //caffemodel文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));

	//net->CopyTrainedLayersFrom(weight_file);

	vector<string> blob_names = net->blob_names();
	vector<string> layer_names = net->layer_names();
	vector<boost::shared_ptr<Blob<float>>> vblobs = net->blobs();
	vector<boost::shared_ptr<Layer<float>>> vlayers = net->layers();
	
	for (int i = 0; i < layer_names.size(); i++){
		cout << layer_names.at(i) << endl;
	}

	for (int i = 0; i < blob_names.size(); i++){
		cout << blob_names.at(i) << endl;
	}

	vector<int> shape;
	for (int i = 0; i < vblobs.size(); i++){
		shape =  vblobs[i]->shape();
	}

	string layer_type;
	for (int i = 0; i < vlayers.size(); i++){
		layer_type = vlayers[i]->type();
	}

	vector<boost::shared_ptr<Blob<float>>> layer_learnable_blob;
	for (int i = 0; i < vlayers.size(); i++){
		layer_type = vlayers[i]->type();
		layer_learnable_blob = vlayers[i]->blobs();
		if (layer_type == "Convolution")
		{
			print(layer_learnable_blob);
		}
	}

	for (int i = 0; i < vlayers.size(); i++){
		layer_type = vlayers[i]->type();
		const LayerParameter layerparam = vlayers[i]->layer_param();
		if (layer_type == "Convolution"){
			const caffe::ConvolutionParameter convlayer = layerparam.convolution_param();
			unsigned int num_output = convlayer.num_output();
			unsigned int kernel_size = convlayer.kernel_size(0);
			//unsigned int kernel_size = convlayer.kernel_size_size;
		}
	}

	system("pause");
	return 0;
}