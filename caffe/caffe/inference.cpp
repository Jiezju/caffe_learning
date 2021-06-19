#include<caffe.hpp>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;
// https://blog.csdn.net/jiongnima/article/details/70197866

int main() {
	string net_file = "D:\\visual_studio\\caffe\\caffe\\caffe\\model-zoo\\SqueezeNet\\deploy.prototxt";      //prototxt文件
	string weight_file = "D:\\visual_studio\\caffe\\caffe\\caffe\\model-zoo\\SqueezeNet\\squeezenet_v1.1.caffemodel";     //caffemodel文件
	Caffe::set_mode(Caffe::CPU);
	//Caffe::SetDevice(0);
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));

	net->CopyTrainedLayersFrom(weight_file);


	Blob<float>* input_layer = net->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	int width = input_layer->width();//得到网络指定的输入图像的宽
	int height = input_layer->height();//得到网络指定的输入图像的高

	float* input_data = input_layer->mutable_cpu_data();//input_data指向网络的输入blob

	string input_path = "D:\\visual_studio\\caffe\\caffe\\caffe\\cat.jpg";
    cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	cv::Mat img = cv::imread(input_path, -1);
	if (img.empty())
	{
		return -1;
	}
	cv::Mat sample_resized;
	cv::resize(img, sample_resized, input_geometry_);
	input_layer->Reshape(1, num_channels_, img.rows, img.cols);

	// Set input
	std::vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);//将网络输入blob的数据同Mat关联起来
		input_channels.push_back(channel);//将上面的Mat同input_channels关联起来
		input_data += width * height;//一个一个通道地操作
	}

	cv::Mat sample_normalized;
	cv::Mat sample_float;
	cv::Mat mean(width, height, CV_32FC1, cv::Scalar(128.0));
	sample_resized.convertTo(sample_float, CV_32FC3);
	//cv::subtract(sample_float, mean, sample_normalized);
	cv::split(sample_resized, input_channels);



	net->Reshape();
	net->Forward();

	Blob<float>* output_layer = net->output_blobs()[0];//output_layer指向网络输出的数据，存储网络输出数据的blob的规格是(1,c,1,1)
	const float* begin = output_layer->cpu_data();//begin指向输入数据对应的第一类的概率
	const float* end = begin + output_layer->channels();

	vector<float> ans(begin,end);

	system("pause");
	return 0;
}