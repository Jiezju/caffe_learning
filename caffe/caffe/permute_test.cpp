#include<caffe.hpp>
#include <string>
#include <vector>
#include "layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

typedef cv::Vec<float, 3> Vec3d;

int main() {
	string net_file = "deploy.prototxt";      //prototxt�ļ�
	
	Phase phase = TEST;
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(net_file, phase));

	Blob<float>* input_layer = net->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	int width = input_layer->width();//�õ�����ָ��������ͼ��Ŀ�
	int height = input_layer->height();//�õ�����ָ��������ͼ��ĸ�

	float* input_data = input_layer->mutable_cpu_data();//input_dataָ�����������blob

	cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	cv::Mat img = cv::Mat::zeros(4, 4, CV_32FC3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int c = 0; c <img.channels(); c++)
			{
				//��M��ÿһ��Ԫ�ظ�ֵ                
				img.at<Vec3d>(i, j)[c] = 1;
			}
		}
	}
	cv::Mat sample_resized;
	cv::resize(img, sample_resized, input_geometry_);
	input_layer->Reshape(1, num_channels_, img.rows, img.cols);

	// Set input
	std::vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);//����������blob������ͬMat��������
		input_channels.push_back(channel);//�������Matͬinput_channels��������
		input_data += width * height;//һ��һ��ͨ���ز���
	}
	cv::split(sample_resized, input_channels);

	net->Reshape();
	net->Forward();

	Blob<float>* output_layer = net->output_blobs()[0];//output_layerָ��������������ݣ��洢����������ݵ�blob�Ĺ����(1,c,1,1)
	const float* begin = output_layer->cpu_data();//beginָ���������ݶ�Ӧ�ĵ�һ��ĸ���
	const float* end = begin + output_layer->channels();

	vector<float> ans(begin, end);

	system("pause");
	return 0;
}