#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>
#include <fstream>
#include <cassert>

using namespace caffe;

namespace CaffeHelper{
	void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
	{
		Blob<float>* input_blobs = net->input_blobs()[0];
		memcpy(input_blobs->mutable_cpu_data(), data_ptr,
			sizeof(float)* input_blobs->count());
		net->Forward();
	}

	//! Note: Layer�������������в㣬���磬CaffeNet����23��
	// char *query_layer_name = "conv1";
	unsigned int get_layer_index(boost::shared_ptr< Net<float> > & net, string query_layer_name)
	{
		std::string str_query(query_layer_name);
		vector< string > const & layer_names = net->layer_names(); // ��ȡ���е�layer name
		for (unsigned int i = 0; i != layer_names.size(); ++i)
		{
			if (str_query == layer_names[i])
			{
				return i;
			}
		}
		LOG(FATAL) << "Unknown layer name: " << str_query;
	}

	unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, string query_blob_name)
	{
		std::string str_query(query_blob_name);
		vector< string > const & blob_names = net->blob_names();
		for (unsigned int i = 0; i != blob_names.size(); ++i)
		{
			if (str_query == blob_names[i])
			{
				return i;
			}
		}
		LOG(FATAL) << "Unknown blob name: " << str_query;
	}

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

	void load_data_from_txt(vector<double>& data, const string& filename)
	{
		std::ifstream file(filename, ios::in);
		//assert(file.is_open());
		if (!file.is_open())//�ж��ļ��Ƿ��
		{
			std::cout << "Error opening file" << std::endl;
		}
		string line;
		double num;
		while (getline(file, line))
		{
			stringstream ss(line);
			ss >> num;
			data.push_back(num);
		}
	}

	void memcpy_from_vector(float* bptr, vector<double> data)
	{
		int elemSize = data.size();

		for (int i = 0; i < elemSize; ++i)
		{
			bptr[i] = (float)data[i];
		}
	}
}