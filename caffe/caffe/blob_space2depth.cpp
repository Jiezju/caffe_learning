#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
#include <numeric>
using namespace caffe;
using namespace std;

/* 参考文章 https://blog.csdn.net/thl789/article/details/109189889
 * 实现tf.space2depth的op操作
 * 
 */

void printBlob(Blob<float>& a)
{
	for (int u = 0; u < a.num(); u++)
	{
		for (int v = 0; v < a.channels(); v++)
		{
			for (int w = 0; w < a.height(); w++)
			{
				for (int x = 0; x < a.width(); x++)
				{
					cout << "a[" << u << "][" << v << "][" << w << "][" << x << "] = " << a.data_at(u, v, w, x) << endl;
				}
				cout << endl;
			}
		}
	}
}

/* 基于输入的index 值计算 输出的 index 索引*/
int ComputeSpace2Depth(Blob<float>& input, Blob<float>& output, int stride)
{
	float* inp_ptr = input.mutable_cpu_data();
	float* out_ptr = output.mutable_cpu_data();

	std::vector<int> in_shape = input.shape();
	std::vector<int> out_shape = output.shape();

	int in_bs = in_shape[0];
	int in_channel = in_shape[1];
	int in_height = in_shape[2];
	int in_width = in_shape[3];

	int o_channel = out_shape[1];
	int o_height = out_shape[2];
	int o_width = out_shape[3];

	// for 循环 4 层 遍历输入的所有index值
	for (int n = 0; n < in_bs; ++n)
	{
		for (int c = 0; c < in_channel;++c)
		{
			for (int h = 0; h < in_height;++h)
			{
				/*
				 * 对矩阵分块，计算索引的技巧， 对于原矩阵的元素索引 [h, w], 对应所在block的索引号为: [h / stride, w / stride]
				 * 对应所在block内部的索引为[h % stride, w % stride]
				 *
				 * 重点在于输出channel维度的计算
				 */
				int out_h = h / stride;
				int offset_h = h % stride;
				for (int w = 0; w < in_width; ++w)
				{
					int out_w = w / stride;
					int offset_w = w % stride;

					int offset_c = offset_h * stride + offset_w;
					int out_c = c * stride * stride + offset_c;

					int in_idx = ((n * in_channel + c)*in_height + h)*in_width + w;
					int o_idx = ((n * o_channel + out_c)*o_height + out_h)*o_width + out_w;

					out_ptr[o_idx] = inp_ptr[in_idx];
				}
			}
		}
	}

	return 0;
}

int main()
{
	vector<int> in_shape = { 1,3, 4, 6};
	Blob<float> input(in_shape);

	cout << "Size : " << input.shape_string() << endl;

	// input 赋值
	float * inptr = input.mutable_cpu_data();

	// 按channel方向初始化
	for (int c = 0; c < 3; ++c)
	{
		float init = static_cast<float>(c);
		/*inptr = static_cast<float>(c);*/
		for (int i = 0; i < input.count()/3; i++)
		{
			*inptr = init;
			inptr++;
			init+=3;
		}
	}
	
	printBlob(input);

	vector<int> out_shape = { 1, 12, 2, 3 };
	Blob<float> out(out_shape);
	ComputeSpace2Depth(input, out, 2);
	printBlob(out);

	return 0;
}