#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
#include <numeric>
using namespace caffe;
using namespace std;

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

int ComputeGather(Blob<float>& input, Blob<int>& index, Blob<float>& output, int axis)
{
	float* inp_ptr = input.mutable_cpu_data();
	int* index_ptr = index.mutable_cpu_data();
	float* out_ptr = output.mutable_cpu_data();

	std::vector<int> in_shape = input.shape();
	std::vector<int> indices_shape = index.shape();
	std::vector<int> out_shape = output.shape();

	const int lhs_size = std::accumulate(in_shape.begin(), in_shape.begin() + axis, 1, std::multiplies<int>());
	const int rhs_size = std::accumulate(in_shape.begin() + axis, in_shape.end(), 1, std::multiplies<int>());

	int index_size = index.count();
	int axis_dim_size = in_shape[axis];

	for (int l = 0; l < lhs_size; ++l) {
		inp_ptr += l * rhs_size;
		for (int idx = 0; idx < index_size; ++idx) {
			out_ptr[l*lhs_size + idx] = *(inp_ptr + index_ptr[idx]);
		}
	}

	return 0;
}

int main()
{
	vector<int> in_shape = { 2, 4 };
	Blob<float> input(in_shape);

	cout << "Size : " << input.shape_string() << endl;

	// input ¸³Öµ
	float * inptr = input.mutable_cpu_data();
	for (int i = 0; i < input.count(); i++)
	{
		inptr[i] = i;
	}
	printBlob(input);
	// indeces
	vector<int> indices_shape = { 2 };
	Blob<int> indices(indices_shape);
	int* idxptr = indices.mutable_cpu_data();
	idxptr[0] = 0, idxptr[1] = 2;

	vector<int> out_shape = { 2, 2 };
	Blob<float> out(out_shape);
	int axis = 1;
	ComputeGather(input, indices, out, axis);
	printBlob(out);

	return 0;
}