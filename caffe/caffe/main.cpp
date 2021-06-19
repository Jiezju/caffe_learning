#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
using namespace std;
using namespace caffe;

int main(void)
{
	Blob<float> a;
	cout << "size : " << a.shape_string() << endl;
	a.Reshape(1, 2, 3, 4);
	const float* a_ptr = a.cpu_data();
	cout << "size : " << a.shape_string() << endl;
	return 0;
}