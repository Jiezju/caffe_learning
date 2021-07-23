#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

using namespace caffe;

namespace CaffeHelper{
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
}