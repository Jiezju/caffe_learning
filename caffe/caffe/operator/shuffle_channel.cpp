#include "caffe_helper.h"

void shuffle(const float* input_ptr, float* output_ptr, const int groups, const int remains, const int space_size)
{
	for (int i = 0; i < groups; ++i)
	{
		for (int j = 0; j < remains; ++j)
		{
			// o[j,i,:] = in[i,j,:]
			const float* in = input_ptr + (i*remains + j) * space_size;
			float* o = output_ptr + (j*groups + i) * space_size;
			memcpy(o, in, space_size*sizeof(float));
		}
	}
}

// [n,c,h,w] -> [n, g, c_, h, w] -> [n, c_, g. h, w] - > [n,c,h,w]
void channelShuffle(Blob<float>* inputs, Blob<float>* outputs, int groups)
{
	const float* bottom_data = inputs->cpu_data();
	float* top_data = outputs->mutable_cpu_data();

	const int bs = inputs->shape(0);
	const int channels = inputs->shape(1);
	const int feature_maps_size = inputs->count(1);
	const int space_size = inputs->count(2);

	int remains = channels / groups;

	for (int n = 0; n < bs; ++n)
	{
		shuffle(bottom_data + n*feature_maps_size, top_data + n*feature_maps_size, groups, remains, space_size);
	}

}

int main()
{
	vector<int> in_shape = {1,256,13,13};
	Blob<float> input(in_shape);

	// init input
	float* in_ptr = input.mutable_cpu_data();
	float* in_data = CaffeHelper::_generate_random_input(input.count());
	memcpy(in_ptr, in_data, input.count()*sizeof(float));
	
	int groups = 2;

	vector<int> out_shape(in_shape.begin(), in_shape.end()); // input_shape = out_shape
	Blob<float> output(out_shape);

	channelShuffle(&input, &output, groups);
	
	
	delete[] in_data;
	return 0;
}