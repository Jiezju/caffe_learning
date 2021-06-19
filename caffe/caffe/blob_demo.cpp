#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
using namespace caffe;
using namespace std;

template <typename dtype>
void print_blob(Blob<dtype>& blob)
{
	for (int u = 0; u < blob.num(); u++)
	{
		for (int v = 0; v < blob.channels(); v++)
		{
			for (int w = 0; w < blob.height(); w++)
			{
				for (int x = 0; x < blob.width(); x++)
				{
					cout << "a[" << u << "][" << v << "][" << w << "][" << x << "] = " << blob.data_at(u, v, w, x) << "\t";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;
}

int main()
{
    Blob<float> a;
    cout<<"Size : "<< a.shape_string()<<endl;
    a.Reshape(1, 2, 3, 4);
	
    cout<<"Size : "<< a.shape_string()<<endl;
    // 续上面代码
    float * p = a.mutable_cpu_data();
    for(int i = 0; i < a.count(); i++)
    {
        p[i] = i;
    }
	print_blob(a);

    vector<int> newshape = { 1, 3, 2, 4 };
    a.Reshape(newshape);
    cout << a.shape_string() << endl;

    float* diff = a.mutable_cpu_diff();
    for (int i = 0; i < a.count(); ++i)
    {
	    diff[i] = 0.1;
    }
 
	a.Update();
	print_blob(a);

	a.scale_data(0.1);
	print_blob(a);
  
    // 续上面代码
    cout<<"SUM = "<<a.asum_data()<<endl;
    cout<<"QuntizeSum = "<<a.sumsq_data()<<endl;
	cout << "diffQsum = " << a.asum_diff() << endl; // 0.1*24
	cout << "diffQsum = " << a.sumsq_diff() << endl; // 0.1*0.1*24
	

    return 0;
}
