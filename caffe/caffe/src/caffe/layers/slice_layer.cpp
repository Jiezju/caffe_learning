#include <algorithm>
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
* forward 原理：
*   [1,3,4,24] -> [1,3,2,24], [1,3,2,24]  axis = 2
*   S1. [1,3,4,24] -> [3, 4, 24]      [1,3,2,24] -> [3, 2, 24]
*   S2. for i in 0-3:
			copy [i*2*48] -> [i*2*48]
		i+= 2
		
*	
*/

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";
  slice_point_.clear();
  // 提取 slice point 参数
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(slice_point_)); // std::back_inserter() 将 slice_param.slice_point() 插入到 vector slice_point_ 末尾， 容量不足时进行动态扩容
}

template <typename Dtype>
void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  if (slice_param.has_slice_dim()) {
    slice_axis_ = static_cast<int>(slice_param.slice_dim());
    // Don't allow negative indexing for slice_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(slice_axis_, 0) << "casting slice_dim from uint32 to int32 "
        << "produced negative result; slice_dim must satisfy "
        << "0 <= slice_dim < " << kMaxBlobAxes;
    CHECK_LT(slice_axis_, num_axes) << "slice_dim out of range.";
  } else {
    slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis()); // 提取 slice 的维度
  }
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  
  // 数组划分 [slice_pre_num, slice , slice_post_num]
  num_slices_ = bottom[0]->count(0, slice_axis_); // 计算 slice 维度之前的 [0,slice_axis_) 内的元素个数
  slice_size_ = bottom[0]->count(slice_axis_ + 1); // 计算 slice 维度之后的元素个数
 
  int count = 0;

  if (slice_point_.size() != 0) {
	// 指定 slice point 的情形
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);

	// 每个 slice 出来的块数
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev); // 块数 = slice_point[i+1] - slice_point_[i]
      prev = slice_point_[i];
    }

    slices.push_back(bottom_slice_axis - prev); // 处理最后的 slice_point
    for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }

  } else {
	// 默认平均分的情形
    CHECK_EQ(bottom_slice_axis % top.size(), 0) // 判断slice的分数刚好能够平均分配
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size(); // 计算输出 shape 在 slice 维度上的 shape
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count(); // 计算所有元素总和
    }
  }
  CHECK_EQ(count, bottom[0]->count());

  // 特殊情形， slice 不做切分
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }

  // bottom 指针在slice axes 上的起始位置相对于上一个 bottom 切片指针的 偏移量
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_); // 获取 bottom 需要 slice axis 的 shape
  // 遍历 输出的个数
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
	// 遍历每一片 需要切分的后续数据
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          top_diff + top_offset, bottom_diff + bottom_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);
REGISTER_LAYER_CLASS(Slice);

}  // namespace caffe
