#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  inferred_axis_ = -1;
  copy_axes_.clear();
  const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape(); // 获取prototxt的reshape的param信息
  const int top_num_axes = top_blob_shape.dim_size(); // 获取 shape param 指定的 dim 数
  constant_count_ = 1;
  /*
   * 将 dim值为0的index存储在 copycopy_axes_  表示复制 bottom->shape[index]值
   * 将 dim值为-1 的index存储在inferred_axis_ 中，用于后续推理
   * 对于非0非-1的值表示top对应维度的真实shape，计算累计的元素数量count
   */
  for (int i = 0; i < top_num_axes; ++i) {
    const int top_dim = top_blob_shape.dim(i);
    if (top_dim == 0) {
      copy_axes_.push_back(i); // 保存等于0的dim
    } else if (top_dim == -1) {
      CHECK_EQ(inferred_axis_, -1) << "new shape contains multiple "
          << "-1 dims; at most a single (1) value of -1 may be specified";
      inferred_axis_ = i; // 保存等于-1的dim
    } else {
      constant_count_ *= top_dim;
    }
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*
   * 计算原理：
   *	指定axis作为start_axis  指定num_axes计算end_axis = start_axis + num_axes
   *    if i in [0, start_axis] or [end_axis, bottom[0]->num_axes()]保持 top->shape(i) = bottom->shape(i)
   *    else top 的 shape 由 shape 的 dim 计算
   *
   *	input: [1,3,4,6]
   *	
   *    reshape_param {
   *		shape {
   *			dim: -1
   *		}
   *		axis: 1
   *		num_axes: 2
   *	}
   *
   *	1. start_axis = axis = 1  ;  end_axis = start_axis + num_axes = 3
   *	2. [0,1) -> top_shape[0] = bottom_shape[0] = 1
   *	3. [3,4( -> top_shape[3] = bottom_shape[3] = 6
   *	4. top_shape [1,-1,6] -> [1,12,6] 
   *	
   * 
   */
  /****************************************  计算 start_axis   end_axis *****************************************************/
  const int input_start_axis = this->layer_param_.reshape_param().axis(); // 起始轴 默认 0
  const int start_axis = (input_start_axis >= 0) ? input_start_axis :
      bottom[0]->num_axes() + input_start_axis + 1;
  CHECK_GE(start_axis, 0) << "axis " << input_start_axis << " out of range";
  CHECK_LE(start_axis, bottom[0]->num_axes()) << "axis " << input_start_axis
      << " out of range for " << bottom[0]->num_axes() << "-D input blob";
  const int num_axes = this->layer_param_.reshape_param().num_axes(); // 获取轴偏移量 默认 -1 表示所有
  CHECK_GE(num_axes, -1) << "num_axes must be >= 0, or -1 for all";
  const int end_axis =
      (num_axes == -1) ? bottom[0]->num_axes() : (start_axis + num_axes); // 通过num_axes 计算end axes 末轴
  CHECK_LE(end_axis, bottom[0]->num_axes())
      << "end_axis = axis + num_axes is out of range";
  const int num_axes_replaced = end_axis - start_axis; // 需要替换的轴数量 [start_axis, end_axis)
  const int num_axes_retained = bottom[0]->num_axes() - num_axes_replaced; // 保持不变dim的轴的数量 [0, bottom[0]->num_axes()] - [start_axis, end_axis)
  const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
  const int num_new_axes = top_blob_shape.dim_size();

  /****************************************  计算 top shape值 *****************************************************/
  vector<int> top_shape(num_axes_retained + num_new_axes);
  int top_shape_index = 0;
  // [0, start_axis) top 维度的形状与 bottom相同
  for (int i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  // 保存 [start_axis, end_axis) 中指定的 shape dim
  for (int i = 0; i < num_new_axes; ++i) {
    top_shape[top_shape_index++] = top_blob_shape.dim(i);
  }
  // [end_axis, bottom[0]->num_axes()) top 维度的形状与 bottom相同
  for (int i = end_axis; i < bottom[0]->num_axes(); ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  CHECK_EQ(top_shape_index, top_shape.size());

  /****************************************  计算 top shape中 0 和 -1 对应的shape 推理 *****************************************************/
  // 针对dim为0的 copy_axes_  计算top对应的shape
  for (int i = 0; i < copy_axes_.size(); ++i) {
    const int copy_axis_index = copy_axes_[i]; 
    CHECK_GT(bottom[0]->num_axes(), start_axis + copy_axis_index)
        << "new shape contains a 0, but there was no corresponding bottom axis "
        << "to copy";
    top_shape[start_axis + copy_axis_index] =
        bottom[0]->shape(start_axis + copy_axis_index);
  }
  // 对 -1 的dim单独计算
  if (inferred_axis_ >= 0) {
    // A -1 dim was specified; infer the correct dimension by computing the
    // product of the other dimensions.
    int explicit_count = constant_count_;
    explicit_count *= bottom[0]->count(0, start_axis); // [0, start_axis）内的元素数量
    explicit_count *= bottom[0]->count(end_axis);
  	// 计算dim 为0的元素和 
    for (int i = 0; i < copy_axes_.size(); ++i) {
      const int copy_axis_index = copy_axes_[i];
      explicit_count *= top_shape[start_axis + copy_axis_index];
    }
    CHECK_EQ(0, bottom[0]->count() % explicit_count) << "bottom count ("
        << bottom[0]->count() << ") must be divisible by the product of "
        << "the specified dimensions (" << explicit_count << ")";
    const int inferred_dim = bottom[0]->count() / explicit_count; // 获取元素的剩余量作为-1
    top_shape[start_axis + inferred_axis_] = inferred_dim;
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "output count must match input count";
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
