#include <vector>

#include "caffe/layers/relevant_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RelevantLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  derivativesSum_.ReshapeLike(*bottom[0]); //存放每一个坐标的导数
}

//人脸坐标的排列方式为(x1,x2,x3,x3,x5,y1,y2,y3,y4,y5)

template <typename Dtype>
void RelevantLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int half_channel = channel / 2; 
  //0----half_channel - 1是坐标x， half_channel ----channel-1是坐标y
  //h,w都是1
  //计算loss ,bottom[0] 是predict, bottom[1] 是groundtruth
  double total_loss = 0;
  for(int n_num = 0; n_num < num; n_num++)
  {
	  //处理坐标x
	  for(int c_channel_current = 0; c_channel_current + 1 < half_channel; c_channel_current++)
	  {
		  float predict_current = bottom[0]->data_at(n_num, c_channel_current, 0, 0);
		  float groundtruth_current = bottom[1]->data_at(n_num, c_channel_current, 0, 0);
		  
		  for(int c_channel_next = c_channel_current + 1;  c_channel_next < half_channel; c_channel_next++)
		  {
			  double predict_next = bottom[0]->data_at(n_num, c_channel_next, 0, 0);
		      double predict_diff = predict_current - predict_next;
		 
		      double groundtruth_next = bottom[1]->data_at(n_num, c_channel_next, 0, 0);
		      double groundtruth_diff = groundtruth_current - groundtruth_next;
		  
			  double dist = predict_diff-groundtruth_diff;
			  //求偏导，供反向传播时使用
			  derivativesSum_.mutable_cpu_data()[derivativesSum_.offset(n_num, c_channel_current, 0, 0)] += dist;
			  derivativesSum_.mutable_cpu_data()[derivativesSum_.offset(n_num, c_channel_next, 0, 0)] -= dist;
			  
		       double current_loss = dist * dist;
		      //double current_loss = sqrtf((dist)*(dist));
		      total_loss += current_loss;
		  }
		  
	  }
	  //处理坐标y
	  for(int c_channel_current = half_channel; c_channel_current + 1 < channel; c_channel_current++)
	  {
		  double predict_current = bottom[0]->data_at(n_num, c_channel_current, 0, 0);
		  double groundtruth_current = bottom[1]->data_at(n_num, c_channel_current, 0, 0);
		  
		  for(int c_channel_next = c_channel_current + 1;  c_channel_next < channel; c_channel_next++)
		  {
			  double predict_next = bottom[0]->data_at(n_num, c_channel_next, 0, 0);
		      double predict_diff = predict_current - predict_next;
		  
		      double groundtruth_next = bottom[1]->data_at(n_num, c_channel_next, 0, 0);
		      double groundtruth_diff = groundtruth_current - groundtruth_next;
		  
		      double dist = predict_diff-groundtruth_diff;
			  
			    //求偏导，供反向传播时使用
			  derivativesSum_.mutable_cpu_data()[derivativesSum_.offset(n_num, c_channel_current, 0, 0)] += dist;
			  derivativesSum_.mutable_cpu_data()[derivativesSum_.offset(n_num, c_channel_next, 0, 0)] -= dist;
			  double current_loss = dist * dist;
		      //double current_loss = sqrtf((dist)*(dist));
		      total_loss += current_loss;
		  }
	  }
  }
  
  Dtype loss = total_loss / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RelevantLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num(); //alpha:1/n
	  //对预测求偏导，每一个得到的结果需*alpha
	  //这里先处理5个点的情况
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          derivativesSum_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(RelevantLossLayer);
REGISTER_LAYER_CLASS(RelevantLoss);

}  // namespace caffe
