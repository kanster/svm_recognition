#pragma once
#ifndef BOW_ASSIGNER_H_
#define BOW_ASSIGNER_H_

#include <opencv2/features2d/features2d.hpp>

namespace ocl {

class BoW
{
public:
  BoW();

  virtual ~BoW();

	void setVocabulary(const cv::Mat& vocab);

	const cv::Mat& getVocabulary() const;

	int getVocabularySize() const;

	void compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor);
	

private:
	cv::Mat vocabulary;

	cv::Ptr<cv::DescriptorMatcher> matcher;
};


} /* ocl */

#endif
