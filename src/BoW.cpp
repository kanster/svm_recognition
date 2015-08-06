#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "BoW.h"

namespace ocl {

BoW::BoW() {
  // create descriptor matcher using BruteForce
	matcher  = cv::DescriptorMatcher::create("BruteForce");
}


BoW::~BoW() {
}


void BoW::setVocabulary(const cv::Mat& vocab) {
	matcher->clear();
	vocabulary = vocab;
	matcher->add( std::vector<cv::Mat>(1, vocab) );
}


const cv::Mat& BoW::getVocabulary() const {
	return vocabulary;
}


int BoW::getVocabularySize() const {
	return vocabulary.empty() ? 0 : vocabulary.rows;
}


void BoW::compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor) {
	bowDescriptor.release();
	std::vector<cv::DMatch> matches;
	matcher->match(queryDesc, matches);

	bowDescriptor = cv::Mat(1, getVocabularySize(), CV_32F, cv::Scalar::all(0.0));
	float *descPtr = (float*) bowDescriptor.data;

	for (size_t i = 0; i < matches.size(); i++) {
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx; // cluster index
		CV_Assert( queryIdx == static_cast<int>(i));

		descPtr[trainIdx] = descPtr[trainIdx] + 1.0f;
	}
  // normalize BoW descriptor
	bowDescriptor /= queryDesc.rows;
}


} /* ocl */
