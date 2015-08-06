#pragma once
#ifndef OCL_UTILS_H_
#define OCL_UTILS_H_

/// Forward declarations to avoid unnecessary includes
namespace pcl {
	template <typename PointType> 
	class PointCloud;
	struct PointXYZRGB;
}


namespace ocl {	

/** 
 * \brief Object Classification Utility functions
 */
class OCLUtils {

public:
	/// Constructs an RGB image from the segmented point cloud provided using the given pose
	static void 
	pointCloudToIntensityImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat3b& rgbImage);


};

} // ocl

#endif /* OCL_UTILS_H_ */
