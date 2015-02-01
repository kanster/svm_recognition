#include <cstdlib>
#include <cstdio>
#include <boost/timer.hpp>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/parse.h>

#include "ObjectDescription.h"
#include "ObjectClassifier.h"


using namespace std;

static string getClassName(const size_t & idx) {
	string categories[11] = {"Bottle", "Bowl", "Box", "Can", "Carton", "Cup", "Mug", "Spray-Can", "Tin", "Tube", "Tub"};
	
	if (idx >= 0 && idx < 11) {
		return categories[idx];
	} else {
		return "Unknown";
	}
}


/** print usage
  */
void printUsage(const char* prog_name) {
    cout<< "\n\nUsage: " << prog_name << " [options]\n\n"
        << "Options:\n"
        << "-------------------------------------------\n"
        << "-h          this help\n"
        << "-data       target rgbd point cloud, with .ply type\n"
        << "-vocab      vocabulary data directory\n"
        << "-svm        svm model directory\n";
}

/**
 * Main entry point for testing Object Classification Module
 */
int main(int argc, char *argv[]) {
    if (pcl::console::find_argument (argc, argv, "-h") != -1) {
        printUsage(argv[0]);
        return 0;
    }
    string cloud_name, vocab_dir, svm_dir;
    pcl::console::parse_argument (argc, argv, "-data", cloud_name);
    pcl::console::parse_argument (argc, argv, "-vocab", vocab_dir);
    pcl::console::parse_argument (argc, argv, "-svm", svm_dir);
    cout << "detect " << cloud_name << " using vocabulary: " << vocab_dir << " and svm " << svm_dir << endl;
	pcl::PointCloud<pcl::PointXYZRGB> loadedCloud;
    pcl::io::loadPLYFile(cloud_name, loadedCloud);

	// Change these paths to reflect the layout of your machine
    ocl::ObjectDescription objDesc(vocab_dir);
    ocl::ObjectClassifier classifier(svm_dir);
	{
		boost::timer t;
		objDesc.extractFeatureDescriptors(loadedCloud);
		objDesc.assignBOWs();
		classifier.classify(objDesc.getFPFHBow(), objDesc.getSIFTBow(), objDesc.getHOGBow());
		cout << "After classification\n";
		printf("Elapsed computation time: %g\n", t.elapsed());
	}
	
	printf("Object class: %s\n", getClassName(classifier.objectCategory.categoryLabel - 1).c_str());
	printf("Overall confidence value: %g\n", classifier.measureConfidence());
	printf("\t Confidence scores:\n");
	for (size_t j = 0; j < classifier.objectCategory.classConfidence.size(); j++) {
		printf("\t    %02ld -> %g\n", j + 1, classifier.objectCategory.classConfidence[j]);
	}

	return EXIT_SUCCESS;
}
