#pragma once

#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <math.h>
#include <unordered_set>

struct ImageNode {
	int id;
	int parent;
	int children;
};

struct ImageEdge {
	int first;
	int second;
	float weight;
	ImageEdge(){}
	ImageEdge(int n, int m, float _weight): first(n),second(m),weight(_weight){}
	bool operator<(const ImageEdge& that) const {return weight < that.weight;}
};

class ImageGraph {
public:
	ImageGraph(cv::Mat &_img, float _scale, float _sigma, float _minSize):
		img(_img), scale(_scale), sigma(_sigma), minSize(_minSize) {
			H = img.rows; W = img.cols; total = H*W, K = total;
		}
	int findParent(int);
	void merge(int n1, int n2);
	void initialSeg();
	void finedSeg();
	void Seg();
	void findLabels(cv::Mat &labels);
	void findLabels(std::map<int, std::unordered_set<int>> &labels);
	void buildGraph();

private:
	cv::Mat& img;
	std::vector<ImageNode> nodes;
	std::vector<ImageEdge> edges;
	int K, H, W, total;
	int minSize;
	float scale, sigma;
};
// };
//helper distance function and threshhold
inline float rgbdistance(const cv::Vec3f& first, const cv::Vec3f& second) {
	return sqrt((first[0] - second[0]) * (first[0] - second[0]) + (first[1] - second[1]) * (first[1] - second[1]) + (first[2] - second[2]) * (first[2] - second[2]));
}
//bool threshHold(const ImageNode& node1, const ImageNode& node2, const ImageEdge& e, float c);
//post processing
void drawContours(const cv::Mat &image, const cv::Mat &labels, cv::Mat &output);
bool isBoundary(cv::Mat &labels, int i, int j);
