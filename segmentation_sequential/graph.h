#pragma once

#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <math.h>
#include <array>
#include <queue>

#define INF 1000000
#define ORIENTATIONS 8
#define CHANNELS 3

typedef unsigned char uchar;

static long int id = 0;

// typedef unsigned char *TextureDistribution[ORIENTATIONS][CHANNELS];
// tydedef 
//typedef 
// typedef struct {unsigned char elements[CHANNELS][ORIENTATIONS];} Textures;
// typedef struct {unsigned char elements[CHANNELS];} Colors;

//for efficient graph based image segmention
// struct ImageNode {
// 	int id;
// 	int parent;
// 	int children;
// 	float max_w;
// };
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
	ImageGraph(){}
	int findParent(int);
	void merge(int n1, int n2);
	void initialSeg();
	void finedSeg(int M);
	void Seg(const cv::Mat& img, float _scale, float sigma, int minSize);
	void findLabels(cv::Mat &labels);
	void findLabels(std::map<int, std::set<int>> &labels);
	void buildGraph(const cv::Mat &image);

private:
	std::vector<ImageNode> nodes;
	std::vector<ImageEdge> edges;
	int K, H, W;
	float scale, sigma;
};

//for region 
struct Region {
	int size;
	int mid;
	cv::Rect rect;
	std::vector<float> corlorHist;
	std::vector<float> textureHist;
	Region() {mid = id++;}
	bool operator == (const Region &ref) {
		return mid == ref.mid;
	}
};


class SelectiveSearch {

public:
	SelectiveSearch(cv::Mat &_img, ImageGraph& _imgGraph, int _smallest, int _largest, double _distorted) : 
		img(_img), graph(_imgGraph), smallest(_smallest), largest(_largest), distorted(_distorted)
		{ 		
				H = _img.rows; 
				W = _img.cols;
		};
	
	std::vector<cv::Rect> run();

	void extractRegions();
	void extractNeighbours();

	std::vector<float> calColorHist(uchar* hsv, int channelsize, int size);
	std::vector<float> calTextureHist(uchar* intensity, int channelsize, int orientations, int size);
	static std::vector<std::pair<int, int>> extractNeighbours(std::map<int, Region> &R);

	static Region mergeRegions(const Region &r1, const Region &r2);
	static std::vector<float> merge(const std::vector<float> &a, const std::vector<float> &b, int asize, int bsize);
	static cv::Mat calTextureGradient(const cv::Mat &img);

	//helper functions to calculate the similarities
	static inline double calSimilarity(const std::vector<float>& Hist1, const std::vector<float>& Hist2) {
		assert(Hist1.size() == Hist2.size());
		float sum = 0.0;
		for (int i = 0; i < Hist1.size(); i++) { sum += std::min(Hist1[i], Hist2[i]);}
		return sum;
	}
	static inline double SizeSimilarity(const Region &r1, const Region &r2, int imSize) {
		return (1.0 - (double)(r1.size + r2.size)/imSize);
	}
	static inline double RectSimilarity(const Region &r1, const Region &r2, int imSize) {
		return (1.0 - (double)((r1.rect | r2.rect).area() - r1.size - r2.size) / imSize);
	}
	static inline double Similarity(const Region &r1, const Region &r2, int imSize) {
		return (calSimilarity(r1.corlorHist, r2.corlorHist) + calSimilarity(r1.textureHist, r2.textureHist) + SizeSimilarity(r1, r2, imSize) + RectSimilarity(r1, r2, imSize));
	}
	static inline bool isIntersecting(const Region &a, const Region &b) {
		return ((a.rect & b.rect).area() != 0);
	}

private:
	cv::Mat& img;
	ImageGraph& graph;
	int smallest;
	int largest;
	int distorted;
	int H;
	int W;
	//region
	std::map<int, Region> R;
	//neighbour
	std::map<std::pair<int, int>, double> S;

};

//helper distance function and threshhold
inline float rgbdistance(const cv::Vec3f& first, const cv::Vec3f& second) {
	return sqrt((first[0] - second[0]) * (first[0] - second[0]) + (first[1] - second[1]) * (first[1] - second[1]) + (first[2] - second[2]) * (first[2] - second[2]));
}
//bool threshHold(const ImageNode& node1, const ImageNode& node2, const ImageEdge& e, float c);
//post processing
void drawContours(const cv::Mat &image, const cv::Mat &labels, cv::Mat &output);
bool isBoundary(cv::Mat &labels, int i, int j);
