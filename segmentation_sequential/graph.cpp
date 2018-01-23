#include "graph.h"
#include <iostream>
#include <cmath>

int ImageGraph::findParent(int id) {
	int n_id = id;
	while(n_id != nodes[n_id].parent) {
		n_id = nodes[n_id].parent;
	}
	nodes[id].parent = n_id;
	return n_id; 
}

void ImageGraph::merge(int n1, int n2) {
	// std::cout <<"K: " << graph.K << std::endl;
	assert(K>0);
	if ( nodes[n1].id < nodes[n2].id )
	{
		nodes[n2].parent = n1;
		nodes[n1].children += nodes[n2].children;
	}
	else
	{
		nodes[n1].parent = n2;
		nodes[n2].children += nodes[n1].children;
	}
	K--;
}

void ImageGraph::initialSeg() {
	assert(K>0);
	struct timeval start, stop;
 	double total_time;
 	gettimeofday(&start, NULL); 
	sort(edges.begin(), edges.end());
	gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000001*(stop.tv_usec-start.tv_usec);
    printf("time (sec) = %8.4f\n", total_time);
	int count = 0;
	std::vector<double> threshold( nodes.size(), scale );
	for (int e = 0; e < edges.size(); e++) {
		ImageEdge& edge = edges[e];
		int n1 = findParent(edge.first);
		int n2 = findParent(edge.second);
		if (n1 != n2) {
			//if (threshHold(nodes[n1], nodes[n2], edge, scale)) {
			if (edge.weight <= threshold[n1] && edge.weight <= threshold[n2]) {
				merge(n1, n2);
				n1 = findParent(n1);
				threshold[n1] = edge.weight + scale / nodes[n1].children;
				count++;
			}
			//}
		}
	}
	printf("count: %d\n", count);
}

void ImageGraph::finedSeg(int M) {
	for (int e = 0; e < edges.size(); e++) {
		ImageEdge& edge = edges[e];
		int n1 = findParent(edge.first);
		int n2 = findParent(edge.second);
		if (n1 != n2 && ((nodes[n1].children < M) || (nodes[n2].children < M))) {
			merge(n1, n2);
		}
	}
}

void ImageGraph::Seg(const cv::Mat& img, float _scale, float _sigma, int M) {
	
	H = img.rows;
	W = img.cols;
	scale = _scale;
	sigma = _sigma;

	buildGraph(img);
	initialSeg();
	finedSeg(M);
	printf("K : %d\n", K);
}	


void ImageGraph::buildGraph(const cv::Mat &image) {

	cv::Mat imgF, blurred;
	image.convertTo( imgF, CV_32FC3 );
	cv::GaussianBlur(imgF, blurred, cv::Size(5, 5), sigma );
	int N = H * W;

	nodes.resize(N);

	K = N;

	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			int n = W * i + j;
			ImageNode& node = nodes[n];
			const cv::Vec3f& bgr = blurred.at<cv::Vec3f>(i, j);

			if (i < H - 1) {
				int m  = W * (i + 1) + j;
				const cv::Vec3f& o_bgr = blurred.at<cv::Vec3f>(i + 1, j);
				float weight = rgbdistance(bgr, o_bgr);
				ImageEdge edge(n, m, weight);
				edges.push_back(edge);
			}
			if (j < W - 1) {
				int m  = W * i + j + 1;
				const cv::Vec3f o_bgr = blurred.at<cv::Vec3f>(i, j + 1);
				float weight = rgbdistance(bgr, o_bgr);
				ImageEdge edge(n, m, weight);
				edges.push_back(edge);

			}

			// if ((j < W - 1) && (i < H - 1))
			// {
			// 	int m  = W * (i + 1) + (j + 1);
			// 	const cv::Vec3f o_bgr = blurred.at<cv::Vec3f>(i + 1, j + 1);
			// 	float weight = rgbdistance(bgr, o_bgr);
			// 	ImageEdge edge(n, m, weight);
			// 	edges.push_back(edge);
			// }

			// if ((j < W - 1) && ( i > 0 ))
			// {
			// 	int m = W * (i - 1) + (j + 1);
			// 	const cv::Vec3f o_bgr = blurred.at<cv::Vec3f>(i - 1, j + 1);
			// 	float weight = rgbdistance(bgr, o_bgr);
			// 	ImageEdge edge(n, m, weight);
			// 	edges.push_back(edge);
			// }

			nodes[n].id = n;
			nodes[n].parent = n;
			nodes[n].children = 1;
		}
	}
	printf("edge size %lu\n", edges.size());

}

void ImageGraph::findLabels(cv::Mat &labels) {

	labels.create(H, W, CV_32SC1);
	labels = cv::Scalar(0);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int n = W*i + j;
            int m = findParent(n);
            labels.at<int>(i, j) = m;
        }
    }
}

void ImageGraph::findLabels(std::map<int, std::set<int>> &labels) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int n = W*i + j;
            int m = findParent(n);
            labels[m].insert(n);
        }
    }
}

bool isBoundary(const cv::Mat &labels, int i, int j) {

	if ((i < labels.rows - 1) && labels.at<int>(i + 1, j) != labels.at<int>(i, j)) return true;
	if ((i > 0) && labels.at<int>(i - 1, j) != labels.at<int>(i, j)) return true;
	if ((j < labels.cols - 1) && labels.at<int>(i, j + 1) != labels.at<int>(i, j)) return true;
	if ((j > 0) && labels.at<int>(i, j - 1) != labels.at<int>(i, j)) return true;
	return false;
}

void drawContours(const cv::Mat &image, const cv::Mat &labels, cv::Mat &output) {
	assert(image.rows == labels.rows);
	assert(image.cols == labels.cols);
	assert(labels.type() == CV_32SC1);

	int H = image.rows, W = image.cols;
	cv::Vec3b color(0, 0, 0);
	output.create(H, W, CV_8UC3);

	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if (isBoundary(labels, i, j)) {
				output.at<cv::Vec3b>(i , j) = color; 
			} else {
				output.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j);
			}
		}
	}

}

//#####################################################################
//selectiveSearch
//
//
//#####################################################################

cv::Mat SelectiveSearch::calTextureGradient(const cv::Mat &img) {
	cv::Mat sobelX, sobelY;
	cv::Sobel(img, sobelX, CV_32F, 1, 0);
	cv::Sobel(img, sobelY, CV_32F, 0, 1);
	cv::Mat magnitude, angle;
	cv::cartToPolar(sobelX, sobelY, magnitude, angle, true);
	return angle;
}

std::vector<float> SelectiveSearch:: calColorHist(uchar* hsv, int channelsize, int size) {

	const int numBins = 25;
	const int maxVal = 256;
	const int minVal = 0;
	std::vector<float> features;
	int range = maxVal - minVal;
	for (size_t i = 0; i < channelsize; i++) {
		int histo[numBins];
		memset(histo, 0, sizeof(int)*numBins);
		for (int j = 0; j < size; j++) {
    		unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((hsv[i * size + j] - minVal) / (range * 1.0) * numBins));
    		histo[bin]++;
		}
		for (int j = 0; j < numBins; j++) {
			float val = histo[j] / (size * 1.0f);
			features.push_back(val);
		} 
  	}
  	assert(features.size()==numBins*channelsize);
	return features;
}

std::vector<float> SelectiveSearch:: calTextureHist(uchar* intensity, int channelsize, int orientations, int size) {	
	const int numBins = 10;
	const int maxVal  = 256;
	const int minVal  =  0;
	int range = maxVal - minVal;
	std::vector<float> features;
	for (int channel = 0; channel < channelsize; channel++) {
		for (int angle = 0; angle < orientations; angle++) {
			int histo[numBins];
			memset(histo, 0, sizeof(int)*numBins);
			for (int i = 0; i < size; i++) {
				unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((intensity[channel * orientations * size + angle * size + i] - minVal) / (range * 1.0) * numBins));
    			histo[bin]++;
			}
			for (int j = 0; j < numBins; j++) {
				features.push_back(histo[j] / (size * 1.0f));
			} 

		}
	}
	assert(features.size()==channelsize*orientations*numBins);
	return features;
}

void SelectiveSearch:: extractRegions() {

   	std::map<int, std::set<int>> labels;
   	graph.findLabels(labels);

	cv::Mat gradient,hsv;
	gradient = calTextureGradient(img);
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	for (auto& entry : labels) {
   		int label = entry.first;
   		int num = 0;
   		int size = entry.second.size();

   		uchar* texturedistribution = new uchar[CHANNELS*ORIENTATIONS*size];
   		uchar* colorDistribution   = new uchar[CHANNELS*size];
   		int xmin = INF, ymin = INF, xmax = 0, ymax = 0;
   		for (int n : entry.second) {
   			int y = n / W, x = n % W;

   			xmin = std::min(xmin, x);
   			ymin = std::min(ymin, y);
   			xmax = std::max(xmax, x);
   			ymax = std::max(ymax, y);

            for (int channel = 0; channel < CHANNELS; channel++) {
   				colorDistribution[channel*size + num] = hsv.at<cv::Vec3b>(y, x)[channel];
   				int angle = (int)(gradient.at<cv::Vec3f>(y, x)[channel]/22.5) % ORIENTATIONS;
   				texturedistribution[channel * ORIENTATIONS * size + angle * size + num] = img.at<cv::Vec3b>(y,x)[channel];
   			}
   			num++;
   		}

   		R[label].rect        = cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
   		R[label].corlorHist  = calColorHist(colorDistribution, CHANNELS, size);
   		R[label].textureHist = calTextureHist(texturedistribution, CHANNELS, ORIENTATIONS, size);
   		R[label].size 		 = size;

   		delete[] texturedistribution;
   		delete[] colorDistribution;
   	}
}

void SelectiveSearch:: extractNeighbours() {

	// neighbours.reserve(R.size() * (R.size() - 1) / 2);
	for (std::map<int, Region>::iterator it = R.begin(); it != R.end(); it++) {
		std::map<int, Region>::iterator tmp = it;
		tmp++;
		for (; tmp != R.end(); tmp++) {
			if (isIntersecting(it->second, tmp->second)){
				double weight = Similarity(R[it->first], R[tmp->first], H*W);
				std::pair<int, int> key  = {std::min(it->first, tmp->first), std::max(it->first, tmp->first)};
				S[key] =  weight;
			}
		}
	}
}

std::vector<float> SelectiveSearch::merge(const std::vector<float> &a, const std::vector<float> &b, int asize, int bsize) {
	std::vector<float> newVector;
	newVector.resize(a.size());
	for (int i = 0; i < a.size(); i++) {
		newVector[i] = (a[i] * asize + b[i] * bsize) / (asize + bsize);
	}
	return newVector;
}

Region SelectiveSearch::mergeRegions(const Region &r1, const Region &r2) {
	assert(r1.corlorHist.size() == r2.corlorHist.size());
	assert(r1.textureHist.size() == r2.textureHist.size());
	Region region;
	region.rect 	   = (r1.rect | r2.rect);
	region.size 	   = (r1.size + r2.size);
	region.corlorHist  = merge(r1.corlorHist,  r2.corlorHist, r1.size, r2.size);
	region.textureHist = merge(r1.textureHist, r2.textureHist,r1.size, r2.size);
	return region;
}

std::vector<cv::Rect> SelectiveSearch::run() {
	
	assert(img.channels() == 3); 
	// std::map<int, Region> R;
	int imgSize = H * W;
	// struct timeval start, stop;
 // 	double total_time;
 // 	gettimeofday(&start, NULL); 

	extractRegions();

	extractNeighbours(); 

	using Edge = std::pair<std::pair<int, int>, double >;
	auto cmp = []( const Edge &a, const Edge &b ) { return a.second < b.second; };

	// gettimeofday(&start, NULL); 
	while (!S.empty()) {
		auto m = std::max_element( S.begin(), S.end(), cmp );
		int i = m->first.first;
		int j = m->first.second;
		auto ij = std::make_pair( i, j );
		
		int t = R.rbegin()->first + 1;
		R[t] = mergeRegions(R[i], R[j]);

		std::vector<std::pair<int ,int>> keyToDelete;

		//findout elements to remove
		for (auto &s : S) {
			auto key = s.first;
			if ((i == key.first) || (i == key.second) || (j == key.first) || (j == key.second)) {
				keyToDelete.push_back( key );
			}
		}
		//remove the elements and combine
		for (auto& key : keyToDelete) {
			S.erase(key);
			if (key == ij) continue;
			int n = (key.first == i || key.first == j) ? key.second : key.first;
			S[std::make_pair(n, t)] = Similarity(R[n], R[t], imgSize);
		}

	}
	// gettimeofday(&stop, NULL); 
 //    total_time = (stop.tv_sec-start.tv_sec)
	// +0.000001*(stop.tv_usec-start.tv_usec);
	// printf("processing times (sec) = %8.4f\n", total_time);

	std::vector<cv::Rect> proposals;
	//filter out the unuseful information
	for (auto &r : R) {
		if(std::find(proposals.begin(), proposals.end(), r.second.rect) != proposals.end()) continue;
		if (r.second.size < smallest || r.second.size > largest)  continue;
		double w = r.second.rect.width;
		double h = r.second.rect.height;
		if ((w/h > distorted) || (h/w > distorted)) continue;
		proposals.push_back(r.second.rect);
	}
	return proposals;
}
