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

	sort(edges.begin(), edges.end());
	// printf("weight is begin: %f, end: %f\n", edges[0].weight, edges.back().weight);
	// printf("size : %d\n", edges.size());
	// int count = 0;
	std::vector<double> threshold( nodes.size(), scale );

	for (int e = 0; e < edges.size(); e++) {
		// int temp = 0;
		// for (int i = 0; i < total; i++) {
		// 	if (minIndexes[i] == -1) continue;
			ImageEdge& edge = edges[e];
			int n1 = findParent(edge.first);
			int n2 = findParent(edge.second);
			if (n1 != n2) {
				//if (threshHold(nodes[n1], nodes[n2], edge, scale)) {
				if (edge.weight <= threshold[n1] && edge.weight <= threshold[n2]) {
					merge(n1, n2);
					n1 = findParent(n1);
					threshold[n1] = edge.weight + scale / nodes[n1].children;
				}
			}
		}

	// printf("count: %d\n", count);
}

void ImageGraph::finedSeg() {
	for (int e = 0; e < edges.size(); e++) {
		ImageEdge& edge = edges[e];
		int n1 = findParent(edge.first);
		int n2 = findParent(edge.second);
		if (n1 != n2 && ((nodes[n1].children < minSize) || (nodes[n2].children < minSize))) {
			merge(n1, n2);
		}
	}
}

void ImageGraph::Seg() {
	buildGraph();
	initialSeg();
	finedSeg();
	printf("K : %d\n", K);
}	


void ImageGraph::buildGraph() {

	cv::Mat imgF, blurred;
	img.convertTo( imgF, CV_32FC3 );
	cv::GaussianBlur(imgF, blurred, cv::Size(5, 5), sigma );

	nodes.resize(total);

	K = total;

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

			if ((j < W - 1) && (i < H - 1))
			{
				int m  = W * (i + 1) + (j + 1);
				const cv::Vec3f o_bgr = blurred.at<cv::Vec3f>(i + 1, j + 1);
				float weight = rgbdistance(bgr, o_bgr);
				ImageEdge edge(n, m, weight);
				edges.push_back(edge);
			}

			if ((j < W - 1) && ( i > 0 ))
			{
				int m = W * (i - 1) + (j + 1);
				const cv::Vec3f o_bgr = blurred.at<cv::Vec3f>(i - 1, j + 1);
				float weight = rgbdistance(bgr, o_bgr);
				ImageEdge edge(n, m, weight);
				edges.push_back(edge);
			}

			nodes[n].id = n;
			nodes[n].parent = n;
			nodes[n].children = 1;
		}
	}

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

void ImageGraph::findLabels(std::map<int, std::unordered_set<int>> &labels) {
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
