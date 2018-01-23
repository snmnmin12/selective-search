#include <fstream>
#include "graph.h"
#include <sys/time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

int main(int argc, char** argv) {

	if (argc != 2) {
		cout << "Usage: ./executable inputfile"<< endl;
		exit(0);
 	}

 	int scale = 700;
 	int mini_pixels = 50;
 	float sigma = 0.8;
 	string filename(argv[1]);

 	struct timeval start, stop;
 	double total_time;
 	gettimeofday(&start, NULL); 

 	//image start
 	cv::Mat img = cv::imread(filename);
 	cv::Mat labels, output;

	ImageGraph graph;
	graph.Seg(img, scale, sigma, mini_pixels);

	gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000001*(stop.tv_usec-start.tv_usec);
    printf("seg time (sec) = %8.4f\n", total_time);	
	SelectiveSearch selective(img, graph, 1000, 30000, 3);
	std::vector<cv::Rect> proposals = selective.run();
	// std::vector<cv::Rect> proposals = selectiveSearch(img, graph, 20000, 100000, 2.5);
	// graph.findLabels(labels);

	// drawContours(img, labels, output);
	// Compute time taken
    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000001*(stop.tv_usec-start.tv_usec);
    printf("time (sec) = %8.4f\n", total_time);
   	printf("proposals size: %lu\n", proposals.size());
	for (auto &&rect : proposals) {
		cv::rectangle( img, rect, cv::Scalar( 0, 255, 0 ), 3, 8 );
	}
	cv::imwrite("result.png", img);
	cv::imshow( "result", img );
	cv::waitKey( 0 );
 	//cv::imshow("name", output);
 	//cv::waitKey(0);
 	
	return 0;
}