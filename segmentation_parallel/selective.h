#pragma once
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>
#include <map>
#include <unordered_set>
#include <thread>
#include <iostream>
#include <sys/time.h>
#include "threadpool.h"


#define INF 1000000
#define ORIENTATIONS 8
#define CHANNELS 3

typedef unsigned char uchar;
static long int id = 0;

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
    SelectiveSearch(cv::Mat &_img, int _smallest, int _largest, double _distorted) : 
        img(_img), smallest(_smallest), largest(_largest), distorted(_distorted)
        {       
                H = _img.rows; 
                W = _img.cols;
        };
    
    std::vector<cv::Rect> run(std::map<int, std::unordered_set<int>> &labels);

    void extractRegions(std::map<int, std::unordered_set<int>> &labels);
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
    int smallest;
    int largest;
    int distorted;
    int H;
    int W;
    //region
    std::map<int, Region> R;
    //neighbour
    std::map<std::pair<int, int>, double> S;
    // ThreadPool pool;
    // std::vector< std::future<bool> > results;
};