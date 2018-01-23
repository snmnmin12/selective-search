#include "selective.h"

cv::Mat SelectiveSearch::calTextureGradient(const cv::Mat &img) {
    cv::Mat sobelX, sobelY;
    std::thread sol1([&img, &sobelX](){
        cv::Sobel(img, sobelX, CV_32F, 1, 0);
    });
    std::thread sol2([&img, &sobelY](){
        cv::Sobel(img, sobelY, CV_32F, 0, 1);
    });
    sol1.join();sol2.join();
    cv::Mat magnitude, angle;
    cv::cartToPolar(sobelX, sobelY, magnitude, angle, true);
    return angle;
}

std::vector<float>  SelectiveSearch:: calColorHist(uchar* hsv, int channelsize, int size) {

    const int numBins = 25;
    const int maxVal = 256;
    const int minVal = 0;
    std::vector<float> features(channelsize * numBins);
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
                features[i * numBins + j] = (val);
            } 
    }
    // region.corlorHist = features;
    assert(features.size()==75);
    return features;
}

std::vector<float> SelectiveSearch:: calTextureHist(uchar* intensity, int channelsize, int orientations, int size) {    
    const int numBins = 10;
    const int maxVal  = 256;
    const int minVal  =  0;
    int range = maxVal - minVal;
    // ThreadPool pool(4);
    // std::vector< std::future<bool> > results;

    std::vector<float> features(numBins * channelsize * orientations);

    for (int channel = 0; channel < channelsize; channel++) {
        for (int angle = 0; angle < orientations; angle++) {
            // results.emplace_back(pool.enqueue([numBins, size, channel, orientations, angle, minVal, range, &features, &intensity](){
                int histo[numBins];
                memset(histo, 0, sizeof(int)*numBins);
                for (int i = 0; i < size; i++) {
                    unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                               static_cast<unsigned int>((intensity[channel * orientations * size + angle * size + i] - minVal) / (range * 1.0) * numBins));
                    histo[bin]++;
                }
                for (int j = 0; j < numBins; j++) {
                    float val = histo[j] / (size * 1.0f);
                    features[channel * orientations * numBins + angle * numBins + j] = val;
                }
                // return true;
            // })); 
        }
    }
    assert(features.size()==240);//channelsize*orientations*numBins);
    return features;
}

void SelectiveSearch:: extractRegions(std::map<int, std::unordered_set<int>> &labels) {
 
    cv::Mat gradient,hsv;
    gradient = calTextureGradient(img);
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    ThreadPool *pool = new ThreadPool(4);
    // std::vector<std::thread> threads;
    // std::vector<int> keys;
    // for (auto& entry : labels)   keys.push_back(entry.first);

    // int numThreads= 4;
    // numThreads = std::min((int)labels.size(), numThreads);
    // int block = (labels.size() + numThreads-1) / numThreads;
    // // printf("block is %d with label size %lu\n", block, labels.size());
    // for (int i = 0 ; i < numThreads; i++) {
    //     int start = i * block;
    //     int end   = std::min((int)labels.size(), (i + 1) * block);
    //     threads.push_back(

    //         std::thread([this, &gradient, &hsv, &labels, &keys, start, end]()
    //         {
    //             for (int j = start; j < end; j++) {
    //                 int label = keys[j];
    //                 // printf("processing label %d, %d\n", j, label);
    //                 int num   = 0;
    //                 int size   = labels[label].size();

    //                 uchar* texturedistribution = new uchar[CHANNELS*ORIENTATIONS*size];
    //                 uchar* colorDistribution   = new uchar[CHANNELS*size];
    //                 int xmin = INF, ymin = INF, xmax = 0, ymax = 0;
    //                 for (int n : labels[label]) {
    //                     int y = n / W, x = n % W;

    //                     xmin = std::min(xmin, x);
    //                     ymin = std::min(ymin, y);
    //                     xmax = std::max(xmax, x);
    //                     ymax = std::max(ymax, y);

    //                     for (int channel = 0; channel < CHANNELS; channel++) {
    //                         colorDistribution[channel*size + num] = hsv.at<cv::Vec3b>(y, x)[channel];
    //                         int angle = (int)(gradient.at<cv::Vec3f>(y, x)[channel]/22.5) % ORIENTATIONS;
    //                         texturedistribution[channel * ORIENTATIONS * size + angle * size + num] = img.at<cv::Vec3b>(y,x)[channel];
    //                     }
    //                     num++;
    //                 }
    //                 this->R[label].size        = size;
    //                 this->R[label].corlorHist  = calColorHist(colorDistribution, CHANNELS, size);
    //                 this->R[label].textureHist = calTextureHist(texturedistribution, CHANNELS, ORIENTATIONS, size);
    //                 this->R[label].rect        = cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
    //                 delete[] texturedistribution;
    //                 delete[] colorDistribution;
    //             }
    //     }));
    // }

    // for (auto& th : threads) th.join();


    for (auto& entry : labels) {
       pool->enqueue([this, &gradient, &hsv, &entry]() {
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
            this->R[label].size        = size;
            this->R[label].corlorHist  = calColorHist(colorDistribution, CHANNELS, size);
            this->R[label].textureHist = calTextureHist(texturedistribution, CHANNELS, ORIENTATIONS, size);
            this->R[label].rect        = cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            delete[] texturedistribution;
            delete[] colorDistribution;
        }
       );
    }
    delete pool;
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
    region.rect        = (r1.rect | r2.rect);
    region.size        = (r1.size + r2.size);
    region.corlorHist  = merge(r1.corlorHist,  r2.corlorHist, r1.size, r2.size);
    region.textureHist = merge(r1.textureHist, r2.textureHist,r1.size, r2.size);
    return region;
}

std::vector<cv::Rect> SelectiveSearch::run(std::map<int, std::unordered_set<int>> &labels) {
    
    assert(img.channels() == 3); 
    // std::map<int, Region> R;
    int imgSize = H * W;
    struct timeval start, stop;
    double total_time;
    gettimeofday(&start, NULL); 

    extractRegions(labels);
    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
    +0.000001*(stop.tv_usec-start.tv_usec);
    printf("region extraction times (sec) = %8.4f\n", total_time);

    extractNeighbours(); 

    using Edge = std::pair<std::pair<int, int>, double >;
    auto cmp = []( const Edge &a, const Edge &b ) { return a.second < b.second; };
    gettimeofday(&start, NULL); 
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
    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
    +0.000001*(stop.tv_usec-start.tv_usec);
    printf("processing times (sec) = %8.4f\n", total_time);

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

    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
    +0.000001*(stop.tv_usec-start.tv_usec);
    printf("processing times (sec) = %8.4f\n", total_time);
    return proposals;
}