#include <iostream>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <unistd.h>
#include <sys/time.h>
#include <algorithm>
#include "selective.h"

#include <pthread.h>

#include <vector>
#include <map>
#include <unordered_set>
using namespace std;
using namespace cv;


#define MAX_THREADS 10
//pthread 

int thread_id[MAX_THREADS]; // User defined id for thread
pthread_t p_threads[MAX_THREADS];// Threads
pthread_attr_t attr;        // Thread attributes 

int rows;
int cols;

//parent , cheapest edge for every vertex, sizes of the parent
int   *parent;
int   *cheapest;
int   *sizes;

//weight for each edges, source, destination, outdegree for each vertex
float  *wt;
int    *dest;
int    *src;
int    *outdeg;

//threshhold for each edges
float *threshhold;

int num_threads = 4;

float weight(const Vec3f& color1, const Vec3f& color2)
{
    float db = color1[0] - color2[0];
    float dg = color1[1] - color2[1];
    float dr = color1[2] - color2[2];
    return sqrt(db*db + dg*dg + dr*dr);
}

int find (int *parent, int i)
{   
    int id = i;
    while(parent[id]!= id)
    {
        id = parent[id];
    }
    parent[i] = id;
    return id;
}

void join(int *sizes, int *parent, int *cheapest, int ra, int rb) {

    if (sizes[ra] > sizes[rb]) {
        sizes[ra] += sizes[rb];
        parent[rb] = ra;
        cheapest[ra] = -1;
    } else {
        sizes[rb]+=sizes[ra];
        parent[ra]=rb;
        cheapest[rb]=-1;
    } 
}


struct margs {
    int id;
    float k;
};

void *process(void *args) {

    margs marg = *((margs*)args);
    int ind = marg.id;
    int r = ind / num_threads;
    int c = ind % num_threads;
    int blockc = cols / num_threads;
    int blockr = rows / num_threads;
    int startr = r * blockr;
    int endr   = min(rows, (r + 1) * blockr);
    int startc = c * blockc;
    int endc   = std::min(cols, (c + 1) * blockc);
    // printf("index %d, startr is %d, endr is %d, startc is %d, endc is %d\n", ind, startr, endr, startc, endc);
    float scale   = marg.k;
    int counter = rows * cols / num_threads / num_threads;
    int t       = counter;
    bool flag    = 0;
    int temp = 0;
    while(flag==0)
    {
        t=counter;
        temp = 0;
        //find the cheapest for all edges
        for (int i = startr; i < endr - 1; i++) {
            for (int j = startc; j < endc - 1; j++) {
                int index = i * cols + j; 
                int lastVetexIndex =  outdeg[index];
                int startVetexIndex = (index > 0)? outdeg[index -1] : 0;

                for (int k = startVetexIndex; k < lastVetexIndex; k++) {
                   int ra = find(parent, index);
                   int rc = find(parent, dest[k]);
                   if (ra == rc) continue;
                   if ((cheapest[ra] == -1) || (wt[cheapest[ra]] > wt[k])) {
                        cheapest[ra] = k;
                   }
                   if ((cheapest[rc]==-1) || (wt[cheapest[rc]] > wt[k])) { 
                        cheapest[rc] = k;
                   }
                }  
            }
        }

        //find start processing the vertexes
        for (int i = startr; i < endr - 1; i++) {
            for (int j = startc; j < endc - 1; j++) {
                int s = i * cols + j; 
                if(cheapest[s] !=-1)
                {
                    int index   = cheapest[s];
                    float b     = wt[index];
                    int   start = src[index];
                    int   end   = dest[index];

                    int ra  = find(parent, start);
                    int rb  = find(parent, end);
                
                    if(ra==rb) continue;

                    if (threshhold[ra] == -1.0) threshhold[ra] = b;
                    if (threshhold[rb] == -1.0) threshhold[rb] = b;

                    float T1 = threshhold[ra] + scale/sizes[ra]; //getWeight(max_w, sizes, ra,  scale);
                    float T2 = threshhold[rb] + scale/sizes[rb]; //getWeight(max_w, sizes, rb,  scale);
                    float minV = std::min(T1, T2);
                    //condition for merging the two trees
                    if (b <= minV)
                    {
                        join(sizes, parent, cheapest, ra, rb);
                        int rc = find(parent, ra);
                        rb = rc==ra?rb: ra;
                        // threshhold[rc] = b + scale / sizes[rc];
                        threshhold[rc] = std::max(std::max(threshhold[rc], threshhold[rb]),b);
                        // max_w[rc] = std::max(max_w[rc], std::max(b, max_w[rb]));
                        temp++;
                    }
                }
            }
          }
        // cout << "counter: " << counter << " temp: " << temp << endl;
        counter-=temp;
        if(t==counter) flag=1;
    }
   //usleep(300);
   pthread_exit(NULL);
}

void update_boundary(int scale) {

    int blockr = rows / num_threads;
    int blockc = cols / num_threads;
    //update rows
    int startc, endc, startr, endr; 
    for (int r = 0; r < num_threads - 1; r++) {
        startr= (r+1) * blockr - 1;
        // endr  = (r+1) * num_threads;
        for (int c = 0; c < cols; c++) {
            // startc = (c + 1) * num_threads - 1;
            // endc   = (c + 1) * num_threads;
        // for (int c = 0; c < cols; c++) {
            int ra = startr * cols + c;
            int k = outdeg[ra-1];
            ra = find(parent, ra);
            // int rb = endr * cols + c;
            int rb = find(parent, dest[k]);
            if (ra == rb)  continue;

                if (threshhold[ra] == -1.0) threshhold[ra] = wt[k];
                if (threshhold[rb] == -1.0) threshhold[rb] = wt[k];

                float T1 = threshhold[ra] + scale / sizes[ra];//getWeight(max_w, sizes,ra,  scale);
                float T2 = threshhold[rb] + scale / sizes[rb];//getWeight(max_w, sizes, rb, scale);
                float minV = std::min(T2, T1);
                //condition for merging the two trees
                if (wt[k] <= minV)
                {
                    join(sizes, parent, cheapest, ra, rb);
                    int rc = find(parent, ra);
                    rb = rc==ra?rb: ra;
                    // threshhold[rc] = wt[k] + scale/sizes[rc];
                    threshhold[rc] = std::max(std::max(threshhold[ra], threshhold[rb]),wt[k]);
                    // max_w[rc] = ;//std::max(max_w[rc], std::max(wt[k], max_w[rb]));
                }
        }
    }
    // update cols
    // for (int r = 0; r < rows; r++) {
        for (int c = 0; c < num_threads - 1; c++) {
            startc = (c + 1) * blockc - 1;
            // endc   = (c + 1) * num_threads;
            for (int r = 0; r < rows; r++) {
                int ra = r * cols + startc;
                int k = outdeg[ra] - 1;
                ra = find(parent, ra);
                int rb = find(parent, dest[k]);

                if (ra == rb)  continue;
                if (threshhold[ra] == -1.0) threshhold[ra] = wt[k];
                if (threshhold[rb] == -1.0) threshhold[rb] = wt[k];

                float T1 = threshhold[ra] + scale / sizes[ra];
                float T2 = threshhold[rb] + scale / sizes[rb];
                float minV = std::min(T1, T2);

                //condition for merging the two trees
                if (wt[k] <= minV)
                {
                    join(sizes, parent, cheapest, ra, rb);
                    int rc = find(parent, ra);
                    rb = rc==ra?rb: ra;
                    threshhold[rc] = std::max(std::max(threshhold[ra], threshhold[rb]),wt[k]);
                }
        }
    }

}

int main(int argc, char** argv)
{
    Mat img1;
    float k =  50.0;
    int minSize = 20;
    float sigma = 0.8;
    if (argc < 2) {
        cout << "Usage: ./executable inputfile"<< endl;
        exit(0);
    }
    if (argc == 3) {
        num_threads = atoi(argv[2]);
    }

    string name(argv[1]);
    img1=imread(name);

    cv::Mat imgF, blurred;
    img1.convertTo( imgF, CV_32FC3 );
    blurred = imgF;
    cv::GaussianBlur(imgF, blurred, cv::Size(5, 5), sigma );

    struct timeval start, stop;
    double total_time;
    gettimeofday(&start, NULL); 
    //resize(img1,img1,cvSize(1920,1080));
    rows=img1.rows;
    cols=img1.cols;

    cout<<rows<<" "<<cols<< " " << endl;

    int total = rows*cols;
    int sz=2*rows*cols-rows-cols;

    //threads initialization
    // int num_threads = 2;
    // pthread_mutex_init(&lock_min, NULL); 
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

// experiements
    int size = sz;
    wt      = new float[size];
    dest    = new int[size];
    src     = new int[size];
    outdeg  = new int[total];

    parent   = new int[total];
    cheapest = new int[total];
    sizes    = new int[total];
    threshhold = new float[total];


    memset(outdeg, 0 , sizeof(int)*total);
    memset(wt, 0, sizeof(float)*size);
    memset(dest, 0, sizeof(int)*size);
    memset(src, 0, sizeof(int)*size);
    memset(parent, 0, sizeof(int)*total);
    memset(cheapest, 0, sizeof(int)*total);
    memset(sizes, 0, sizeof(int)*total);

    //build the graph
    int startindex = 0;
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {   
            int n = i * cols + j;
            Vec3f color1 = blurred.at<Vec3f>(i,j);
            int count = 0;
            if(i < rows-1)
            {
                Vec3f color2 = blurred.at<Vec3f>(i+1,j);
                float weigh=weight(color1, color2);
                wt[startindex] = weigh;
                dest[startindex] = (i + 1) * cols + j; 
                src[startindex] = n;
                startindex++;
                count++;
            }
            if(j < cols -1)
            {
                Vec3f color2 = blurred.at<Vec3f>(i,j+1);
                float weigh=weight(color1,color2);
                wt[startindex]   = weigh;
                dest[startindex] = i * cols + j + 1;
                src[startindex] = n;
                startindex++;
                count++;
            }
            outdeg[n] = count;
            if (n>0) outdeg[n] += outdeg[n-1];

            parent[n] = n;
            sizes[n] = 1;
            cheapest[n] = -1;
            threshhold[n] = -1.0;
        }
    }

    //
    int total_thread = num_threads * num_threads;
    margs* args = new margs[total_thread];
    for (int i = 0; i < total_thread; i++) {
        thread_id[i] = i;
        args[i].id = i;
        args[i].k  = k;
        pthread_create(&p_threads[id], &attr, process, (void*) (args + i));
    }
    for(int i = 0; i < total_thread; i++) {
        pthread_join(p_threads[i], NULL);
    }

   update_boundary(k);

    cout <<"Finish Running!"<< endl;
    //final run
    for (int i = 0; i < total; i++) {
       int end   =  outdeg[i];
       int start = (i > 0)? outdeg[i-1] : 0;
       for (int j = start; j < end; j++) {
           int ra = find(parent, i);
           int rc = find(parent, dest[j]);
           if (ra != rc && (sizes[ra] < minSize || sizes[rc] < minSize))
            join(sizes, parent, cheapest, ra, rc);
       }

    }

    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
    +0.000001*(stop.tv_usec-start.tv_usec);
    printf("time (sec) = %8.4f\n", total_time);

    map<int, unordered_set<int>> labels;
    for(int i=0;i<total;i++)
    {
        int rc=find(parent,i);
        labels[rc].insert(i);
    }
    printf("labels size: %lu\n", labels.size());

    SelectiveSearch selective(img1, 1000, 10000, 3);
    vector<Rect> proposals = selective.run(labels);

    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)
    +0.000001*(stop.tv_usec-start.tv_usec);
    printf("total (sec) = %8.4f\n", total_time);

    for (auto &&rect : proposals) {
        cv::rectangle( img1, rect, cv::Scalar( 0, 255, 0 ), 3, 8 );
    }
    printf("proposals size: %lu\n", proposals.size());

    delete[] parent;
    delete[] cheapest;
    delete[] sizes;
    delete[] threshhold;

    delete[] wt;
    delete[] dest;
    delete[] src;
    delete[] outdeg;
    delete[] args;

    printf("end the program now!\n");

    pthread_attr_destroy(&attr);

    //show
    std::string file = "result.png";
    cv::imwrite(file, img1);
    cv::imshow( "result", img1 );
    cv::waitKey( 0 );
    return 0;
}