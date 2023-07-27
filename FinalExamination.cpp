// 最终考核
// 一帧一坐标一半径 证明没有缺失识别和误识别；
// 青色为下一秒预测位置。
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/types_c.h>
#include<ctime>
#include<queue>

using namespace std;
using namespace cv;

Mat frame, ero2;  // 帧、二值图
int frame_counter = 0;  // 循环播放视频
const char *threshold_title = "Threshold Value:";
const char *output_win = "output win";
const char *result_win = "result win";

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

KalmanFilter kal(4, 2, 0);  // 滤波器
Mat state(4, 1, CV_32F);  // 状态向量(x,y,vx,vy)
Mat measurement;  // 测量向量(x,y)
Mat predp;  // 预测坐标
vector<Point> trac;  // 预测轨迹

int aleft = 0, aright = 0;  // 判断换向
Point2f precenter(0, 0);  // 前一帧坐标

Point2f center; // 当前帧橘子中心坐标
float radius;  // 当前帧橘子半径

void TrackBack_Demo(int, void *){};
void OrangeDetection(int);  // 检测橘子
void Judge_Change_Direction(Point2f); // 判断换向
void InitKalmanFilter();  // 初始化卡尔曼滤波器
void Prediction_Update();  // 预测

int main(){
    VideoCapture video("orange2.mp4");

    if(video.isOpened()){
        cout << "Success!" << endl;
    }
    else{
        cout << "Fault!" << endl;
        return -1;
    }

    double fps = video.get(CAP_PROP_FPS);
    cout << "fps:" << fps << endl;

    namedWindow(output_win, WINDOW_AUTOSIZE);
    namedWindow(result_win, WINDOW_AUTOSIZE);

    createTrackbar(threshold_title, output_win, 0, 255, TrackBack_Demo);
    setTrackbarPos(threshold_title, output_win, 202);  // 171

    InitKalmanFilter();  // -初始化卡尔曼滤波器

    clock_t start, end;  // 耗时
    while (1)
    {
        start = clock();

        video >> frame;
        if(frame.empty()){
            break;
        }

        // 循环播放
        frame_counter += 1;
        if (frame_counter == int(video.get(cv::CAP_PROP_FRAME_COUNT)))
        {
            frame_counter = 0;
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        cout << "frame counter:" << frame_counter << endl;

        // 识别橘子
        OrangeDetection(getTrackbarPos(threshold_title, output_win));

        // 判断换向
        Judge_Change_Direction(center);

        // 预测
        Prediction_Update();

        imshow(output_win, ero2);  // 识别橘子的二值图
        imshow(result_win, frame);  // 原图

        end = clock();
        //cout << "时间：" << (double)(end - start) / CLOCKS_PER_SEC + 1 / fps << endl;

        if(waitKey(5000.0 / fps)==27){
            break;
        }
    }

    video.release();
    waitKey(0);

    return 0;
}


void OrangeDetection(int threshold_v) {
    // 检测橘子
    Mat hsv_img;

    cvtColor(frame, hsv_img, COLOR_BGR2HSV_FULL);

    vector<Mat> channels;
    split(hsv_img, channels);
    Mat S = channels.at(1);  // 取颜色深浅特征

    Mat ero1, dila1;
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(7, 7));
    
    erode(S, ero1, kernel1, Point(-1, -1), 2);  // 2-4，去噪
    dilate(ero1, dila1, kernel1, Point(-1, -1), 2);

    Mat tmp_img;
    threshold(dila1, tmp_img, threshold_v, 255, CV_THRESH_BINARY);

    Mat dila2;
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(tmp_img, ero2, kernel2, Point(-1, -1), 5);
    erode(ero2, dila2, kernel2, Point(-1, -1), 5);  // 2-7

    findContours(dila2, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
    //drawContours(frame, contours, 0, Scalar(0, 255, 0), 3);

    // --- 求形心，画橘子轮廓 ---
    for (int j = 0; j < contours.size();j++){  // 用for循环为检测多个轮廓作准备。实际上只有0或1个轮廓。
        float new_radius;
        minEnclosingCircle(contours[j], center, new_radius);
        if(new_radius > 55){  // 将误识别的小圆替换为自定义的最小圆
            radius = new_radius;
        }
        circle(frame, center, 3, Scalar(0, 255, 0), -1);
        circle(frame, center, radius, Scalar(0, 255, 0), 4);
        cout << "(" << center.x << "," << center.y << ") " << radius << endl;
    }

    if (contours.size() == 0)  // 这里要打印出消失的橘子
    {
        center = trac.back();  // 消失的帧 用前一时刻预测的位置
        circle(frame, center, 3, Scalar(0, 255, 0), -1);
        circle(frame, center, radius, Scalar(0, 255, 0), 4);
        cout << "disappeared orange:" << "(" << center.x << "," << center.y << ") " << radius << endl;
    }
}

void InitKalmanFilter() {
    // 初始化卡尔曼滤波器
    measurement = Mat::zeros(2, 1, CV_32F);

    kal.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  // 2D状态转移矩阵
    setIdentity(kal.measurementMatrix);  // 测量矩阵
    setIdentity(kal.processNoiseCov, Scalar::all(1e-3));  // 过程噪声协方差矩阵，越大越滞后
    setIdentity(kal.measurementNoiseCov, Scalar::all(1e-3));  // 测量噪声协方差矩阵
    setIdentity(kal.errorCovPost, Scalar::all(0.1));  // 后验协方差矩阵

    //state.at<float>(0) = 966.695;  // 初始化x坐标
    //state.at<float>(1) = 77.8444;  // 初始化y坐标
    state.at<float>(0) = 0;  // 初始化x坐标
    state.at<float>(1) = 0;  // 初始化y坐标
    state.at<float>(2) = 0;  // 初始化x轴速度
    state.at<float>(3) = 0;  // 初始化y轴速度
    kal.statePost = state;  // 初始化状态向量
}

void Judge_Change_Direction(Point2f curCenter) {
    // --- 判断换向 ---
    if(precenter.x != 0 && precenter.y != 0 && curCenter.x < precenter.x && aleft == 0) {  // 开始向左的时刻
        putText(frame, "To the left!", Point(curCenter.x - 80, curCenter.y + 60), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
        aleft ++;
        aright = 0;
    }
    else if(precenter.x != 0 && precenter.y != 0 && curCenter.x > precenter.x && aright == 0) {  // 开始向右的时刻
        putText(frame, "To the right!", Point(curCenter.x - 80, curCenter.y + 60), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
        aright ++;
        aleft = 0;
    }

    precenter = curCenter; 
}

void Prediction_Update() {
    // 预测
    measurement.at<float>(0) = center.x;  // 测量x坐标
    measurement.at<float>(1) = center.y;  // 测量y坐标

    kal.correct(measurement);  // 修正估计值

    predp = kal.predict();

    Point point(predp.at<float>(0), predp.at<float>(1));
    trac.push_back(point);  // 保留预测位置

    circle(frame, point, 5, Scalar(255, 0, 0), -1);  // 画出下一帧预测坐标
    cout << "(" << point.x << "," << point.y << ") " << endl;
    //polylines(frame, trac, false, Scalar(255, 0, 0));  // 画轨迹

    // --- 下一秒预测位置 ---
    Mat sp = kal.statePost;
    float vel_x = sp.at<float>(2, 0);
    float vel_y = sp.at<float>(3, 0);
    float x_next = center.x + vel_x * 15;  // 位置用实际的
    float y_next = center.y + vel_y * 15;
    circle(frame, Point(x_next, y_next), 5, Scalar(255, 255, 0), -1);  // 画出下一秒预测坐标
}