// 最终考核
// 一帧一坐标一半径 证明没有缺失识别和误识别；
// 绿色为橘子识别的主体，绿色数字为当前帧数；
// 青色为下一秒预测位置；
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/types_c.h>
#include<ctime>
#include<queue>

using namespace std;
using namespace cv;

double fps;
Mat frame, ero2;  // 帧、二值图
int frame_counter = 1;  // 循环播放视频
int frame_number = 0;  // 记帧数
const char *threshold_title = "Threshold Value:";
const char *output_win = "output win";
const char *result_win = "result win";

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
Point2f center; // 当前帧橘子中心坐标
float radius;  // 当前帧橘子半径

KalmanFilter kal(4, 2, 0);  // 滤波器
Mat state(4, 1, CV_32F);  // 状态向量(x,y,vx,vy)
Mat u(2, 1, CV_32F);
Mat measurement;  // 测量向量(x,y)
Mat predp;  // 预测坐标
Mat preVel(2, 1, CV_32F);
queue<Point> predTrac; // 用于在当前帧之后的第15帧画出预测位置，进行对比
vector<Point> trac; // 预测轨迹

int aleft = 0, aright = 0;  // 判断换向
Point2f precenter(0, 0);  // 前一帧坐标

void TrackBack_Demo(int, void *){};  // TrackBarCallBack
void OrangeDetection(int);  // 检测橘子
void Judge_Change_Direction(Point2f); // 判断换向
void InitKalmanFilter();  // 初始化卡尔曼滤波器
void Prediction_Update();  // 预测
Mat polyfit(const vector<double> &x, const vector<double> &y, int degree);  // 自定义多项式拟合
void Polynomial_Fitting();  // 多项式拟合预测
Point FirstOrder_MarkovPredict(Point currentState, Point previousState, double dt);  // 一阶马尔科夫
Point SecondOrder_MarkovPredict(Point currentState, Point previousState, Point prePreviousState, double dt); // 二阶马尔科夫
void MarkovPrediction();  // 马尔科夫预测
// void LRPrediciton();  // 逻辑回归

int main(){
    VideoCapture video("orange2.mp4");

    if(video.isOpened()){
        cout << "Success!" << endl;
    }
    else{
        cout << "Fault!" << endl;
        return -1;
    }

    video >> frame;

    fps = video.get(CAP_PROP_FPS);
    cout << "fps:" << fps << endl;

    namedWindow(output_win, WINDOW_AUTOSIZE);
    namedWindow(result_win, WINDOW_AUTOSIZE);

    createTrackbar(threshold_title, output_win, 0, 255, TrackBack_Demo);
    setTrackbarPos(threshold_title, output_win, 202);  // 171

    InitKalmanFilter();  // 初始化卡尔曼滤波器

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
        frame_number += 1;
        if (frame_counter == int(video.get(cv::CAP_PROP_FRAME_COUNT)))
        {
            frame_counter = 0;
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        cout << "1.frame counter:" << frame_counter << endl;

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
        cout << "2.(" << center.x << "," << center.y << ") " << radius << endl;
    }

    if (contours.size() == 0)  // 这里要打印出消失的橘子
    {
        center = trac.back();  // 消失的帧 用前一时刻预测的位置
        circle(frame, center, 3, Scalar(0, 255, 0), -1);
        circle(frame, center, radius, Scalar(0, 255, 0), 4);
        cout << "9.disappeared orange:" << "(" << center.x << "," << center.y << ") " << radius << endl;
    }

    string str = to_string(frame_counter);
    putText(frame, str, Point(center.x - 10, center.y + 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);  // 显示帧数
}

void InitKalmanFilter() {
    // 初始化卡尔曼滤波器
    measurement = Mat::zeros(2, 1, CV_32F);
    // Mat B = (Mat_<double>(4, 2) << 0.5, 0, 0, 0.5, 1, 0, 0, 1);
    double dt = 1.0 / fps;

    kal.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);  // 2D状态转移矩阵
    setIdentity(kal.measurementMatrix);  // 测量矩阵
    // setIdentity(kal.processNoiseCov, Scalar::all(1e-3));  // 过程噪声协方差矩阵，越大越滞后
    kal.processNoiseCov = (Mat_<float>(4, 4) << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1);
    // setIdentity(kal.measurementNoiseCov, Scalar::all(1e-3)); // 测量噪声协方差矩阵
    kal.measurementNoiseCov = (Mat_<float>(2, 2) << 0.01, 0, 0, 0.01);
    setIdentity(kal.errorCovPost, Scalar::all(1)); // 后验协方差矩阵
    // kal.controlMatrix = B;

    state.at<float>(0) = 966.695;  // 初始化x坐标
    state.at<float>(1) = 77.8444;  // 初始化y坐标
    //state.at<float>(0) = 0;  // 初始化x坐标
    //state.at<float>(1) = 0;  // 初始化y坐标
    state.at<float>(2) = 20;  // 初始化x轴速度
    state.at<float>(3) = 20;  // 初始化y轴速度
    kal.statePost = state;  // 初始化状态向量
}

void Judge_Change_Direction(Point2f curCenter) {
    // --- 判断换向 ---
    if(precenter.x != 0 && precenter.y != 0 && curCenter.x < precenter.x && aleft < 3) {  // 开始向左的时刻。aleft、aright控制显示的时间
        putText(frame, "To the left!", Point(curCenter.x - 80, curCenter.y + 60), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
        aleft ++;
        aright = 0;
    }
    else if(precenter.x != 0 && precenter.y != 0 && curCenter.x > precenter.x && aright < 3) {  // 开始向右的时刻
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
    trac.push_back(point); // 保留预测位置

    circle(frame, point, 5, Scalar(255, 0, 0), -1);  // 画出下一帧预测坐标。蓝色实心圆
    cout << "3.(" << point.x << "," << point.y << ") " << endl;
    //polylines(frame, trac, false, Scalar(255, 0, 0));  // 画轨迹

    // 多项式预测下一秒，即15帧。黑色实心圆
    //Polynomial_Fitting();

    // 马尔科夫预测下一秒，即15帧。黑色圆环
    // MarkovPrediction();

    // 匀速模型预测下一秒，即15帧。青色实心圆
    Mat sp = kal.statePost;
    float pos_x = sp.at<float>(0, 0);
    float pos_y = sp.at<float>(1, 0);
    float vel_x = sp.at<float>(2, 0); // 自带方向
    float vel_y = sp.at<float>(3, 0);
    float x_next = pos_x + vel_x / fps * 14;  // 位置
    float y_next = pos_y + vel_y / fps * 14;
    if(frame_number >= 16) {
        Point temp = predTrac.front();
        predTrac.pop();
        circle(frame, temp, 10, Scalar(0, 0, 0), 3);  // 预测的坐标。黑色圆环
    }
    predTrac.push(Point(x_next, y_next));
    cout << "4.(" << x_next << "," << y_next << ") " << endl;
    cout << "vel: " << vel_x << " " << vel_y << endl;
    cout << "======" << endl;
    circle(frame, Point(x_next, y_next), 5, Scalar(255, 255, 0), -1); // 画出下一秒，即15帧预测坐标
}

Mat polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
    int n = x.size();
    cv::Mat X(n, degree + 1, CV_64F);
    cv::Mat Y(n, 1, CV_64F);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= degree; j++) {
            X.at<double>(i, j) = std::pow(x[i], j);
        }
        Y.at<double>(i, 0) = y[i];
    }

    cv::Mat coeffs;
    cv::solve(X, Y, coeffs, cv::DecompTypes::DECOMP_QR);
    return coeffs;
}

void Polynomial_Fitting() {
    int degree = 2;  // 多项式次数
    Mat coe;  // 拟合的系数
    vector<double> x, y;
    vector<Point> trac_clone = trac;

    if (trac_clone.size() > degree)
    {
        for (int i = 0; i < 14; i++) {
            for (const auto &point : trac_clone)
            {
                x.push_back(point.x);
                y.push_back(point.y);
            }
            coe = polyfit(x, y, degree);

            if (!coe.empty())
            {
                double t = trac_clone.size();
                double prex = 0.0, prey = 0.0;
                for (int j = 0; j <= degree; j++)
                {
                    prex += coe.at<double>(j, 0) * pow(t, j);
                    prey += coe.at<double>(j, 0) * pow(t, j);
                }
                Point2f pred(center.x + prex, center.y + prey);
                trac_clone.push_back(pred);
            }
            else
            {
                cout << "无法拟合！" << endl;
            }
        }
        cout << "4." << "(" << trac_clone.back().x << ", " << trac_clone.back().y << ") " << endl;
        circle(frame, trac_clone.back(), 5, (0, 0, 0), -1);
    }
}

Point FirstOrder_MarkovPredict(Point currentState, Point previousState, double dt) {
    // 一阶马尔科夫
    Point nextState;
    nextState.x = 2 * currentState.x - previousState.x; // 位置状态与前一时刻有关
    nextState.y = 2 * currentState.y - previousState.y;
    return nextState;
}

Point SecondOrder_MarkovPredict(Point currentState, Point previousState, Point prePreviousState, double dt) {
    // 二阶马尔科夫
    Point nextState;
    nextState.x = 3 * currentState.x - 3 * previousState.x + prePreviousState.x; // 位置状态与前两时刻有关
    nextState.y = 3 * currentState.y - 3 * previousState.y + prePreviousState.y;
    return nextState;
}


void MarkovPrediction() {
    // 马尔科夫预测
    vector<Point> trac_clone = trac;

    // 二阶马尔科夫
    if (trac_clone.size() >= 3)
    {
        for (int i = 0; i < 14; i++) {
            Point currentState = trac_clone.back();
            Point preState = trac_clone[trac_clone.size() - 2];
            Point prePreState = trac_clone[trac_clone.size() - 3];
            Point predpositon = SecondOrder_MarkovPredict(currentState, preState, prePreState, 1.0 / fps);
            trac_clone.push_back(predpositon);
        }
        cout << "4." << "(" << trac_clone.back().x << ", " << trac_clone.back().y << ") " << endl;
        circle(frame, trac_clone.back(), 8, (0, 0, 0), 3);
    }
    /*
    // 一阶马尔科夫
    if (trac_clone.size() >= 2)
    {
        for (int i = 0; i < 14; i++) {
            Point currentState = trac_clone.back();
            Point preState = trac_clone[trac_clone.size() - 2];
            Point predpositon = FirstOrder_MarkovPredict(currentState, preState, 1.0 / fps);
            trac_clone.push_back(predpositon);
        }
        cout << "4." << "(" << trac_clone.back().x << ", " << trac_clone.back().y << ") " << endl;
        circle(frame, trac_clone.back(), 8, (0, 0, 0), 3);
    }
    */
}



/*
void LRPrediciton() {
    if(trac.size() > 3) {
        Mat features(trac.size(), 1, CV_32F);
        Mat label1(trac.size(), 1, CV_32F);
        Mat label2(trac.size(), 1, CV_32F);

        for (int i = 0; i < trac.size(); i++) {
            features.at<float>(i, 0) = i + 1;
            label1.at<float>(i, 0) = trac[i].x;
            label2.at<float>(i, 1) = trac[i].y;
        }

        // x的预测
        Ptr<ml::LogisticRegression> LR1 = ml::LogisticRegression::create();
        TermCriteria termCrit1(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 1e-6);
        LR1->setTermCriteria(termCrit1);
        LR1->setLearningRate(0.1);
        LR1->setIterations(1000);
        LR1->setRegularization(cv::ml::LogisticRegression::REG_L2);
        LR1->setTrainMethod(cv::ml::LogisticRegression::BATCH);

        Ptr<ml::TrainData> trainData1 = ml::TrainData::create(features, ml::ROW_SAMPLE, label1);
        LR1->train(trainData1);

        // y的预测
        Ptr<ml::LogisticRegression> LR2 = ml::LogisticRegression::create();
        TermCriteria termCrit2(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 1e-6);
        LR2->setTermCriteria(termCrit2);
        LR2->setLearningRate(0.1);
        LR2->setIterations(1000);
        LR2->setRegularization(cv::ml::LogisticRegression::REG_L2);
        LR2->setTrainMethod(cv::ml::LogisticRegression::BATCH);

        Ptr<ml::TrainData> trainData2 = ml::TrainData::create(features, ml::ROW_SAMPLE, label2);
        LR2->train(trainData2);

        Mat testdata(14, 1, CV_32F);
        for (int i = 1; i <= 14; i++) {
            testdata.at<float>(i - 1) = trac.size() + i;
        }
        Mat prediction1(14, 1, CV_32F);
        Mat prediction2(14, 1, CV_32F);
        LR1->predict(testdata, prediction1);
        LR2->predict(testdata, prediction2);

        float predx = prediction1.at<double>(13, 0);
        float predy = prediction2.at<double>(13, 0);

        //cout << prediction1 << endl;

        cout << "===(" << predx << "," << predy << ") "<< endl;
    
        //circle(frame, Point(prediction1.at<float>(13, 0), prediction2.at<float>(13, 0)), 8, (0, 0, 0), 3);

    }
    
}

*/