#include <iostream>
#include<map>
#include<string>
#include<cstdlib>
#include<cstring>
using namespace std;
class Kalmanfiter
{
    public:
        float x_P = 0.9;
        float x_Q = 0.08;
        float y_P = 0.9;
        float y_Q = 0.08;
        float x_priori_estimated_covariance = 1;
        float y_priori_estimated_covariance = 1;
        float output_x;
        float output_y;
        float x_kalman_gain;
        float y_kalman_gain;
        map<string, float> x_estimated_value_map = { {"nose",0},{"left eye",0},{"right eye",0} ,{"left ear",0} ,{"right ear",0}
                                        ,{"left shoulder",0},{"right shoulder",0},{"left elbow",0},{"right elbow",0}
                                        ,{"left wrist",0},{"right wrist",0},{"left hip",0},{"right hip",0}
                                        ,{"left knee",0},{"right knee",0},{"left ankle",0},{"right ankle",0} };

        map<string, float> x_post_estimated_covariance_map = { {"nose",1},{"left eye",1},{"right eye",1} ,{"left ear",1} ,{"right ear",1}
                                                ,{"left shoulder",1},{"right shoulder",1},{"left elbow",1},{"right elbow",1}
                                                ,{"left wrist",1},{"right wrist",1},{"left hip",1},{"right hip",1}
                                                ,{"left knee",1},{"right knee",1},{"left ankle",1},{"right ankle",1} };
        map<string, float> y_estimated_value_map = { {"nose",0},{"left eye",0},{"right eye",0} ,{"left ear",0} ,{"right ear",0}
                                        ,{"left shoulder",0},{"right shoulder",0},{"left elbow",0},{"right elbow",0}
                                        ,{"left wrist",0},{"right wrist",0},{"left hip",0},{"right hip",0}
                                        ,{"left knee",0},{"right knee",0},{"left ankle",0},{"right ankle",0} };

        map<string, float> y_post_estimated_covariance_map = { {"nose",1},{"left eye",1},{"right eye",1} ,{"left ear",1} ,{"right ear",1}
                                                ,{"left shoulder",1},{"right shoulder",1},{"left elbow",1},{"right elbow",1}
                                                ,{"left wrist",1},{"right wrist",1},{"left hip",1},{"right hip",1}
                                                ,{"left knee",1},{"right knee",1},{"left ankle",1},{"right ankle",1} };
        void reset_x(float P, float Q)
        {
            x_P = P;
            x_Q = Q;
        }
        void reset_y(float P, float Q)
        {
            y_P = P;
            y_Q = Q;
        }
        float cal_X(float current_value, const char* label)
        {
            x_priori_estimated_covariance = x_post_estimated_covariance_map[label];
            x_kalman_gain = x_priori_estimated_covariance / (x_priori_estimated_covariance + x_P);
            output_x = x_estimated_value_map[label] + x_kalman_gain * (current_value-x_estimated_value_map[label]);
            x_estimated_value_map[label] = output_x;
            x_post_estimated_covariance_map[label] = (1 - x_kalman_gain) * x_priori_estimated_covariance + x_Q;
            return output_x;
        }
        float cal_Y(float current_value, const char* label)
        {
            y_priori_estimated_covariance = y_post_estimated_covariance_map[label];
            y_kalman_gain = y_priori_estimated_covariance / (y_priori_estimated_covariance + y_P);
            output_y = y_estimated_value_map[label] + y_kalman_gain * (current_value - y_estimated_value_map[label]);
            y_estimated_value_map[label] = output_y;
            y_post_estimated_covariance_map[label] = (1 - y_kalman_gain) * y_priori_estimated_covariance + y_Q;
        
            return output_y;
        }
};
class Point_Kalman_process
{
    Kalmanfiter kalmanfilter_X;
    Kalmanfiter kalmanfilter_Y;
    public:
        float X_cal;
        float Y_cal;
        void reset_kalman_filter_X(float P, float Q)
        {
            kalmanfilter_X.reset_x(P, Q);
        }
        void reset_kalman_filter_Y(float P, float Q)
        {
            kalmanfilter_Y.reset_y(P, Q);
        }
        float do_kalman_filter_x(float X,const char* label)
        {
            X_cal = kalmanfilter_X.cal_X(X, label);
            return X_cal;
        }
	float do_kalman_filter_y(float Y,const char* label)
        {
            Y_cal = kalmanfilter_Y.cal_Y(Y, label);
            return Y_cal;
        }

};


extern "C"
{
    int out;
    Point_Kalman_process kalman;
    int call_x(float x,const char *label)
    {
	out=kalman.do_kalman_filter_x(x,label);	
	return out;	
    }
    int call_y(float y,const char *label)
    {
	out=kalman.do_kalman_filter_y(y,label);
	return out;	
    }
}


