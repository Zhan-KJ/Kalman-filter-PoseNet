class Kalmanfilter:
    def __init__(self):  # initialization value
        self.x_P = 0.9
        self.x_Q = 0.08
        self.y_P = 0.9
        self.y_Q = 0.08
        self.x_priori_estimated_covariance = 1  # 先驗估計協方差
        self.y_priori_estimated_covariance = 1
        # X估計值
        self.x_estimated_value_dict = {'nose': 0, 'left eye': 0, 'right eye': 0, 'left ear': 0, 'right ear': 0,
                                       'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
                                       'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
                                       'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # X後驗估計協方差
        self.x_post_estimated_covariance_dict = {'nose': 1, 'left eye': 1, 'right eye': 1, 'left ear': 1,
                                                 'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
                                                 'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
                                                 'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
                                                 'left ankle': 1, 'right ankle': 1}
        # Y估計值
        self.y_estimated_value_dict = {'nose': 0, 'left eye': 0, 'right eye': 0, 'left ear': 0, 'right ear': 0,
                                       'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
                                       'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
                                       'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # Y後驗估計協方差
        self.y_post_estimated_covariance_dict = {'nose': 1, 'left eye': 1, 'right eye': 1, 'left ear': 1,
                                                 'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
                                                 'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
                                                 'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
                                                 'left ankle': 1, 'right ankle': 1}

    def x_reset(self, P, Q):  # reset P and Q
        self.x_P = P
        self.x_Q = Q

    def y_reset(self, P, Q):  # reset P and Q
        self.y_P = P
        self.y_Q = Q

    def cal_X(self, current_value, label):  # input current value
        self.current_value = current_value
        self.label = label
        self.x_priori_estimated_covariance = self.x_post_estimated_covariance_dict[label]
        x_kalman_gain = self.x_priori_estimated_covariance / (self.x_priori_estimated_covariance + self.x_P)  #
        output = self.x_estimated_value_dict[label] + x_kalman_gain * (
                    current_value - self.x_estimated_value_dict[label])  #
        self.x_estimated_value_dict[label] = output
        self.x_post_estimated_covariance_dict[label] = (
                                                                   1 - x_kalman_gain) * self.x_priori_estimated_covariance + self.x_Q
        self.x_priori_estimated_covariance = self.x_post_estimated_covariance_dict[label]
        return output  # Kalmanfilter formula

    def cal_Y(self, current_value, label):  # input current value
        self.current_value = current_value
        self.label = label
        self.y_priori_estimated_covariance = self.y_post_estimated_covariance_dict[label]
        y_kalman_gain = self.y_priori_estimated_covariance / (self.y_priori_estimated_covariance + self.y_P)  #
        output = self.y_estimated_value_dict[label] + y_kalman_gain * (
                    current_value - self.y_estimated_value_dict[label])  #
        self.y_estimated_value_dict[label] = output
        self.y_post_estimated_covariance_dict[label] = (
                                                                   1 - y_kalman_gain) * self.y_priori_estimated_covariance + self.y_Q
        self.y_priori_estimated_covariance = self.y_post_estimated_covariance_dict[label]
        return output


class Point_Kalman_process:
    def __init__(self):
        self.kalman_filter_X = Kalmanfilter()
        self.kalman_filter_Y = Kalmanfilter()

    def reset_kalman_filter_X(self, P, Q):
        self.kalman_filter_X.x_reset(P, Q)

    def reset_kalman_filter_Y(self, P, Q):
        self.kalman_filter_Y.y_reset(P, Q)

    def do_kalman_filter(self, X, Y, label):
        X_cal = self.kalman_filter_X.cal_X(X, label)
        Y_cal = self.kalman_filter_Y.cal_Y(Y, label)
        return X_cal, Y_cal
