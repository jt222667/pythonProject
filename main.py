import numpy as np
import matplotlib.pyplot as plt


class SingleLinkArm:
    def __init__(self, length, dt):
        self.length = length
        self.angle = 0
        self.angular_velocity = 0
        self.dt = dt

    def update(self, torque):
        inertia = 1
        damping = 1
        angular_acceleration = (torque - damping * self.angular_velocity) / inertia
        self.angular_velocity += angular_acceleration * self.dt
        self.angle += self.angular_velocity * self.dt
        return angular_acceleration

    def get_position(self):
        x = self.length * np.cos(self.angle)
        y = self.length * np.sin(self.angle)
        return x, y


# 滑模控制器（符号函数版）
class SlidingModeControllerSign:
    def __init__(self, lambda_param, k, dt):
        self.lambda_param = lambda_param
        self.k = k
        self.LAST_angular = 0.0
        self.LAST_desired_angle = 0.0
        self.dt = dt

    def control(self, angle, desired_angle):
        angular_velocity = (angle - self.LAST_angular) / self.dt
        desired_angular_velocity = (desired_angle - self.LAST_desired_angle) / self.dt

        e = angle - desired_angle
        e_dot = angular_velocity - desired_angular_velocity
        s = e_dot + self.lambda_param * e
        u = -self.k * np.sign(s)

        self.LAST_angular = angle
        self.LAST_desired_angle = desired_angle

        return u


# 滑模控制器（饱和函数版）
class SlidingModeControllerSaturation:
    def __init__(self, lambda_param, k, epsilon, dt):
        self.lambda_param = lambda_param
        self.k = k
        self.epsilon = epsilon
        self.LAST_angular = 0.0
        self.LAST_desired_angle = 0.0
        self.dt = dt

    def control(self, angle, desired_angle):
        angular_velocity = (angle - self.LAST_angular) / self.dt
        desired_angular_velocity = (desired_angle - self.LAST_desired_angle) / self.dt

        e = angle - desired_angle
        e_dot = angular_velocity - desired_angular_velocity
        s = e_dot + self.lambda_param * e
        # 使用饱和函数
        u = -self.k * np.clip(s / self.epsilon, -1, 1)

        self.LAST_angular = angle
        self.LAST_desired_angle = desired_angle

        return u


# 模拟参数
dt = 0.01
length = 1.0
time = np.arange(0, 50, dt)

# 正弦轨迹参数
amplitude = 0.5
frequency = 0.2
desired_angle = amplitude * np.sin(2 * np.pi * frequency * time)
desired_angular_velocity = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time)

# 初始化机械臂和控制器
arm_sign = SingleLinkArm(length, dt)
arm_saturation = SingleLinkArm(length, dt)

controller_sign = SlidingModeControllerSign(lambda_param=10, k=5, dt=dt)
controller_saturation = SlidingModeControllerSaturation(lambda_param=10, k=5, epsilon=0.1, dt=dt)

# 存储结果
angles_sign = []
errors_sign = []
angles_saturation = []
errors_saturation = []

# 仿真循环
for t, angle_d, angular_velocity_d in zip(time, desired_angle, desired_angular_velocity):
    # 使用符号函数控制器
    torque_sign = controller_sign.control(arm_sign.angle, angle_d)
    arm_sign.update(torque_sign)
    angles_sign.append(arm_sign.angle)
    errors_sign.append(arm_sign.angle - angle_d)

    # 使用饱和函数控制器
    torque_saturation = controller_saturation.control(arm_saturation.angle, angle_d)
    arm_saturation.update(torque_saturation)
    angles_saturation.append(arm_saturation.angle)
    errors_saturation.append(arm_saturation.angle - angle_d)

# 绘制结果
plt.figure(figsize=(12, 8))

# 角度对比
plt.subplot(2, 1, 1)
plt.plot(time, desired_angle, 'r', label='Desired Angle')
plt.plot(time, angles_sign, 'b', label='Actual Angle (Sign)')
plt.plot(time, angles_saturation, 'g--', label='Actual Angle (Saturation)')
plt.ylabel('Angle (rad)')
plt.title('Sliding Mode Control of Single Link Arm')
plt.legend()

# 误差对比
plt.subplot(2, 1, 2)
plt.plot(time, errors_sign, 'b', label='Tracking Error (Sign)')
plt.plot(time, errors_saturation, 'g--', label='Tracking Error (Saturation)')
plt.xlabel('Time (s)')
plt.ylabel('Error (rad)')
plt.title('Tracking Error Comparison')
plt.legend()

plt.tight_layout()
plt.show()