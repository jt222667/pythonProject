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


# 滑模控制器
class SlidingModeController:
    def __init__(self, lambda_param, k, dt):
        self.lambda_param = lambda_param
        self.k = k
        self.LAST_angular = 0.0
        self.LAST_desired_angle = 0.0
        self.dt = dt

    def control(self, angle, desired_angle):
        angular_velocity = (angle - self.LAST_angular)/self.dt
        desired_angular_velocity = (desired_angle - self.LAST_desired_angle)/self.dt


        e = angle - desired_angle
        e_dot = angular_velocity - desired_angular_velocity
        s = e_dot + self.lambda_param * e
        u = -self.k * np.sign(s)

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

# 初始化控制器和机械臂
arm = SingleLinkArm(length, dt)
controller = SlidingModeController(lambda_param=10, k=5, dt=dt)

# 存储结果
angles = []
errors = []

# 仿真循环
for t, angle_d, angular_velocity_d in zip(time, desired_angle, desired_angular_velocity):
    # 计算控制输入
    torque = controller.control(arm.angle,angle_d)

    # 更新机械臂状态
    arm.update(torque)

    # 存储结果
    angles.append(arm.angle)
    errors.append(arm.angle - angle_d)

# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, desired_angle, 'r', label='Desired Angle')
plt.plot(time, angles, 'b', label='Actual Angle')
plt.ylabel('Angle (rad)')
plt.title('Sliding Mode Control of Single Link Arm')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, errors, 'g')
plt.xlabel('Time (s)')
plt.ylabel('Error (rad)')
plt.title('Tracking Error')

plt.tight_layout()
plt.show()