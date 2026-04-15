import time
import numpy as np
from FTSensor.ForceThread import ForceThread
from UR_Base import UR_BASE


class ImpedanceController:
    """
    阻抗控制器类 - 真正的阻抗控制（位置→力）
    
    核心原理：
    1. 检测位置偏差：Δx = target_position - current_position
    2. 计算期望力：F = K * Δx + D * Δv
    3. 转换为关节力矩：τ = J^T * F
    4. 使用关节力矩控制
    
    其中 K 为刚度矩阵，D 为阻尼矩阵，J 为雅可比矩阵
    """
    
    def __init__(self, robot_ip, ft_sensor_ip=None, tc3=None):
        """
        初始化阻抗控制器
        
        参数:
            robot_ip: UR机器人IP地址
            ft_sensor_ip: 力传感器IP地址（可选，用于力反馈）
            tc3: TwinCAT控制器实例（可选）
        """
        self.robot = UR_BASE(robot_ip)
        
        self.ft_thread = None
        if ft_sensor_ip is not None:
            self.ft_thread = ForceThread(ft_sensor_ip, tc3)
            self.ft_thread._ft_data.connect(self._on_ft_data_received)
        
        self.current_ft = np.zeros(6)
        self.target_position = None
        self.current_position = None
        self.current_velocity = np.zeros(6)
        self.last_position = None
        
        self.K = np.array([500.0, 500.0, 500.0, 50.0, 50.0, 50.0])
        self.D = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        
        self._is_running = False
        self.control_frequency = 500
        self.dt = 1.0 / self.control_frequency
        
    def _on_ft_data_received(self, ft_data):
        """
        力传感器数据回调函数
        
        参数:
            ft_data: [Fx, Fy, Fz, Tx, Ty, Tz] 力和扭矩数据
        """
        self.current_ft = np.array(ft_data)
        
    def set_target_position(self, position):
        """
        设置目标位置（弹簧平衡位置）
        
        参数:
            position: [x, y, z, rx, ry, rz] 目标位姿
        """
        self.target_position = np.array(position)
        print(f"目标位置已设置: {self.target_position}")
        
    def set_impedance_parameters(self, stiffness=None, damping=None):
        """
        设置阻抗参数
        
        参数:
            stiffness: 刚度矩阵 [Kx, Ky, Kz, Krx, Kry, Krz] (N/m, Nm/rad)
            damping: 阻尼矩阵 [Dx, Dy, Dz, Drx, Dry, Drz] (Ns/m, Nms/rad)
        """
        if stiffness is not None:
            self.K = np.array(stiffness)
        if damping is not None:
            self.D = np.array(damping)
        print(f"刚度参数: {self.K}")
        print(f"阻尼参数: {self.D}")
        
    def get_jacobian(self):
        """
        获取雅可比矩阵
        
        返回:
            J: 6x6 雅可比矩阵
        """
        jacobian = self.robot.getJacobian()
        return np.array(jacobian)   
    
    def calculate_cartesian_error(self):
        """
        计算笛卡尔空间的位置误差和速度误差
        
        返回:
            position_error: 位置误差 [dx, dy, dz, drx, dry, drz]
            velocity_error: 速度误差 [vx, vy, vz, vrx, vry, vrz]
        """
        self.current_position = np.array(self.robot.get_tcp())
        
        position_error = self.target_position - self.current_position
        
        if self.last_position is not None:
            self.current_velocity = (self.current_position - self.last_position) / self.dt
        else:
            self.current_velocity = np.zeros(6)
        self.last_position = self.current_position.copy()
        
        velocity_error = -self.current_velocity
        
        return position_error, velocity_error
        
    def calculate_impedance_force(self, position_error, velocity_error):
        """
        根据阻抗模型计算期望力
        
        F = K * Δx + D * Δv
        
        参数:
            position_error: 位置误差
            velocity_error: 速度误差
            
        返回:
            desired_force: 期望力/力矩 [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        desired_force = self.K * position_error + self.D * velocity_error
        
        return desired_force
        
    def force_to_joint_torque(self, cartesian_force):
        """
        将笛卡尔力转换为关节力矩
        
        τ = J^T * F
        
        参数:
            cartesian_force: 笛卡尔力/力矩 [Fx, Fy, Fz, Tx, Ty, Tz]
            
        返回:
            joint_torques: 关节力矩 [τ1, τ2, τ3, τ4, τ5, τ6]
        """
        J = self.get_jacobian()
        
        joint_torques = J.T @ cartesian_force
        
        return joint_torques
    
    def get_gravity_torques(self):
        """
        获取重力补偿力矩
        
        返回:
            gravity_torques: 重力补偿力矩
        """
        return np.array(self.robot.getJointTorques())
    
    def start(self):
        """
        启动阻抗控制
        """
        if self.target_position is None:
            self.target_position = np.array(self.robot.get_tcp())
            print(f"使用当前位置作为目标位置: {self.target_position}")
            
        self._is_running = True
        
        if self.ft_thread is not None:
            self.ft_thread.start()
        
        print("=" * 50)
        print("阻抗控制已启动")
        print("控制模式: 关节力矩控制")
        print("机器人将表现出弹簧行为")
        print("=" * 50)
        
        time.sleep(0.2)
        
        try:
            while self._is_running:
                start_time = time.time()
                
                position_error, velocity_error = self.calculate_cartesian_error()
                
                desired_force = self.calculate_impedance_force(position_error, velocity_error)
                
                joint_torques = self.force_to_joint_torque(desired_force)
                
                max_torque = np.array([50.0, 50.0, 50.0, 30.0, 30.0, 30.0])
                joint_torques = np.clip(joint_torques, -max_torque, max_torque)
                
                self.robot.setJointTorque(joint_torques)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n用户中断，停止控制")
        except Exception as e:
            print(f"控制错误: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """
        停止阻抗控制
        """
        self._is_running = False
        if self.ft_thread is not None:
            self.ft_thread.stop()
            self.ft_thread.wait()
        self.robot.stop_robot()
        print("阻抗控制已停止")
        
    def disconnect(self):
        """
        断开连接
        """
        self.stop()
        self.robot.disconnect()
        print("已断开所有连接")


def main():
    """
    主函数 - 阻抗控制示例
    """
    ROBOT_IP = "192.168.111.10"
    FT_SENSOR_IP = "192.168.111.20"
    
    controller = ImpedanceController(ROBOT_IP, FT_SENSOR_IP)
    
    target_pos = np.array([0.4, -0.2, 0.3, 0.0, 3.14, 0.0])
    controller.set_target_position(target_pos)
    
    stiffness = [300.0, 300.0, 300.0, 30.0, 30.0, 30.0]
    damping = [30.0, 30.0, 30.0, 3.0, 3.0, 3.0]
    controller.set_impedance_parameters(stiffness=stiffness, damping=damping)
    
    print("=" * 50)
    print("阻抗控制示例 - 关节力矩控制")
    print("=" * 50)
    print(f"目标位置: {target_pos}")
    print(f"刚度参数: {stiffness}")
    print(f"阻尼参数: {damping}")
    print("=" * 50)
    print("控制原理:")
    print("  1. 检测位置偏差 Δx = target - current")
    print("  2. 计算期望力 F = K*Δx + D*Δv")
    print("  3. 转换为关节力矩 τ = J^T * F")
    print("  4. 输出关节力矩控制")
    print("=" * 50)
    print("机器人将像弹簧一样响应位置偏移")
    print("按 Ctrl+C 停止控制")
    print("=" * 50)
    
    try:
        controller.start()
    except Exception as e:
        print(f"控制错误: {e}")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
