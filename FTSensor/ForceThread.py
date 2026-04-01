import time
from PyQt5.QtCore import QThread, pyqtSignal
from .rpi_ati_net_ft import NET_FT


class ForceThread(QThread):
    _ft_data = pyqtSignal(list)

    def __init__(self, ftSensorIp, tc3):
        super().__init__()
        self.ip = ftSensorIp
        self._is_running = False
        self.sensor = NET_FT(self.ip)
        self.sensor.set_tare_from_ft()
        self.setObjectName("ForceThread")
        self.tc3 = tc3

    def run(self):
        self._is_running = True
        try:
            self.sensor.start_streaming()

            while self._is_running:
                try:
                    success, ft, status = self.sensor.try_read_ft_streaming(0.01)
                    
                    if success:
                        self._ft_data.emit(ft.tolist())
                        self.writeFT(ft.tolist())
                    else:
                        time.sleep(0.05)

                    # data_record = self.sensor.measurement()
                    # if data_record:
                    #     ft = [data_record.Fx, data_record.Fy, data_record.Fz,
                    #           data_record.Tx, data_record.Ty, data_record.Tz]
                    #     self._ft_data.emit(ft)
                    # else:
                    #     time.sleep(0.05)
                except Exception as e:
                    print("ftSensor 解包错误：", e)
        except Exception as e:
            print("ftSensor 连接错误：", e)
        finally:
            if self.sensor:
                self.sensor.stop_streaming()
                self._is_running = False
                print("ftSensor 流式传输断开")

    def stop(self):
        self.sensor.stop_streaming()
        self._is_running = False
