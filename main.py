import asyncio
from bleak import BleakClient, BleakScanner
import numpy as np
# import matplotlib.pyplot as plt
from bitstring import BitArray
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QObject
from qasync import QEventLoop, asyncClose
import pyqtgraph as pg
import pyaudio
import time
import sys
import ctypes
import wave
from picosdk.ps3000a import ps3000a
from picosdk.ps4000 import ps4000
from picosdk.functions import assert_pico_ok


class PolarVeritySense:
    BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

    PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
    PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

    PPG_START = bytearray([0x02, 0x01, 0x00, 0x01, 0x37, 0x00, 0x01, 0x01, 0x16, 0x00, 0x04, 0x01, 0x04])
    PPG_STOP = bytearray([0x03, 0x01])

    MAX_BUFFER_LEN = 20 * 55

    def __init__(self, device, signal, global_start_time):
        self.device = device
        self.signal = signal
        self.global_start_time = global_start_time

        self.connected = False
        self.started = False

        self.previous_timestamp = -1
        self.t = []

        self.ppg0 = []
        self.ppg1 = []
        self.ppg2 = []
        self.ambient = []

    async def connect(self):
        if not self.connected:
            self.client = BleakClient(self.device)
            await self.client.connect()
            self.connected = True
    
    async def disconnect(self):
        if self.connected:
            await self.client.disconnect()
            self.connected = False

    async def get_battery_level(self):
        data = await self.client.read_gatt_char(PolarVeritySense.BATTERY_LEVEL_UUID)
        return data[0]

    async def start_ppg_stream(self):
        if not self.started:
            await self.client.write_gatt_char(PolarVeritySense.PMD_CONTROL, PolarVeritySense.PPG_START)
            await self.client.start_notify(PolarVeritySense.PMD_DATA, self.decode_data)
            self.started = True
            print("Starting PPG stream")
    
    async def stop_ppg_stream(self):
        if self.started:
            await self.client.stop_notify(PolarVeritySense.PMD_DATA)
            await self.client.write_gatt_char(PolarVeritySense.PMD_CONTROL, PolarVeritySense.PPG_STOP)
            self.started = False
            print("Stopping PPG stream")

    def decode_data(self, sender, data):
        if data[0] != 0x01:
            print("Unexpected measurement type")
        else:
            timestamp = PolarVeritySense.convert_to_unsigned_long(data, 1, 8) / 1e9 + 946684800 + 3.5  # empirically determined
            frame_type = data[9]

            if frame_type != 0x80:
                print("Unexpected frame type")
            else:
                self.ppg0.append(PolarVeritySense.convert_array_to_signed_int(data, 10, 3))
                self.ppg1.append(PolarVeritySense.convert_array_to_signed_int(data, 13, 3))
                self.ppg2.append(PolarVeritySense.convert_array_to_signed_int(data, 16, 3))
                self.ambient.append(PolarVeritySense.convert_array_to_signed_int(data, 19, 3))
                samples_size = 1

                offset = 22
                while offset < len(data):
                    delta_size = data[offset]
                    sample_count = data[offset + 1]
                    offset += 2

                    samples = ''.join(format(byte, '08b')[::-1] for byte in data[offset: offset + (delta_size * sample_count // 2)])
                    for sample in range(0, len(samples), delta_size * 4):
                        deltas = [BitArray(bin=samples[sample + delta_size * i: sample + delta_size * (i + 1)][::-1]).int for i in range(4)]
                        ppg0 = self.ppg0[-1] + deltas[0]
                        ppg1 = self.ppg1[-1] + deltas[1]
                        ppg2 = self.ppg2[-1] + deltas[2]
                        ambient = self.ambient[-1] + deltas[3]

                        self.ppg0.append(ppg0)
                        self.ppg1.append(ppg1)
                        self.ppg2.append(ppg2)
                        self.ambient.append(ambient)
                        samples_size += 1

                    offset += delta_size * sample_count // 2
                
                if self.previous_timestamp == -1:
                    delta = 1 / 55
                    t = np.linspace(timestamp - delta * (samples_size - 1), timestamp, num=samples_size)
                else:
                    delta = (timestamp - self.previous_timestamp) / samples_size
                    t = np.linspace(self.previous_timestamp + delta, timestamp, num=samples_size)
                self.t = np.concatenate((self.t, t - self.global_start_time))
                self.previous_timestamp = timestamp

                self.signal.data.emit(self.t[-PolarVeritySense.MAX_BUFFER_LEN:], self.ppg0[-PolarVeritySense.MAX_BUFFER_LEN:])
                
                # TODO: WRITE DATA

    @staticmethod
    def convert_array_to_signed_int(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=True,
        )

    @staticmethod
    def convert_to_unsigned_long(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=False,
        )


class Microphone:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == 'darwin' else 2
    RATE = 44100
    MAX_BUFFER_LEN = 10 * RATE

    def __init__(self, signal, global_start_time):
        self.signal = signal
        self.global_start_time = global_start_time

        self.p = pyaudio.PyAudio()
        self.wf = wave.open('microphone.wav', 'wb')
        self.wf.setnchannels(Microphone.CHANNELS)
        self.wf.setsampwidth(self.p.get_sample_size(Microphone.FORMAT))
        self.wf.setframerate(Microphone.RATE)

        self.total_frames = 0
        self.t = []
        self.frames = []

        self.stream = self.p.open(format=Microphone.FORMAT, channels=Microphone.CHANNELS, rate=Microphone.RATE, input=True, stream_callback=self.callback)
        self.start_time = time.time()

    def callback(self, in_data, frame_count, time_info, status):
        self.frames = np.concatenate((self.frames, np.frombuffer(in_data, dtype=np.int16)))[-Microphone.MAX_BUFFER_LEN:]
        self.total_frames += frame_count

        start_t = self.start_time + ((self.total_frames - len(self.frames)) / Microphone.RATE) - self.global_start_time
        end_t = self.start_time + (self.total_frames / Microphone.RATE) - self.global_start_time
        self.t = np.linspace(start_t, end_t, num=len(self.frames), endpoint=False)

        self.signal.data.emit(self.t, self.frames)

        self.wf.writeframes(in_data)
        return (in_data, pyaudio.paContinue)
    
    def close(self):
        self.stream.close()
        self.p.terminate() 
        self.wf.close()


# class PicoScope:
#     channel_range = 8  # 10MV, 20MV, 50MV, 100MV, 200MV, 500MV, 1V, 2V, 5V, 10V, 20V, 50V
#     v_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200][channel_range]
    
#     def __init__(self):
#         self.chandle_4000 = ctypes.c_int16()
#         self.status_4000 = {}

#         self.chandle_3000a = ctypes.c_int16()
#         self.status_3000a = {}
    
#     def open(self):
#         self.status_3000a["openunit"] = ps3000a.ps3000aOpenUnit(ctypes.byref(self.chandle_3000a), None)
#         try:
#             assert_pico_ok(self.status_3000a["openunit"])
#         except:
#             powerStatus = self.status_3000a["openunit"]
#             if powerStatus == 286:
#                 self.status_3000a["changePowerSource"] = ps3000a.ps3000aChangePowerSource(self.chandle_3000a, powerStatus)
#             elif powerStatus == 282:
#                 self.status_3000a["changePowerSource"] = ps3000a.ps3000aChangePowerSource(self.chandle_3000a, powerStatus)
#             else:
#                 raise
#             assert_pico_ok(self.status_3000a["changePowerSource"])

#         self.status_4000["openunit"] = ps4000.ps4000OpenUnit(ctypes.byref(self.chandle_4000))
#         assert_pico_ok(self.status_4000["openunit"])

#     def stop(self):
#         self.status_4000["stop"] = ps4000.ps4000Stop(self.chandle_4000)
#         assert_pico_ok(self.status_4000["stop"])
#         self.status_4000["close"] = ps4000.ps4000CloseUnit(self.chandle_4000)
#         assert_pico_ok(self.status_4000["close"])

#         self.status_3000a["stop"] = ps3000a.ps3000aStop(self.chandle_3000a)
#         assert_pico_ok(self.status_3000a["stop"])
#         self.status_3000a["close"] = ps3000a.ps3000aCloseUnit(self.chandle_3000a)
#         assert_pico_ok(self.status_3000a["close"])


class Signal(QObject):
    data = pyqtSignal(object, object)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.pvs = None
        self.microphone = None

        self.graphWidget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphWidget)

        self.plots = {
            "PPG0": self.graphWidget.addPlot(row=0, col=0),
            "Microphone": self.graphWidget.addPlot(row=1, col=0)
        }

        self.curves = {}
        for name, plot in self.plots.items():
            plot.setLabel("left", name)
            plot.setMouseEnabled(x=False, y=False)
            if name != "Microphone":
                plot.setXLink(self.plots["Microphone"])
            self.curves[name] = plot.plot([], [])
        
    async def start(self):
        device = await BleakScanner.find_device_by_name("Polar Sense DE957E2E", timeout=1)

        global_start_time = time.time()

        if device is None:
            print("Polar Sense DE957E2E not found")
        else:
            pvs_signal = Signal()
            pvs_signal.data.connect(self.update_pvs_graph)
            self.pvs = PolarVeritySense(device, pvs_signal, global_start_time)
            await self.pvs.connect()
            print("Battery:", await self.pvs.get_battery_level())
            await self.pvs.start_ppg_stream()

        microphone_signal = Signal()
        microphone_signal.data.connect(self.update_microphone_graph)
        self.microphone = Microphone(microphone_signal, global_start_time)
    
    @asyncClose
    async def closeEvent(self, event):
        if self.pvs is not None:
            await self.pvs.stop_ppg_stream()
            await self.pvs.disconnect()
        if self.microphone is not None:
            self.microphone.close()
    
    def update_pvs_graph(self, t, ppg0):
        self.curves["PPG0"].setData(t, ppg0)

    def update_microphone_graph(self, t, frames):
        self.curves["Microphone"].setData(t, frames) 


if __name__ == "__main__":
    app = QApplication([])

    event_loop = QEventLoop(app)
    asyncio.set_event_loop(event_loop)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)
    
    main_window = MainWindow()
    main_window.show()

    event_loop.create_task(main_window.start())
    event_loop.run_until_complete(app_close_event.wait())
    event_loop.close()
