import asyncio
from bleak import BleakClient, BleakScanner
import numpy as np
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
import threading
import os
from picosdk.ps3000a import ps3000a
from picosdk.ps4000 import ps4000
from picosdk.functions import assert_pico_ok


DISPLAY_TIME = 10
SAVE_TIME = 300


class PolarVeritySense:
    BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

    PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
    PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

    PPG_START = bytearray([0x02, 0x01, 0x00, 0x01, 0x37, 0x00, 0x01, 0x01, 0x16, 0x00, 0x04, 0x01, 0x04])
    PPG_STOP = bytearray([0x03, 0x01])

    MAX_EMIT_LEN = DISPLAY_TIME * 2 * 55

    def __init__(self, device, signal, global_start_time, session_dir):
        self.device = device
        self.signal = signal
        self.global_start_time = global_start_time
        self.session_dir = session_dir

        self.connected = False
        self.started = False

        self.previous_timestamp = -1
        self.t = []

        self.save_t = 0
        self.save_index = 0

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
            np.save(self.session_dir + "/pvs" , np.stack((self.t, self.ppg0, self.ppg1, self.ppg2, self.ambient), axis=1), allow_pickle=False)

    def decode_data(self, sender, data):
        if data[0] != 0x01:
            print("Unexpected measurement type")
        else:
            timestamp = PolarVeritySense.convert_to_unsigned_long(data, 1, 8) / 1e9 + 946684800 + 4.4  # empirically determined
            frame_type = data[9]

            # print(time.time() - timestamp)  # approx. 0.5

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

                self.signal.data.emit(np.stack((self.t, self.ppg0), axis=1)[-PolarVeritySense.MAX_EMIT_LEN:])

                if self.t[-1] > self.save_t + SAVE_TIME:
                    np.save(
                        self.session_dir + "/tmp/pvs_" + str(self.save_t),
                        np.stack((self.t, self.ppg0, self.ppg1, self.ppg2, self.ambient), axis=1)[self.save_index:],
                        allow_pickle=False
                    )
                    self.save_t += SAVE_TIME
                    self.save_index = len(self.t)

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
    MAX_EMIT_LEN = DISPLAY_TIME * RATE

    def __init__(self, signal, session_dir):
        self.signal = signal

        self.p = pyaudio.PyAudio()
        self.wf = wave.open(session_dir + '/microphone.wav', 'wb')
        self.wf.setnchannels(Microphone.CHANNELS)
        self.wf.setsampwidth(self.p.get_sample_size(Microphone.FORMAT))
        self.wf.setframerate(Microphone.RATE)

        self.total_frames = 0
        self.t = []
        self.frames = []

        self.stream = self.p.open(format=Microphone.FORMAT, channels=Microphone.CHANNELS, rate=Microphone.RATE, input=True, stream_callback=self.callback)
        self.start_time = time.time()

    def callback(self, in_data, frame_count, time_info, status):
        self.frames = np.concatenate((self.frames, np.frombuffer(in_data, dtype=np.int16)))[-Microphone.MAX_EMIT_LEN:]
        self.total_frames += frame_count

        self.t = np.linspace((self.total_frames - len(self.frames)) / Microphone.RATE, self.total_frames / Microphone.RATE, num=len(self.frames), endpoint=False)

        self.signal.data.emit(np.stack((self.t, self.frames), axis=1))

        self.wf.writeframes(in_data)
        return (in_data, pyaudio.paContinue)
    
    def close(self):
        self.stream.close()
        self.p.terminate() 
        self.wf.close()


class PicoScope:
    channel_range = 8  # 10MV, 20MV, 50MV, 100MV, 200MV, 500MV, 1V, 2V, 5V, 10V, 20V, 50V
    v_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200][channel_range]

    sizeOfOneBuffer = 5000
    numBuffersToCapture = 1800
    totalSamples = sizeOfOneBuffer * numBuffersToCapture

    sampleInterval = ctypes.c_int32(200)
    sampleUnits = 3  # FS, PS, NS, US, MS, S
    actualSampleInterval = sampleInterval.value / 1e6

    MAX_EMIT_LEN = int(1e6 / sampleInterval.value * DISPLAY_TIME * 2)
    
    def __init__(self, ps, signal, global_start_time, session_dir):
        self.ps = ps
        self.signal = signal
        self.global_start_time = global_start_time
        self.session_dir = session_dir

        self.chandle = ctypes.c_int16()
        self.status = {}

        self.kill = False

    def open(self):
        if self.ps == ps3000a:
            self.status["openunit"] = self.ps.ps3000aOpenUnit(ctypes.byref(self.chandle), None)
            try:
                assert_pico_ok(self.status["openunit"])
            except:
                powerStatus = self.status["openunit"]
                if powerStatus == 286:
                    self.status["changePowerSource"] = self.ps.ps3000aChangePowerSource(self.chandle, powerStatus)
                elif powerStatus == 282:
                    self.status["changePowerSource"] = self.ps.ps3000aChangePowerSource(self.chandle, powerStatus)
                else:
                    raise
                assert_pico_ok(self.status["changePowerSource"])

            self.status["setChA"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChA"])
            self.status["setChB"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChB"])
            self.status["setChC"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChC"])
            self.status["setChD"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_D'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChD"])

            self.bufferAMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferBMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferCMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferDMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)

            self.status["setDataBuffersA"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersA"])
            self.status["setDataBuffersB"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'],
                self.bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersB"])
            self.status["setDataBuffersC"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'],
                self.bufferCMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersC"])
            self.status["setDataBuffersD"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_D'],
                self.bufferDMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersD"])
        else:
            self.status["openunit"] = self.ps.ps4000OpenUnit(ctypes.byref(self.chandle))
            assert_pico_ok(self.status["openunit"])
        
            self.status["setChA"] = self.ps.ps4000SetChannel(self.chandle, self.ps.PS4000_CHANNEL['PS4000_CHANNEL_A'], 1, 1, PicoScope.channel_range)
            assert_pico_ok(self.status["setChA"])
            self.status["setChB"] = self.ps.ps4000SetChannel(self.chandle, self.ps.PS4000_CHANNEL['PS4000_CHANNEL_B'], 1, 1, PicoScope.channel_range)
            assert_pico_ok(self.status["setChB"])

            self.bufferAMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferBMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)

            self.status["setDataBuffersA"] = self.ps.ps4000SetDataBuffers(
                self.chandle,
                self.ps.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer
            )
            assert_pico_ok(self.status["setDataBuffersA"])
            self.status["setDataBuffersB"] = self.ps.ps4000SetDataBuffers(
                self.chandle,
                self.ps.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                self.bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer
            )
            assert_pico_ok(self.status["setDataBuffersB"])
    
    def streaming_callback(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.wasCalledBack = True
        destEnd = self.nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples

        self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        self.bufferCompleteB[self.nextSample:destEnd] = self.bufferBMax[startIndex:sourceEnd]
        self.bufferCompleteA[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
        self.bufferCompleteB[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
        if self.ps == ps3000a:
            self.bufferCompleteC[self.nextSample:destEnd] = self.bufferCMax[startIndex:sourceEnd]
            self.bufferCompleteD[self.nextSample:destEnd] = self.bufferDMax[startIndex:sourceEnd]
            self.bufferCompleteC[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
            self.bufferCompleteD[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value

        self.nextSample += noOfSamples
        if autoStop:
            self.autoStopOuter = True

    def run(self):
        if self.ps == ps3000a:
            self.status["runStreaming"] = self.ps.ps3000aRunStreaming(
                self.chandle,
                ctypes.byref(PicoScope.sampleInterval),
                PicoScope.sampleUnits,
                0,
                PicoScope.totalSamples,
                1,
                1, 
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'],
                PicoScope.sizeOfOneBuffer
            )
        else:
            self.status["runStreaming"] = self.ps.ps4000RunStreaming(
                self.chandle,
                ctypes.byref(PicoScope.sampleInterval),
                PicoScope.sampleUnits,
                0,
                PicoScope.totalSamples,
                1,
                1,
                PicoScope.sizeOfOneBuffer
            )
        assert_pico_ok(self.status["runStreaming"])
        print(self.ps.name + ": Capturing at", 1 / PicoScope.actualSampleInterval, "Hz for", PicoScope.totalSamples * PicoScope.actualSampleInterval, "s")
        self.t = time.time() - self.global_start_time + np.linspace(0, (PicoScope.totalSamples - 1) * PicoScope.actualSampleInterval, num=PicoScope.totalSamples)
        self.save_t = 0
        self.save_index = 0

        self.bufferCompleteA = np.zeros(shape=PicoScope.totalSamples)
        self.bufferCompleteB = np.zeros(shape=PicoScope.totalSamples)
        if self.ps == ps3000a:
            self.bufferCompleteC = np.zeros(shape=PicoScope.totalSamples)
            self.bufferCompleteD = np.zeros(shape=PicoScope.totalSamples)

        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False

        if self.ps == ps3000a:
            self.maxADC = ctypes.c_int16()
            self.status["maximumValue"] = self.ps.ps3000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
            assert_pico_ok(self.status["maximumValue"])
        else:
            self.maxADC = ctypes.c_int16(32767)

        self.cFuncPtr = self.ps.StreamingReadyType(self.streaming_callback)

        while self.nextSample < self.totalSamples and not self.autoStopOuter and not self.kill:
            self.wasCalledBack = False
            if self.ps == ps3000a:
                self.status["getStreamingLatestValues"] = self.ps.ps3000aGetStreamingLatestValues(self.chandle, self.cFuncPtr, None) 
                if self.wasCalledBack:
                    data = np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB, self.bufferCompleteC, self.bufferCompleteD), axis=1)
                    self.signal.data.emit(data[max(0, self.nextSample - PicoScope.MAX_EMIT_LEN):self.nextSample])
                    if self.t[self.nextSample - 1] > self.save_t + SAVE_TIME:
                        np.save(
                            self.session_dir + "/tmp/ps3000a_" + str(self.save_t),
                            data[self.save_index:self.nextSample],
                            allow_pickle=False
                        )
                        self.save_t += SAVE_TIME
                        self.save_index = self.nextSample
                else:
                    time.sleep(0.01)
            else:
                self.status["getStreamingLatestValues"] = self.ps.ps4000GetStreamingLatestValues(self.chandle, self.cFuncPtr, None) 
                if self.wasCalledBack:
                    data = np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB), axis=1)
                    self.signal.data.emit(data[max(0, self.nextSample - PicoScope.MAX_EMIT_LEN):self.nextSample])
                    if self.t[self.nextSample - 1] > self.save_t + SAVE_TIME:
                        np.save(
                            self.session_dir + "/tmp/ps4000_" + str(self.save_t),
                            data[self.save_index:self.nextSample],
                            allow_pickle=False
                        )
                        self.save_t += SAVE_TIME
                        self.save_index = self.nextSample
                else:
                    time.sleep(0.01)

        if self.ps == ps3000a:
            self.status["stop"] = self.ps.ps3000aStop(self.chandle)
            assert_pico_ok(self.status["stop"])
            self.status["close"] = self.ps.ps3000aCloseUnit(self.chandle)
            assert_pico_ok(self.status["close"])
            print(self.status)
            np.save(self.session_dir + "/" + self.ps.name, np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB, self.bufferCompleteC, self.bufferCompleteD), axis=1), allow_pickle=False)
        else:
            self.status["stop"] = self.ps.ps4000Stop(self.chandle)
            assert_pico_ok(self.status["stop"])
            self.status["close"] = self.ps.ps4000CloseUnit(self.chandle)
            assert_pico_ok(self.status["close"])
            print(self.status)
            np.save(self.session_dir + "/" + self.ps.name, np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB), axis=1), allow_pickle=False)


class Signal(QObject):
    data = pyqtSignal(object)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.pvs = None
        self.microphone = None
        self.ps4000 = None
        self.ps3000a = None

        self.graphWidget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphWidget)

        self.plots = {
            "PPG0": self.graphWidget.addPlot(row=0, col=0),
            "Microphone": self.graphWidget.addPlot(row=1, col=0),
            "1A": self.graphWidget.addPlot(row=2, col=0),
            "1B": self.graphWidget.addPlot(row=3, col=0),
            "2A": self.graphWidget.addPlot(row=4, col=0),
            "2B": self.graphWidget.addPlot(row=5, col=0),
            "2C": self.graphWidget.addPlot(row=6, col=0),
            "2D": self.graphWidget.addPlot(row=7, col=0)
        }

        self.curves = {}
        for name, plot in self.plots.items():
            plot.setLabel("left", name)
            plot.setMouseEnabled(x=False, y=False)

            if name != "Microphone":
                plot.setXLink(self.plots["Microphone"])

            if name[0] == "1":
                self.curves[name] = plot.plot([], [], pen=(225, 109, 103))
            elif name[0] == "2":
                self.curves[name] = plot.plot([], [], pen=(62, 167, 160))
            else:
                self.curves[name] = plot.plot([], [], pen=(63, 169, 217))
            
            if name != "PPG0":
                self.curves[name].setDownsampling(ds=50, method='peak')
        
    async def start(self):
        i = 1
        while True:
            session_dir = f"session-{i:02d}"
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
                os.makedirs(session_dir + "/tmp")
                break
            i += 1

        microphone_signal = Signal()
        microphone_signal.data.connect(self.update_microphone_graph)
        self.microphone = Microphone(microphone_signal, session_dir)
        global_start_time = self.microphone.start_time

        device = await BleakScanner.find_device_by_name("Polar Sense DE957E2E", timeout=3)
        if device is None:
            print("Polar Sense DE957E2E not found")
        else:
            pvs_signal = Signal()
            pvs_signal.data.connect(self.update_pvs_graph)
            self.pvs = PolarVeritySense(device, pvs_signal, global_start_time, session_dir)
            await self.pvs.connect()
            print("Battery:", await self.pvs.get_battery_level())
            await self.pvs.start_ppg_stream()
        
        ps4000_signal = Signal()
        ps4000_signal.data.connect(self.update_ps4000_graph)
        self.ps4000 = PicoScope(ps4000, ps4000_signal, global_start_time, session_dir)

        ps3000a_signal = Signal()
        ps3000a_signal.data.connect(self.update_ps3000a_graph)
        self.ps3000a = PicoScope(ps3000a, ps3000a_signal, global_start_time, session_dir)

        t1 = threading.Thread(target=self.run_ps4000)
        t1.start()
        
        t2 = threading.Thread(target=self.run_ps3000a)
        t2.start()
    
    def run_ps4000(self):
        try:
            self.ps4000.open()
            self.ps4000.run()
        except Exception as e:
            print("ps4000 failed: ", e)
    
    def run_ps3000a(self):
        try:
            self.ps3000a.open()
            self.ps3000a.run()
        except Exception as e:
            print("ps3000a failed: ", e)
    
    @asyncClose
    async def closeEvent(self, event):
        if self.pvs is not None:
            await self.pvs.stop_ppg_stream()
            await self.pvs.disconnect()
        if self.microphone is not None:
            self.microphone.close()
        if self.ps4000 is not None:
            self.ps4000.kill = True
        if self.ps3000a is not None:
            self.ps3000a.kill = True
    
    def update_pvs_graph(self, data):
        self.curves["PPG0"].setData(data[:, 0], data[:, 1])

    def update_microphone_graph(self, data):
        self.curves["Microphone"].setData(data[:, 0], data[:, 1])
    
    def update_ps4000_graph(self, data):
        self.curves["1A"].setData(data[:, 0], data[:, 1])
        self.curves["1B"].setData(data[:, 0], data[:, 2])

    def update_ps3000a_graph(self, data):
        self.curves["2A"].setData(data[:, 0], data[:, 1])
        self.curves["2B"].setData(data[:, 0], data[:, 2])
        self.curves["2C"].setData(data[:, 0], data[:, 3])
        self.curves["2D"].setData(data[:, 0], data[:, 4])


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
