import numpy as np
import matplotlib.pyplot as plt
import wave

SESSION_DIR = "session-03"

ps4000 = np.load(SESSION_DIR + "/ps4000.npy")
t1 = ps4000[:, 0]
fiber_1A = ps4000[:, 1]
fiber_1B = ps4000[:, 2]

ps3000a = np.load(SESSION_DIR + "/ps3000a.npy")
t2 = ps3000a[:, 0]
fiber_2A = ps3000a[:, 1]
fiber_2B = ps3000a[:, 2]
fiber_2C = ps3000a[:, 3]
fiber_2D = ps3000a[:, 4]

pvs = np.load(SESSION_DIR + "/pvs.npy")
t = pvs[:, 0]
ppg0 = pvs[:, 1]
ppg1 = pvs[:, 2]
ppg2 = pvs[:, 3]
ambient = pvs[:, 4]

wf = wave.open(SESSION_DIR + "/microphone.wav", "rb")
sample_rate = wf.getframerate()
num_frames = wf.getnframes()
frames = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
wf.close()

ax1 = plt.subplot(811)
plt.plot(t, ppg0)
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(812, sharex=ax1)
plt.plot(np.arange(num_frames) / sample_rate, frames)
plt.tick_params('x', labelbottom=False)

ax3 = plt.subplot(813, sharex=ax1)
plt.plot(t1, fiber_1A)
plt.tick_params('x', labelbottom=False)

ax4 = plt.subplot(814, sharex=ax1)
plt.plot(t1, fiber_1B)
plt.tick_params('x', labelbottom=False)

ax5 = plt.subplot(815, sharex=ax1)
plt.plot(t2, fiber_2A)
plt.tick_params('x', labelbottom=False)

ax6 = plt.subplot(816, sharex=ax1)
plt.plot(t2, fiber_2B)
plt.tick_params('x', labelbottom=False)

ax7 = plt.subplot(817, sharex=ax1)
plt.plot(t2, fiber_2C)
plt.tick_params('x', labelbottom=False)

ax8 = plt.subplot(818, sharex=ax1)
plt.plot(t2, fiber_2D)

plt.show()
