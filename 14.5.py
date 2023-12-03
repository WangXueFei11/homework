import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sample_rate, audio_data = wavfile.read("Dive.wav")

fft_result = np.fft.fft(audio_data)
fft_freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

magnitude_spectrum = np.abs(fft_result)

plt.rcParams['font.sans-serif'] = 'SimHei' 
plt.rcParams['axes.unicode_minus'] = False  
plt.figure(figsize=(10, 6))
plt.title("频谱图")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅度")
plt.xlim(0, sample_rate / 2)  # 只显示正频率部分
plt.plot(fft_freqs[:len(fft_freqs) // 2], magnitude_spectrum[:len(fft_freqs) // 2])
plt.show()
