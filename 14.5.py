# -*- coding: utf-8 -*-
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# 读取音频文件
sample_rate, audio_data = wavfile.read('my_audio_file.wav')

# 选择要分析的音频片段长度
segment_length = 4096
num_segments = int(len(audio_data) / segment_length)

# 对每个音频片段进行FFT
fft_results = []
for i in range(num_segments):
    start = i * segment_length
    end = start + segment_length
    segment = audio_data[start:end]
    fft_result = np.fft.fft(segment)
    fft_results.append(np.abs(fft_result))  # 取绝对值得到频谱幅度
    
# 计算频率轴
frequencies = np.linspace(0, sample_rate / 2, num=segment_length)# 只考虑正频率范围

# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.mean(fft_results, axis=0))# 平均所有片段的频谱幅度
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrogram of Audio File')
plt.show()
