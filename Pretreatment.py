# -*- coding: utf-8 -*-
# Pretreatment.py
"""
Created on Sun Oct 15 22:32:53 2023

@author: ljjjun
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd

from scipy.ndimage import gaussian_filter1d

import os

import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy import interpolate
import re
import csv
from datetime import datetime
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
#---------------数据读取---------------------#

def read_Raman(file_path):

    # 使用内置的open函数打开文件
    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()
    #-------------------------------------#
    #--------数据格式转换-------------#


    # 找到数据开始和结束的标记
    start_marker = "#shining_data"
    end_marker = "#shining_end"
    # 截取标记之间的文本
    data_text = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    # 将文本按制表符分割，并转换为浮点数数组
    data_text2 = data_text.split('\n')[1:-1]

    data_list = []
    for x in data_text2:
        aa = [float(x.split('\t')[0]),float(x.split('\t')[1])]
        data_list.append(aa)

    data_array = np.array(data_list)# data_array[0]是波数 data_array[0]是intensity
    return data_array

def read_label(file_path):
    # 用于读取lable
    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()


    end_marker = "\n#shining_data"
    #print(text)
    # 截取标记之间的文本
    try:
        try:
            start_marker = "#Lable:"
            data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
        except:
            start_marker = "#label "
            data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    except:
        start_marker = "#label:"
        data_text_concentration = text[text.index(start_marker) + len(start_marker): text.index(end_marker)]


    lable = float(data_text_concentration)
    return lable

def read_conc(file_path):

    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()

    #start_marker = "Concentration:"
    #end_marker = "}"
    # 截取标记之间的文本
    #data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    #lable = float(data_text_concentration)
    pattern = r'(?<=Concentration:)(.*?)(?=})'
    match1 = re.search(pattern, text)
    result = float(match1[0])
    return result


#---------------y变换方法--------------#
def conclog10_Normalization(y,miny,maxy):
    y=conclog10y(y)
    y=concNormalization(y,miny,maxy)
    return y

def Inverse_conclog10_Normalization(y,miny,maxy):
    y=Inverse_concNormalization(y,miny,maxy)
    y=Inverse_conclog10y(y)
    return y
def conclog10y(y):
    #输入应该大于0
    y=np.log10(y)
    return y
def Inverse_conclog10y(y):
    y = 10**y;
    return y
def concNormalization(y,miny,maxy):
    y1=(y-miny)/(maxy-miny)

    return y1

def Inverse_concNormalization(y,miny,maxy):
    y1 = y*(maxy-miny)+miny
    return y1

#-----------光谱二维化方法----------------------------#
def Recurrence_plot(wn_range,data_array,normalization=True):
    #光谱递推图方法
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)
    Raman2Dresult = np.abs(y_normal[:, None] - y_normal[None, :])
    return Raman2Dresult

def Gramian_angular(wn_range,data_array,transformation='s',normalization=True):
    # 格拉米角场
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    #transformation: 转换方式，s:夹角和的余弦(GASF);d:夹角差的正弦(GADF),默认s方式,放大拉曼峰影响

    """
    GASF和GADF是两种将时间序列转化为二维图像的方法，它们都是基于Gramian angular field的概念。它们的区别在于：
    GASF是Gramian angular summation field的缩写，它是通过计算两个时间序列之间的夹角余弦之和来得到的。
    它可以保留时间序列的周期性和趋势性信息，但也可能放大噪声和异常值的影响。

    GADF是Gramian angular difference field的缩写，它是通过计算两个时间序列之间的夹角差值之正弦来得到的。
    它可以突出时间序列的差异性和变化性信息，但也可能损失周期性和趋势性信息。
    """

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    # 转换为极坐标
    theta = np.arccos(y_normal) # 角度
    #r = t # 半径
    if transformation == 's':

        # 计算Gramian angular summation field (GASF)[^3^][3]
        Raman2Dresult = np.cos(theta[:, None] + theta[None, :])
    elif transformation == 'd':

        # 计算Gramian angular difference field (GADF)[^2^][2]
        Raman2Dresult = np.sin(theta[:, None] - theta[None, :])
    return Raman2Dresult

def Short_time_Fourier_transform(wn_range,data_array,fs =1000,window='hann',nperseg=50,normalization=True):
    # 短时傅里叶变换

    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    # fs采样频率,window：窗函数选择 nperseg :每个段的窗口长度

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    frequencies, times, Zxx = stft(y_normal, fs = fs,window = window, nperseg=nperseg,)
    # 创建插值函数
    interpolator = interpolate.interp2d(times, frequencies, np.abs(Zxx), kind='cubic')
    new_f = np.linspace(frequencies.min(), frequencies.max(), y_normal.shape[0])
    new_t = np.linspace(times.min(), times.max(), y_normal.shape[0])
    # 使用插值函数计算新的STFT矩阵
    Raman2Dresult = interpolator(new_t, new_f)
    return Raman2Dresult


def Markov_transition_field(wn_range,data_array,Q,normalization=True):
    # 马尔科夫跃迁场
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    # 划分状态空间
    #Q =len(y_noraml) # 状态空间个数
    #Q =500
    bins = np.linspace(0, 1+0.1, Q + 1) # 状态空间边界  +0.1避免y0归一化后的1值被划分到状态空间外。
    #labels = np.arange(1, Q + 1) # 状态空间标签
    q = np.digitize(y_normal, bins) - 1 # 将时间序列值分配到状态空间

    # 构建转移矩阵
    V = np.zeros((Q, Q)) # 初始化转移矩阵
    for i in range(len(q) - 1):
        V[q[i], q[i + 1]] += 1 # 统计相邻两个状态出现的次数

    # 计算概率
    Raman2Dresult =V /np.sum(V)

    # 计算概率
    #Raman2Dresult =V /np.sum(V)
    return Raman2Dresult


def Heat_map(wn_range,data_array,num_intervals=100,normalization=False):
    #直接将曲线映射到二维图像。
    num_intervals = 100
    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)
    data_array2 = np.column_stack((x, y_normal))

    x_range = [np.min(data_array2[:,0]),np.max(data_array2[:,0])]
    y_range = [np.min(data_array2[:,1]),np.max(data_array2[:,1])]
    step_x = (x_range[1]-x_range[0])/num_intervals
    step_y = (y_range[1]-y_range[0])/num_intervals
    # 创建一个二维数组，大小等于区间数
    heatmap = np.zeros((num_intervals, num_intervals))

    # 对于序列中的每一个点
    for x, y in data_array2:
        # 找到其在二维图像中对应的格子
        j = int((x-x_range[0]-1e-6) / step_x)
        i = int((y-y_range[0]-1e-6 )/ step_y)

        # 将该格子的值设为1
        heatmap[i, j] = heatmap[i, j]+1
    for i in range(heatmap.shape[1]):
        index = np.argmax(heatmap[:,i])
        heatmap[0:index,i]=np.max(heatmap[:,i])
    return heatmap

def Data_Interception(wn_range,data_array,normalization=True):

    #截取数据并归一化
    x0 = data_array[:,0]
    y0 = data_array[:,1]
    index_low = np.absolute(x0-wn_range[0]).argmin()
    index_high = np.absolute(x0-wn_range[1]).argmin()
    #x = x0[index_low:index_high]
    y = y0[index_low:index_high]
    x = x0[index_low:index_high]
    # 归一化y序列



    if normalization:
        y_normal = (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        y_normal = y
    return x,y_normal



def baseline_als(y, lam=10**4,p=0.001, niter=3):
    """
    偏最小二乘法生成基线
    """
    #L = len(y)
    L= np.size(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    baseline = 0
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    r_baseline = y-baseline
    return baseline,r_baseline
import math

class WitecRamanProcessor:
    def __init__(self, input_dir, output_dir, target_wavelength=1339.6, lam=1e5, p=0.01, niter=3,start_wavelength=500,end_wavelength=2500,sigma=3,
                 compat_mode=False, log_path=None, header_mode='legacy', sniff_delimiters=None):
        """
        本类用于处理Witec的共聚焦拉曼显微镜所采集的拉曼mapping数据
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_wavelength = target_wavelength
        self.lam = lam#去基线时的平滑参数，越大越跟随低频，越小越跟随高频/
        self.p = p#权重更新参数。接近1，基线更可能低于大部分数据点，适用于数据中有较多的高值离群点。接近 0：基线更可能高于大部分数据点，适用于数据中有较多的低值离群点。
        self.niter = niter#基线校正循环次数
        self.start_wavelength = start_wavelength
        self.end_wavelength = end_wavelength
        self.sigma = sigma  #去噪窗口宽度
        self.reback = True
        self.Denoise = True

        # --- Optional cosmic-ray spike removal controls ---
        self.RemoveCosmicRay = False
        self.cosmic_window = 7
        self.cosmic_z_threshold = 8.0
        self.cosmic_max_width = 2
        self.cosmic_repair_half_window = 2

        # --- Compatibility / logging controls (optional; default keeps legacy behavior) ---
        self.compat_mode = compat_mode
        self.log_path = log_path
        self.header_mode = header_mode
        # sniff_delimiters: tuple of delimiters to try when compat_mode=True and delimiter is None
        self.sniff_delimiters = tuple(sniff_delimiters) if sniff_delimiters is not None else ('\t', ',', ';', 'whitespace')
        # Last read diagnostics (set by read_data in compat_mode)
        self._last_read_info = {}

    def find_closest_point_index(self, spectral_data, x1, y1):
        """
        从spectral_data数据中找出指定(x1,y1)最近的索引
        """
        # 初始化最小距离为正无穷大
        l = self.extract_positions(spectral_data)
        min_distance = float('inf')
        closest_index = -1

        # 遍历列表中的每个点
        for i, (x, y) in enumerate(l):
            # 计算当前点与 (x1, y1) 之间的距离
            distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

            # 如果当前距离小于最小距离，则更新最小距离和最近点的索引
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        data_column = spectral_data.iloc[:, closest_index]
        return data_column
    def parse_filename(self, file_path):
        """
        解析文件名，获取相关信息。
        """
        file_name = os.path.basename(file_path)
        self.file_name0 = file_name
        file_info = re.match(r'^(.*?),(.*?),(-?\d+),(\d+(\.\d+)?),(\d+),(\d+)(?:_.*)?\.txt$', file_name)
        if file_info:
            process, analyte, concentration, integration_time, _, x_points, y_points = file_info.groups()
            concentration = f"10^{int(concentration)} M"
            x_points, y_points = int(x_points), int(y_points)
            return process, analyte, concentration, integration_time, x_points, y_points
        return None

    def _read_first_nonempty_line(self, file_path, max_lines=30):
        """Read the first non-empty, non-null line from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s != '':
                        return s
        except Exception:
            return None
        return None

    def _sniff_delimiter(self, first_line):
        """Heuristically choose a delimiter from self.sniff_delimiters based on the first line."""
        if not first_line:
            return None
        best = None
        best_cols = 0
        for d in self.sniff_delimiters:
            if d == 'whitespace':
                cols = len(re.split(r'\s+', first_line.strip()))
            else:
                cols = len(first_line.split(d))
            if cols > best_cols:
                best_cols = cols
                best = d
        # Need at least 2 columns (x + at least one spectrum)
        if best_cols < 2:
            return None
        return best

    def _detect_header(self, first_line, delimiter_token):
        """Return True if the first line looks like a header row, False if it looks like numeric data."""
        if not first_line:
            return True
        if delimiter_token == 'whitespace':
            tokens = re.split(r'\s+', first_line.strip())
        else:
            tokens = first_line.split(delimiter_token)
        tokens = [t.strip() for t in tokens if t.strip() != '']
        if len(tokens) < 2:
            return True

        def is_float(x):
            try:
                float(x)
                return True
            except Exception:
                return False

        floatable = sum(1 for t in tokens if is_float(t))
        ratio = floatable / max(1, len(tokens))
        # If most tokens are numeric, treat as *no header*
        return ratio < 0.8

    def _ensure_log_header(self):
        """Create log file with header if needed."""
        if not self.log_path:
            return
        try:
            if not os.path.exists(self.log_path):
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None
                with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['src_filename', 'dst_filename', 'issues', 'timestamp'])
        except Exception:
            # Logging must never break processing
            pass

    def _append_log(self, src_filename, dst_filename, issues):
        """Append one row to the log. issues can be list[str] or str."""
        if not self.log_path:
            return
        try:
            self._ensure_log_header()
            if isinstance(issues, (list, tuple)):
                issues_str = ';'.join([str(x) for x in issues if x])
            else:
                issues_str = str(issues)
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([src_filename, dst_filename, issues_str, datetime.now().isoformat(timespec='seconds')])
        except Exception:
            pass

    def _try_read_manual_marked_data(self, file_path, delimiter='	'):
        """
        识别并读取“手工标记后的物理数据 / 自标记后的物理数据 / 循环自标记后的物理数据”。
        文件结构：
        第1行：富光谱级标签（仅第一列有效）
        第2行：逐光谱标签
        第3行：逐光谱置信度
        第4行：列名（第一列为波数字段名，后续列为xy字符串）
        第5行起：波数 + 强度矩阵
        """
        try:
            raw = pd.read_csv(file_path, delimiter=delimiter or '	', header=None, dtype=str, keep_default_na=False)
        except Exception:
            return None

        if raw.shape[0] < 5 or raw.shape[1] < 2:
            return None

        row0_tail = [str(x).strip() for x in raw.iloc[0, 1:].tolist()]
        conf_tail = pd.to_numeric(raw.iloc[2, 1:], errors='coerce')
        wavelength_tail = pd.to_numeric(raw.iloc[4:, 0], errors='coerce')

        row0_blank_ratio = sum(x == '' for x in row0_tail) / max(1, len(row0_tail))
        conf_numeric_ratio = conf_tail.notna().sum() / max(1, len(conf_tail))
        wavelength_numeric_ratio = wavelength_tail.notna().sum() / max(1, len(wavelength_tail))

        if row0_blank_ratio < 0.7 or conf_numeric_ratio < 0.7 or wavelength_numeric_ratio < 0.9:
            return None

        header_row = raw.iloc[3, :].tolist()
        column_names = []
        for i, value in enumerate(header_row[1:], start=1):
            value = str(value).strip()
            column_names.append(value if value else f'Spec{i}')

        wavelengths = pd.to_numeric(raw.iloc[4:, 0], errors='coerce').reset_index(drop=True)
        spectral_data = raw.iloc[4:, 1:].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
        spectral_data.columns = column_names

        self._last_read_info = {
            'delimiter_used': delimiter or '	',
            'has_header': True,
            'ncols': int(spectral_data.shape[1] + 1),
            'manual_marked_format': True,
        }
        self._last_manual_label_bundle = {
            'file_label': str(raw.iloc[0, 0]).strip() or 'U',
            'spectrum_labels': [str(x).strip() or 'U' for x in raw.iloc[1, 1:].tolist()],
            'confidences': [float(x) if str(x).strip() not in ('', 'nan', 'None') else 1.0 for x in raw.iloc[2, 1:].tolist()],
            'position_headers': column_names,
            'wavelength_header': str(raw.iloc[3, 0]).strip() or 'wave',
        }
        return wavelengths, spectral_data

    def read_data(self, file_path, delimiter='	'):
        """
        读取高光谱数据。

        兼容两类txt：
        1. 普通预处理后的物理数据：第一列波数，后续列为各条光谱。
        2. 手工/自/循环标记后的物理数据：前4行为标签与元数据，第5行起为波数和强度矩阵。

        Backward compatible behavior:
        - compat_mode=False (default): 默认保持原有读取习惯，但现在会优先自动识别_MM等标记格式。
        - compat_mode=True: 在delimiter=None时自动探测分隔符，并可在header_mode='auto'时自动判断表头。
        """
        manual = None
        if delimiter is None:
            try_delims = ['	', ',', ';']
        else:
            try_delims = [delimiter]

        for d in try_delims:
            manual = self._try_read_manual_marked_data(file_path, delimiter=d)
            if manual is not None:
                return manual

        if not getattr(self, 'compat_mode', False):
            used_delim = delimiter if delimiter is not None else '	'
            data = pd.read_csv(file_path, delimiter=used_delim)
            self._last_read_info = {
                'delimiter_used': used_delim,
                'has_header': True,
                'ncols': int(data.shape[1]),
                'manual_marked_format': False,
            }
            self._last_manual_label_bundle = None
            wavelengths = data.iloc[:, 0]
            spectral_data = data.iloc[:, 1:]
            return wavelengths, spectral_data

        lines = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(30):
                    line = f.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s:
                        lines.append(s)
        except Exception:
            lines = []

        if not lines:
            raise ValueError('read_failed')

        def _floatable_ratio(tokens):
            ok = 0
            for t in tokens:
                try:
                    float(t)
                    ok += 1
                except Exception:
                    pass
            return ok / max(1, len(tokens))

        candidate_lines = lines[1:] if len(lines) >= 2 else lines[:]
        data_line = None
        for s in candidate_lines:
            for split_kind in ('	', ',', ';', 'whitespace'):
                toks = (re.split(r'\s+', s) if split_kind == 'whitespace' else s.split(split_kind))
                toks = [x.strip() for x in toks if x.strip() != '']
                if len(toks) >= 2 and _floatable_ratio(toks) >= 0.8:
                    data_line = s
                    break
            if data_line is not None:
                break

        if data_line is None:
            data_line = candidate_lines[0] if candidate_lines else lines[0]

        used_delim = delimiter
        if used_delim is None:
            tab_cols = len(data_line.split('	'))
            if tab_cols >= 2:
                used_delim = '	'
            else:
                used_delim = self._sniff_delimiter(data_line)

        if used_delim is None:
            raise ValueError('delimiter_unknown')

        first_line = lines[0]
        has_header = True
        if getattr(self, 'header_mode', 'legacy') == 'auto':
            has_header = self._detect_header(first_line, used_delim)

        read_kwargs = {}
        if used_delim == 'whitespace':
            read_kwargs['sep'] = r'\s+'
            read_kwargs['engine'] = 'python'
        else:
            read_kwargs['delimiter'] = used_delim

        read_kwargs['header'] = 0 if has_header else None
        data = pd.read_csv(file_path, **read_kwargs)

        if not has_header:
            ncols = data.shape[1]
            if ncols < 2:
                raise ValueError('data_format_invalid')
            cols = ['X-Axis'] + [f'Spec{i}' for i in range(1, ncols)]
            data.columns = cols

        if data.shape[1] < 2:
            raise ValueError('data_format_invalid')

        self._last_read_info = {
            'delimiter_used': used_delim,
            'has_header': bool(has_header),
            'ncols': int(data.shape[1]),
            'manual_marked_format': False,
        }
        self._last_manual_label_bundle = None

        wavelengths = data.iloc[:, 0]
        spectral_data = data.iloc[:, 1:]
        return wavelengths, spectral_data

    def baseline_als(self, y):
        """
        偏最小二乘法生成基线。
        """
        L = np.size(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        baseline = 0
        for i in range(self.niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            baseline = spsolve(Z, w * y)
            w = self.p * (y > baseline) + (1 - self.p) * (y < baseline)
        r_baseline = y - baseline
        return baseline, r_baseline
    def denoise_spectral_data(self, spectral_data, sigma=1):
        """
        对光谱数据进行去噪。
        """
        denoised_data = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        for column in spectral_data.columns:
            denoised_data[column] = gaussian_filter1d(spectral_data[column].to_numpy(), sigma=self.sigma)
        return denoised_data

    def _rolling_median_reference(self, y, window):
        """
        使用局部滚动中位数生成鲁棒参考曲线，仅用于尖刺检测。
        """
        window = int(max(3, window))
        if window % 2 == 0:
            window += 1
        return pd.Series(y).rolling(window=window, center=True, min_periods=1).median().to_numpy()

    def _mad_sigma(self, residual):
        """
        基于MAD估计鲁棒噪声尺度。
        """
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.std(residual))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        return sigma

    def _find_narrow_positive_spikes(self, y):
        """
        基于局部鲁棒参考曲线 + 残差标准化打分 + 宽度约束，检测宇宙射线尖刺。
        """
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return np.zeros(0, dtype=bool)

        reference = self._rolling_median_reference(y, self.cosmic_window)
        residual = y - reference
        sigma = self._mad_sigma(residual)
        candidate_mask = residual > (self.cosmic_z_threshold * sigma)

        spike_mask = np.zeros(n, dtype=bool)
        idx = 0
        max_width = max(1, int(self.cosmic_max_width))
        while idx < n:
            if not candidate_mask[idx]:
                idx += 1
                continue
            start = idx
            while idx + 1 < n and candidate_mask[idx + 1]:
                idx += 1
            end = idx
            width = end - start + 1
            if width <= max_width:
                spike_mask[start:end + 1] = True
            idx += 1
        return spike_mask

    def _repair_spike_points(self, y, spike_mask):
        """
        以修复位置为中心，取局部有效点均值修复异常值。
        若邻近点也为异常值，则不会参与均值计算。
        """
        y = np.asarray(y, dtype=float)
        repaired = y.copy()
        n = y.size
        half_window = max(1, int(self.cosmic_repair_half_window))

        spike_indices = np.flatnonzero(spike_mask)
        if spike_indices.size == 0:
            return repaired

        for idx in spike_indices:
            left = max(0, idx - half_window)
            right = min(n, idx + half_window + 1)
            valid = [j for j in range(left, right) if (not spike_mask[j]) and j != idx]

            expand = half_window + 1
            while len(valid) == 0 and (left > 0 or right < n):
                left = max(0, idx - expand)
                right = min(n, idx + expand + 1)
                valid = [j for j in range(left, right) if (not spike_mask[j]) and j != idx]
                expand += 1

            if len(valid) == 0:
                repaired[idx] = y[idx]
            else:
                repaired[idx] = float(np.mean(y[valid]))

        return repaired

    def remove_cosmic_ray_spikes(self, spectral_data):
        """
        对基线校正后的光谱执行可选的宇宙射线尖刺移除。
        流程：局部鲁棒参考曲线 -> 残差标准化打分 -> 宽度约束 -> 局部均值修复。
        """
        despiked_data = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        for column in spectral_data.columns:
            y = spectral_data[column].to_numpy(dtype=float)
            spike_mask = self._find_narrow_positive_spikes(y)
            despiked_data[column] = self._repair_spike_points(y, spike_mask)
        return despiked_data

    def apply_baseline_correction(self, spectral_data):
        """
        对所有光谱数据进行优化。
        基线校正。self.reback = True 基线校正
        和去噪 self.Denoise = True  去噪开。
        """
        reback = self.reback
        denoise = self.Denoise
        baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        r_baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        if reback:
            for column in spectral_data.columns:
                y = spectral_data[column].to_numpy()
                baseline, r_baseline = self.baseline_als(y)
                baselines[column] = baseline
                r_baselines[column] = r_baseline
        else:
            r_baselines = spectral_data

        if self.RemoveCosmicRay:
            r_baselines = self.remove_cosmic_ray_spikes(r_baselines)

        if denoise:
            denoise_data = self.denoise_spectral_data(r_baselines)
        else:
            denoise_data = r_baselines
        return denoise_data
    def apply_baseline_correction_SpectralLabelingApp(self, spectral_data):
        """
        SpectralLabelingApp类配合专用。
        对所有光谱数据进行优化。
        基线校正。self.reback = True 基线校正
        和去噪 self.Denoise = True  去噪开。
        """
        reback = self.reback
        denoise = self.Denoise
        baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        r_baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        if reback:
            for column in tqdm(spectral_data.columns,desc='Data preprocessing '):
                y = spectral_data[column].to_numpy()
                baseline, r_baseline = self.baseline_als(y)
                baselines[column] = baseline
                r_baselines[column] = r_baseline
        else:
            r_baselines = spectral_data
        if denoise:
            denoise_data = self.denoise_spectral_data(r_baselines)
        else:
            denoise_data = r_baselines
        return denoise_data
    def extract_positions(self, spectral_data):
        """
        从列标签中提取位置信息。
        """
        positions = []
        for col in spectral_data.columns:
            label = col.split('(')[-1].split(')')[0]
            x, y = map(int, label.split('/'))
            positions.append((x, y))
        return positions

    def create_position_to_col_map(self, spectral_data):
        """
        创建位置到列的映射。
        """
        positions = self.extract_positions(spectral_data)
        return dict(zip(positions, spectral_data.columns))

    def find_closest_wavelength_index(self, wavelengths):
        """
        查找最接近目标波长的索引。
        """
        return (np.abs(wavelengths - self.target_wavelength)).idxmin()

    def create_hotmap(self, r_baselines, positions, closest_wavelength_idx):
        """
        创建热图数据。
        """
        x_max = max([pos[0] for pos in positions])
        y_max = max([pos[1] for pos in positions])
        hotmap = np.zeros((x_max + 1, y_max + 1))

        for pos, col in self.create_position_to_col_map(r_baselines).items():
            x, y = pos
            hotmap[x, y] = r_baselines[col].iloc[closest_wavelength_idx]

        return hotmap

    def plot_and_save_hotmap(self, hotmap, wavelengths, closest_wavelength_idx, analyte, concentration, output_path):
        """
        绘制并保存热图。
        """
        plt.imshow(hotmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title(
            f'Hotmap for Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(output_path, 'hotmap.png'))
        plt.close()

    def plot_stacked_spectra(self, denoised_r_baselines,wavelengths, top_10_cols, closest_wavelength_idx,analyte, concentration, output_path):
        """
        绘制并保存堆积曲线图。 还有问题
        """
        # 获取每个光谱的强度数据
        spectra = [denoised_r_baselines[col] for col in top_10_cols]

        # 转置数据，使得每行对应一个波长，每列对应一个光谱
        #spectra = np.array(spectra).T

        # 绘制堆积曲线图
        plt.figure()
        plt.stackplot(wavelengths, spectra, labels=top_10_cols)
        plt.title(
            f'Top 10 Spectra at Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('Wavelength (cm^-1)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_path, 'stacked_spectra.png'))
        plt.close()
    def plot_and_save_top_spectra(self, r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
                                  concentration, output_path):
        """
        绘制并保存最强的十个光谱图。
        """
        # 指定绘制的波长范围，例如 500 到 600 nm
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength

        # 找到波长范围内的索引
        start_idx = np.argmax(wavelengths >= start_wavelength)
        end_idx = np.argmax(wavelengths > end_wavelength)

        plt.figure()
        for col in top_10_cols:
            label = col.split('(')[-1].split(')')[0]
            x, y = map(int, label.split('/'))
            # 选择指定范围内的波长和对应的光谱强度
            selected_wavelengths = wavelengths[start_idx:end_idx]
            selected_intensity = r_baselines[col].iloc[start_idx:end_idx]
            plt.plot(selected_wavelengths, selected_intensity, label=f'({x}, {y})')
        plt.title(
            f'Top 10 Spectra at Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('Wavelength (cm^-1)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'top_10_spectra.png'))
        plt.close()

    def process_file(self, file_path,delimiter='\t'):
        """
        处理单个文件，生成结果并保存。
        """
        # 从文件名中解析信息
        parsed_info = self.parse_filename(file_path)
        if parsed_info is None:
            return
        process, analyte, concentration, integration_time, x_points, y_points = parsed_info

        # 读取数据
        wavelengths, spectral_data = self.read_data(file_path,delimiter)

        # 应用基线校正和去噪
        r_baselines = self.apply_baseline_correction(spectral_data)

        # 查找特定波长或最近的波长

        closest_wavelength_idx = self.find_closest_wavelength_index(wavelengths)

        # 创建热图数据
        hotmap = self.create_hotmap(r_baselines, self.extract_positions(spectral_data), closest_wavelength_idx)

        # 找到在该波长处光谱强度最强的列
        top_10_cols = r_baselines.iloc[closest_wavelength_idx].nlargest(10).index

        # 创建输出目录
        output_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(file_path))[0])
        os.makedirs(output_path, exist_ok=True)

        # 绘制并保存热图
        self.plot_and_save_hotmap(hotmap, wavelengths, closest_wavelength_idx, analyte, concentration, output_path)

        # 绘制并保存最强的十个光谱图
        self.plot_and_save_top_spectra(r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
                                       concentration, output_path)
        #self.plot_stacked_spectra(r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
        #
        #concentration, output_path)
    def process_directory(self,delimiter='\t'):
        """
        批量处理文件夹中的所有文件。目的是绘制去基线后mapping，并选指定波长前十光谱绘图保存。
        """
        files = [file for file in os.listdir(self.input_dir) if file.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing directory", position=0):
            file_path = os.path.join(self.input_dir, file_name)
            self.process_file(file_path,delimiter=delimiter)

    def process_file_reBaseLine(self, file_path, delimiter='\t'):
        """
        处理单个文件，生成结果并保存。

        Backward compatible behavior:
        - compat_mode=False (default): legacy strict behavior (filename must match parse_filename, delimiter defaults to '\t').
        - compat_mode=True: if filename does not match, still process with a fallback output name; auto header/delimiter when enabled;
          record non-standard or failed files into log (when log_path is provided).
        """
        issues = []

        # From filename: legacy validation (keep parse logic unchanged)
        parsed_info = self.parse_filename(file_path)
        if parsed_info is None:
            issues.append('filename_not_standard')
            if not getattr(self, 'compat_mode', False):
                return  # legacy behavior
        else:
            process, analyte, concentration, integration_time, x_points, y_points = parsed_info

        # Determine output filename (fallback if parse failed)
        base = os.path.splitext(os.path.basename(file_path))[0]
        
        out_name = f"{base}_reBaseLine.txt"


        output_path = os.path.join(self.output_dir, out_name)

        # Read data
        try:
            if getattr(self, 'compat_mode', False):
                # allow auto delimiter sniffing when caller didn't force delimiter
                delim_to_use = delimiter
                if delim_to_use == '\t' and getattr(self, 'header_mode', 'legacy') == 'auto':
                    # In compat mode, user may still pass '\t' by default; we only auto-sniff when delimiter=None.
                    # To enable sniffing from the GUI, pass delimiter=None.
                    pass
                wavelengths, spectral_data = self.read_data(file_path, delim_to_use)
            else:
                wavelengths, spectral_data = self.read_data(file_path, delimiter)
        except Exception as e:
            # In compat mode, log and skip; in legacy mode, re-raise to preserve behavior as much as possible
            if getattr(self, 'compat_mode', False):
                err = str(e)
                if 'delimiter_unknown' in err:
                    issues.append('delimiter_unknown')
                elif 'data_format_invalid' in err:
                    issues.append('data_format_invalid')
                else:
                    issues.append('read_failed')
                self._append_log(os.path.basename(file_path), os.path.basename(output_path), issues)
                return
            raise

        # Header / XY checks (compat diagnostics only)
        if getattr(self, 'compat_mode', False):
            info = getattr(self, '_last_read_info', {}) or {}
            if info.get('has_header') is False:
                issues.append('header_missing')

            # If there is a header but no (x/y) pattern in any spectrum column name, record it.
            # This is relevant for mapping workflows; multi-spectra may legitimately not have xy.
            try:
                cols = list(spectral_data.columns)
                has_xy = any(re.search(r'\(\s*\d+\s*/\s*\d+\s*\)', str(c)) for c in cols)
                if info.get('has_header') is True and not has_xy:
                    issues.append('header_no_xy')
            except Exception:
                pass

        # Baseline correction
        r_baselines = self.apply_baseline_correction(spectral_data)

        # Save
        try:
            result = pd.concat([wavelengths, r_baselines], axis=1)
            os.makedirs(self.output_dir, exist_ok=True)
            result.to_csv(output_path, sep='\t', index=False)  # output_path ends with .txt, that's fine
        except Exception:
            if getattr(self, 'compat_mode', False):
                issues.append('save_failed')
                self._append_log(os.path.basename(file_path), os.path.basename(output_path), issues)
                return
            raise

        # Log non-standard but processed
        if getattr(self, 'compat_mode', False) and issues:
            self._append_log(os.path.basename(file_path), os.path.basename(output_path), issues)

    def process_directory_reBaseLine(self):
        """
        批量处理文件夹中的所有文件，目的是去基线并保存。
        """
        if getattr(self, 'compat_mode', False) and self.log_path:
            self._ensure_log_header()

        files = [file for file in os.listdir(self.input_dir) if file.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing directory", unit="file"):
            file_path = os.path.join(self.input_dir, file_name)
            # In compat mode, pass delimiter=None to enable sniffing; otherwise keep legacy default '\t'
            if getattr(self, 'compat_mode', False):
                self.process_file_reBaseLine(file_path, delimiter=None)
            else:
                self.process_file_reBaseLine(file_path)

    def save_manual_marked_data(self, source_file_path, wavelengths, spectral_data,
                                spectrum_labels=None, file_label='U', confidences=None,
                                output_dir=None, suffix='_MM'):
        """
        将一个输入txt保存为“手工标记后的物理数据”格式。
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def _clean_label(value, default='U'):
            text = str(value).strip()
            if text == '' or '\t' in text or '\n' in text or '\r' in text:
                return default
            return text

            return text

        n_spectra = spectral_data.shape[1]
        if spectrum_labels is None:
            spectrum_labels = ['U'] * n_spectra
        if confidences is None:
            confidences = [1.0] * n_spectra

        if len(spectrum_labels) != n_spectra:
            raise ValueError('spectrum_labels长度与光谱列数不一致')
        if len(confidences) != n_spectra:
            raise ValueError('confidences长度与光谱列数不一致')

        file_label = _clean_label(file_label, default='U')
        spectrum_labels = [_clean_label(x, default='U') for x in spectrum_labels]
        confidences = [float(x) for x in confidences]

        manual_bundle = getattr(self, '_last_manual_label_bundle', {}) or {}
        wavelength_header = manual_bundle.get('wavelength_header', None)
        if not wavelength_header:
            wavelength_header = getattr(wavelengths, 'name', None) or 'wave'

        column_names = [str(col) if str(col).strip() else f'Spec{i+1}' for i, col in enumerate(spectral_data.columns)]

        base = os.path.splitext(os.path.basename(source_file_path))[0]
        output_base = base if base.endswith(suffix) else f'{base}{suffix}'
        output_path = os.path.join(output_dir, f'{output_base}.txt')

        body = pd.concat([
            pd.Series(wavelengths).reset_index(drop=True),
            spectral_data.reset_index(drop=True)
        ], axis=1)
        body.columns = [wavelength_header] + column_names

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='	')
            writer.writerow([file_label] + [''] * n_spectra)
            writer.writerow([''] + spectrum_labels)
            writer.writerow([''] + [f'{x:.6g}' for x in confidences])
            writer.writerow([wavelength_header] + column_names)
            for row in body.itertuples(index=False, name=None):
                writer.writerow(list(row))

        return output_path

    def read_template(self,template_file):
        with open(template_file, 'r') as file:
            lines = file.readlines()
        end_index = lines.index('#shining_end')
        label_index = lines.index('#Lable:1\n')
        self.header1 = lines[:label_index]
        self.header2 = lines[label_index+1:end_index]
        self.footer = lines[end_index:]

    def save_spectrum(self, wavelengths, spectrum, label, output_dir, index):
        labeled_data_str = "\n".join(
            f"{wavelength}\t{intensity}" for wavelength, intensity in zip(wavelengths, spectrum))


        output_file = os.path.join(output_dir, f"s_{self.file_name0}_n{index + 1}_l{label}.txt")

        with open(output_file, 'w') as file:
            file.writelines(self.header1)
            file.write(f"#label:{label}\n")
            file.writelines(self.header2)
            file.write(f"{labeled_data_str}\n")
            file.writelines(self.footer)
    def update_plot(self):
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength
        wavelengths = self.wavelengths

        # 找到波长范围内的索引
        start_idx = np.argmax(wavelengths >= start_wavelength)
        end_idx = np.argmax(wavelengths > end_wavelength)

        self.ax.clear()
        current_spectrum = self.spectral_data.iloc[:, self.current_index]
        self.column_label = self.spectral_data.iloc[:, self.current_index].name
        selected_wavelengths = wavelengths[start_idx:end_idx]
        selected_intensity = current_spectrum.iloc[start_idx:end_idx]
        self.ax.plot(selected_wavelengths, selected_intensity)
        # 在每个位置添加竖线
        for mark in self.mark_list:
            plt.axvline(x=mark, color='r', linestyle='--')
            plt.text(mark, plt.ylim()[1] *0.8, mark, rotation=90, ha='right')
        self.ax.set_title(f'Spectrum{self.column_label}, {self.current_index + 1}/{self.spectral_data.shape[1]}')
        self.ax.set_xlabel('Raman shift (cm^-1)')
        self.ax.set_ylabel('Intensity')
        self.canvas.draw()


    def label_spectrum(self, label):

        current_spectrum = self.spectral_data.iloc[:, self.current_index]
        #processor = WitecRamanProcessor(self.template_file)
        self.read_template(self.template_file)
        if label != 2: #如果是2，不保存。
            self.save_spectrum(self.wavelengths, current_spectrum, label, self.output_dir, self.current_index)

        self.labels.append(label)
        self.current_index += 1
        if self.current_index < self.spectral_data.shape[1]:
            self.update_plot()
        else:
            messagebox.showinfo("Info", "All spectra have been labeled.")
            self.master.quit()





class SpectralLabelingApp:
    """
    手工打标界面：
    - 一个输入txt最终输出一个对应的_MM.txt
    - 三个可编辑标签输入框 + 对应打标按钮
    - 上一条 / 下一条（保持U） / 完成打标
    - 支持辅助线，用于参考目标峰位
    - 未标记光谱默认保持为U
    """
    def __init__(self, master, processor, input_file, output_dir, wavelengths, spectral_data,
                 start_wavelength=500, end_wavelength=1800,
                 button_label_defaults=None, initial_labels=None, initial_file_label='U',
                 mark_list=None):
        self.master = master
        self.processor = processor
        self.input_file = input_file
        self.output_dir = output_dir
        self.wavelengths = pd.Series(wavelengths).reset_index(drop=True)
        self.spectral_data = spectral_data.reset_index(drop=True).copy()
        self.start_wavelength = start_wavelength
        self.end_wavelength = end_wavelength
        self.n_spectra = self.spectral_data.shape[1]
        self.current_index = 0
        self.file_label = str(initial_file_label).strip() or 'U'
        self.saved_output_path = None
        self.mark_list = self._normalize_mark_list(mark_list)

        if self.n_spectra == 0:
            raise ValueError('输入数据中没有可打标的光谱列。')

        self.labels = ['U'] * self.n_spectra
        if initial_labels is not None:
            for i, value in enumerate(list(initial_labels)[:self.n_spectra]):
                self.labels[i] = self._clean_label(value)
        self.confidences = [1.0] * self.n_spectra

        self.master.title('Manual Spectral Labeling | MM_v3')
        self.master.geometry('1100x760')

        defaults = list(button_label_defaults or ['I', 'MB', ''])
        while len(defaults) < 3:
            defaults.append('')

        self.current_label_var = tk.StringVar(value='U')
        self.progress_var = tk.StringVar(value='')
        self.position_var = tk.StringVar(value='')

        self._build_ui(defaults)
        self.update_plot()

    def _clean_label(self, value):
        text = str(value).strip()
        if text == '' or '\t' in text or '\n' in text or '\r' in text:
            return 'U'
        return text

    def _normalize_mark_list(self, values):
        if values is None:
            return []
        if isinstance(values, str):
            raw_items = re.split(r'[,;\s]+', values.strip())
        else:
            raw_items = values
        marks = []
        for item in raw_items:
            s = str(item).strip()
            if not s:
                continue
            try:
                marks.append(float(s))
            except Exception:
                continue
        return marks

    def _parse_mark_list_from_entry(self):
        return self._normalize_mark_list(self.mark_entry.get())

    def refresh_mark_lines(self):
        self.mark_list = self._parse_mark_list_from_entry()
        self.update_plot()

    def _build_ui(self, defaults):
        container = tk.Frame(self.master)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        plot_frame = tk.Frame(container)
        plot_frame.pack(side='left', fill='both', expand=True)

        control_frame = tk.Frame(container, width=320)
        control_frame.pack(side='right', fill='y', padx=(10, 0))
        control_frame.pack_propagate(False)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)


        tk.Label(control_frame, text='版本：MM_v3（诊断版）', fg='blue', font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 6))
        tk.Label(control_frame, text='打标状态', font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        tk.Label(control_frame, textvariable=self.progress_var, justify='left').pack(anchor='w', pady=2)
        tk.Label(control_frame, textvariable=self.position_var, justify='left').pack(anchor='w', pady=2)
        tk.Label(control_frame, text='当前标签：').pack(anchor='w', pady=(12, 2))
        tk.Label(control_frame, textvariable=self.current_label_var, fg='blue', font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 12))

        tk.Label(control_frame, text='辅助线峰位（逗号分隔）').pack(anchor='w', pady=(2, 4))
        mark_row = tk.Frame(control_frame)
        mark_row.pack(fill='x', pady=(0, 10))
        self.mark_entry = tk.Entry(mark_row)
        if self.mark_list:
            self.mark_entry.insert(0, ', '.join(f'{x:g}' for x in self.mark_list))
        self.mark_entry.pack(side='left', fill='x', expand=True)
        tk.Button(mark_row, text='更新辅助线', command=self.refresh_mark_lines).pack(side='left', padx=(6, 0))

        self.label_entries = []
        for idx in range(3):
            row = tk.Frame(control_frame)
            row.pack(fill='x', pady=6)
            tk.Label(row, text=f'标签{idx + 1}:', width=7, anchor='w').pack(side='left')
            entry = tk.Entry(row)
            entry.insert(0, defaults[idx])
            entry.pack(side='left', fill='x', expand=True, padx=4)
            tk.Button(row, text='标记', command=lambda i=idx: self.apply_label_from_entry(i)).pack(side='left', padx=(4, 0))
            self.label_entries.append(entry)

        nav_frame = tk.Frame(control_frame)
        nav_frame.pack(fill='x', pady=(18, 8))
        tk.Button(nav_frame, text='上一条', command=self.go_previous).pack(side='left', fill='x', expand=True)
        tk.Button(nav_frame, text='下一条（保持U）', command=self.go_next).pack(side='left', fill='x', expand=True, padx=6)

        tk.Button(control_frame, text='完成打标', command=self.finish_labeling, height=2).pack(fill='x', pady=(10, 0))
        tk.Label(control_frame, text='提示：未打标或跳过的光谱会保持为 U。', justify='left', fg='gray').pack(anchor='w', pady=(12, 0))

        self.master.bind('<Left>', lambda event: self.go_previous())
        self.master.bind('<Right>', lambda event: self.go_next())
        self.master.bind('<Return>', lambda event: self.apply_label_from_entry(0))

    def _get_plot_slice(self):
        wavelength_values = pd.to_numeric(self.wavelengths, errors='coerce').to_numpy()
        valid_mask = np.isfinite(wavelength_values)
        if not np.any(valid_mask):
            return self.wavelengths, self.spectral_data.iloc[:, self.current_index]

        valid_idx = np.where(valid_mask)[0]
        start_candidates = np.where(wavelength_values >= self.start_wavelength)[0]
        end_candidates = np.where(wavelength_values > self.end_wavelength)[0]
        start_idx = int(start_candidates[0]) if len(start_candidates) > 0 else int(valid_idx[0])
        end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(wavelength_values)
        if end_idx <= start_idx:
            start_idx, end_idx = 0, len(wavelength_values)
        return self.wavelengths.iloc[start_idx:end_idx], self.spectral_data.iloc[start_idx:end_idx, self.current_index]

    def update_plot(self):
        selected_wavelengths, selected_intensity = self._get_plot_slice()
        column_label = str(self.spectral_data.columns[self.current_index])
        current_label = self.labels[self.current_index]

        self.ax.clear()
        self.ax.plot(selected_wavelengths, selected_intensity)

        ymin = float(np.nanmin(selected_intensity)) if len(selected_intensity) else 0.0
        ymax = float(np.nanmax(selected_intensity)) if len(selected_intensity) else 1.0
        if not np.isfinite(ymin):
            ymin = 0.0
        if not np.isfinite(ymax):
            ymax = 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0

        for mark in self.mark_list:
            self.ax.axvline(x=mark, color='r', linestyle='--', linewidth=1.0, alpha=0.8)
            self.ax.text(mark, ymax - 0.05 * (ymax - ymin), f'{mark:g}',
                         rotation=90, ha='right', va='top', color='r', fontsize=9)

        self.ax.set_title(f'{column_label} | {self.current_index + 1}/{self.n_spectra}')
        self.ax.set_xlabel('Raman shift (cm^-1)')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(alpha=0.2)
        self.canvas.draw()

        self.progress_var.set(f'当前进度：{self.current_index + 1} / {self.n_spectra}')
        self.position_var.set(f'列名/位置：{column_label}')
        self.current_label_var.set(current_label)

    def apply_label_from_entry(self, entry_index):
        label = self._clean_label(self.label_entries[entry_index].get())
        self.labels[self.current_index] = label
        self.current_label_var.set(label)
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
        self.update_plot()

    def go_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

    def go_next(self):
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
            self.update_plot()
        else:
            self.finish_labeling()

    def finish_labeling(self):
        value = simpledialog.askstring(
            '富光谱级别标签',
            '请输入富光谱级别标签（留空默认U）：',
            initialvalue=self.file_label,
            parent=self.master,
        )
        self.file_label = self._clean_label(value)
        output_path = self.processor.save_manual_marked_data(
            source_file_path=self.input_file,
            wavelengths=self.wavelengths,
            spectral_data=self.spectral_data,
            spectrum_labels=self.labels,
            file_label=self.file_label,
            confidences=self.confidences,
            output_dir=self.output_dir,
            suffix='_MM',
        )
        self.saved_output_path = output_path
        messagebox.showinfo('保存完成', f'已保存：\n{output_path}', parent=self.master)
        self.master.destroy()

if __name__ == "__main__" :


    # 定义文件路径
    file_path = 'D:/北工博士阶段/论文冲冲冲/LIG-EDAg-LINE-SERS/AI/conc_txt/0.5_xy0.txt'

    #result1 = read_conc(file_path)
    data_array = read_Raman(file_path)
    Raman2Dresult1 = data_array[:,1].reshape(1,data_array.shape[0])
    conc=read_conc(file_path)
    #wn_range = [0,1000]
    #x,y_normal = Data_Interception(wn_range,data_array,normalization=True)
    #data_array1 = np.column_stack((x, y_normal))

    wn_range = [0,1000]
    x,y_normal = Data_Interception(wn_range,data_array,normalization=False)
    baseline,baseline_removed_data= baseline_als(y_normal, lam=10**4,p=0.001,niter=3)  # 调用之前实现的基线漂移去除方法
    data_array2 = np.column_stack((x, baseline_removed_data))
    #Raman2Dresult = Recurrence_plot(wn_range,data_array)
    #wn_range = [500,1800] # 截取的信号区域
    #Raman2Dresult = Recurrence_plot(wn_range,data_array)
    #Raman2Dresult = Gramian_angular(wn_range,data_array,transformation='s')
    Raman2Dresult = Short_time_Fourier_transform(wn_range,data_array,fs =1044,window='hann',nperseg=10)
    # = Markov_transition_field(wn_range,data_array,Q=150)

    #x,y_normal = Data_Interception(wn_range,data_array)

    #plt.imshow(Raman2Dresult, cmap='rainbow', interpolation='none', origin='lower',extent=(x.min(),x.max(),x.min(),x.max()))
    #plt.colorbar(fraction=0.0457, pad=0.04)
    #plt.show()


    #x = data_array[:,0]


    #plt.plot(x,y_normal)
    #plt.plot(x,baseline)
    #plt.plot(x,baseline_removed_data)
    #plt.show()
    #plt.imshow(Raman2Dresult, cmap='rainbow', origin='lower',vmin=0,vmax=1)
    #plt.title(file_path)
    #plt.show()

    # 假设我们有一个一维序列数据
    #sequence = np.random.rand(100, 2)
    # 我们首先需要确定x和y轴的区间数

    num_intervals = 1000
    x_range = [np.min(data_array2[:,0]),np.max(data_array2[:,0])]
    y_range = [np.min(data_array2[:,1]),np.max(data_array2[:,1])]
    step_x = (x_range[1]-x_range[0])/num_intervals
    step_y = (y_range[1]-y_range[0])/num_intervals
    # 创建一个二维数组，大小等于区间数
    heatmap = np.zeros((num_intervals, num_intervals))

    # 对于序列中的每一个点
    for x, y in data_array2:
        # 找到其在二维图像中对应的格子
        j = int((x-x_range[0]-1e-6) / step_x)
        i = int((y-y_range[0]-1e-6 )/ step_y)

        # 将该格子的值设为1
        heatmap[i, j] = heatmap[i, j]+1
    for i in range(heatmap.shape[1]):
        index = np.argmax(heatmap[:,i])
        heatmap[0:index,i]=np.max(heatmap[:,i])

    # 使用热图将这个二维数组可视化
    plt.plot(data_array2[:,0],data_array2[:,1])
    plt.show()
    plt.imshow(heatmap, cmap='rainbow', origin='lower')
    plt.show()







