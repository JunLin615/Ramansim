# -*- coding: utf-8 -*-
"""
本文件功能：
读取一个预处理后的Witec mapping拉曼光谱txt，逐条人工打标，
最终输出一个与输入txt对应的“手工标记后的物理数据”文件（_MM.txt）。
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import Pretreatment as pre


class ManualLabelLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title('Manual Spectral Labeling Launcher | MM_v3')
        self.root.geometry('520x640')

        self.input_file = ''
        self.output_dir = ''

        self._build_ui()

    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=15, pady=15)


        tk.Label(main, text='版本：MM_v3（诊断版）', fg='blue', font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 8))
        tk.Label(main, text=f'预处理模块：{getattr(pre, "__file__", "unknown")}', anchor='w', justify='left', wraplength=580, fg='gray').pack(fill='x', pady=(0, 10))
        tk.Button(main, text='选择输入txt', command=self.select_input_file).pack(fill='x', pady=6)
        self.input_label = tk.Label(main, text='未选择输入文件', anchor='w', justify='left', wraplength=480)
        self.input_label.pack(fill='x')

        tk.Button(main, text='选择输出文件夹', command=self.select_output_directory).pack(fill='x', pady=(12, 6))
        self.output_label = tk.Label(main, text='未选择输出目录', anchor='w', justify='left', wraplength=480)
        self.output_label.pack(fill='x')

        tk.Label(main, text='三个快捷标签（可在打标窗口中继续修改）').pack(anchor='w', pady=(16, 6))
        self.label_entries = []
        defaults = ['I', 'MB', '']
        for i in range(3):
            row = tk.Frame(main)
            row.pack(fill='x', pady=4)
            tk.Label(row, text=f'标签{i + 1}:', width=8, anchor='w').pack(side='left')
            entry = tk.Entry(row)
            entry.insert(0, defaults[i])
            entry.pack(side='left', fill='x', expand=True)
            self.label_entries.append(entry)

        tk.Label(main, text='辅助线峰位（逗号分隔，例如 520, 1075, 1580）').pack(anchor='w', pady=(14, 6))
        self.mark_entry = tk.Entry(main)
        self.mark_entry.pack(fill='x')

        tk.Button(main, text='开始打标', command=self.process, height=2).pack(fill='x', pady=(18, 0))
        tk.Label(main, text='如果你能看到这行，但看不到上面的按钮，请告诉我。', fg='red').pack(anchor='w', pady=(8, 0))

    def select_input_file(self):
        path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if path:
            self.input_file = path
            self.input_label.config(text=path)
            if not self.output_dir:
                self.output_dir = os.path.dirname(path)
                self.output_label.config(text=self.output_dir)

    def select_output_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self.output_label.config(text=path)

    def process(self):
        if not self.input_file:
            messagebox.showerror('错误', '请先选择输入文件。')
            return
        if not self.output_dir:
            messagebox.showerror('错误', '请先选择输出目录。')
            return

        try:
            processor = pre.WitecRamanProcessor(
                input_dir=os.path.dirname(self.input_file),
                output_dir=self.output_dir,
                compat_mode=True,
                header_mode='auto',
            )
            wavelengths, spectral_data = processor.read_data(self.input_file, delimiter=None)
        except Exception as exc:
            messagebox.showerror('读取失败', f'文件读取失败：\n{exc}')
            return

        if spectral_data.shape[1] == 0:
            messagebox.showerror('读取失败', '输入文件中没有检测到可打标的光谱列。')
            return

        defaults = [entry.get() for entry in self.label_entries]
        mark_list = self.mark_entry.get()
        manual_bundle = getattr(processor, '_last_manual_label_bundle', None) or {}
        initial_labels = manual_bundle.get('spectrum_labels')
        initial_file_label = manual_bundle.get('file_label', 'U')

        self.root.withdraw()
        top = tk.Toplevel(self.root)
        app = pre.SpectralLabelingApp(
            master=top,
            processor=processor,
            input_file=self.input_file,
            output_dir=self.output_dir,
            wavelengths=wavelengths,
            spectral_data=spectral_data,
            start_wavelength=500,
            end_wavelength=1800,
            button_label_defaults=defaults,
            initial_labels=initial_labels,
            initial_file_label=initial_file_label,
            mark_list=mark_list,
        )
        self.root.wait_window(top)
        self.root.deiconify()

        if getattr(app, 'saved_output_path', None):
            messagebox.showinfo('完成', f'输出文件：\n{app.saved_output_path}')


if __name__ == '__main__':
    root = tk.Tk()
    app = ManualLabelLauncher(root)
    root.mainloop()
