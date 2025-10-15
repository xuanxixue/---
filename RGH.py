import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re
import pandas as pd
import random
import os
from typing import List, Tuple, Dict, Set
import threading
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import tempfile
import webbrowser
from pathlib import Path

class DocumentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📚 文档句子分析系统")
        self.root.geometry("1200x800")
        
        # 存储数据
        self.sentences = []
        self.analysis_data = []
        self.words_data = []
        self.words_value_data = []
        self.char_mapping = {}  # 字符映射表：字符 -> 在不同句子中的数值列表
        self.relation_data = []  # 字符词语关系分析数据
        self.stable_mappings = {}  # 稳定的映射关系
        
        self.setup_ui()
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="📚 文档句子分析系统", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="文档文件:").grid(row=0, column=0, sticky=tk.W)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(file_frame, text="提取句子", command=self.extract_sentences).grid(row=0, column=3, padx=(5, 0))
        
        # 统计信息区域
        stats_frame = ttk.LabelFrame(main_frame, text="统计信息", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_text = tk.StringVar()
        self.stats_text.set("总句子数: 0\n平均长度: 0.0\n最长句子: 0\n最短句子: 0")
        ttk.Label(stats_frame, textvariable=self.stats_text, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W)
        
        # 创建笔记本（选项卡）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # 提取结果标签页
        self.extract_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.extract_tab, text="提取结果")
        self.setup_extract_tab()
        
        # 字符分析标签页
        self.analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_tab, text="字符组合分析")
        self.setup_analysis_tab()
        
        # 字符映射表标签页
        self.char_mapping_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.char_mapping_tab, text="字符映射表")
        self.setup_char_mapping_tab()
        
        # 词语分析标签页
        self.words_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.words_tab, text="词语分析")
        self.setup_words_tab()
        
        # 词语数值分配标签页
        self.words_value_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.words_value_tab, text="词语数值分配")
        self.setup_words_value_tab()
        
        # 详细示例标签页
        self.example_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.example_tab, text="详细示例")
        self.setup_example_tab()
        
        # 图表分析标签页
        self.chart_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.chart_tab, text="图表分析")
        self.setup_chart_tab()
        
        # 字符思维网络标签页
        self.flow_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.flow_tab, text="字符思维网络")
        self.setup_flow_tab()
        
        # 句子还原测试标签页
        self.restore_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.restore_tab, text="句子还原测试")
        self.setup_restore_tab()
        
        # 字符词语关系分析标签页
        self.relation_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.relation_tab, text="字符词语关系分析")
        self.setup_relation_tab()
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def setup_extract_tab(self):
        self.extract_tab.columnconfigure(0, weight=1)
        self.extract_tab.rowconfigure(0, weight=1)
        
        # 创建表格框架
        table_frame = ttk.Frame(self.extract_tab)
        table_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 创建滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 创建Treeview表格
        self.extract_tree = ttk.Treeview(table_frame, columns=('序号', '句子', '长度'), 
                                        show='headings', 
                                        yscrollcommand=scrollbar_y.set,
                                        xscrollcommand=scrollbar_x.set)
        
        # 配置列
        self.extract_tree.heading('序号', text='序号')
        self.extract_tree.heading('句子', text='句子')
        self.extract_tree.heading('长度', text='长度')
        
        self.extract_tree.column('序号', width=80, anchor=tk.CENTER)
        self.extract_tree.column('句子', width=600, anchor=tk.W)
        self.extract_tree.column('长度', width=80, anchor=tk.CENTER)
        
        self.extract_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置滚动条
        scrollbar_y.config(command=self.extract_tree.yview)
        scrollbar_x.config(command=self.extract_tree.xview)
    
    def setup_analysis_tab(self):
        # 分析控制区域
        control_frame = ttk.Frame(self.analysis_tab)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="起始序号:").grid(row=0, column=0, padx=(0, 5))
        self.start_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.start_seq_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(control_frame, text="结束序号:").grid(row=0, column=2, padx=(0, 5))
        self.end_seq_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.end_seq_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Button(control_frame, text="生成分析", command=self.generate_analysis).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="保存结果", command=self.save_analysis).grid(row=0, column=5)
        
        # 分析结果表格
        self.analysis_tab.columnconfigure(0, weight=1)
        self.analysis_tab.rowconfigure(1, weight=1)
        
        table_frame = ttk.Frame(self.analysis_tab)
        table_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Treeview表格
        self.analysis_tree = ttk.Treeview(table_frame, 
                                         columns=('序号', '句子', '长度', '字符组合', '数值', '总和验证'), 
                                         show='headings',
                                         yscrollcommand=scrollbar_y.set,
                                         xscrollcommand=scrollbar_x.set)
        
        # 配置列
        columns_config = {
            '序号': 80, '句子': 200, '长度': 60, 
            '字符组合': 300, '数值': 200, '总和验证': 80
        }
        
        for col, width in columns_config.items():
            self.analysis_tree.heading(col, text=col)
            self.analysis_tree.column(col, width=width, anchor=tk.CENTER)
        
        self.analysis_tree.column('句子', anchor=tk.W)
        self.analysis_tree.column('字符组合', anchor=tk.W)
        self.analysis_tree.column('数值', anchor=tk.W)
        
        self.analysis_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=self.analysis_tree.yview)
        scrollbar_x.config(command=self.analysis_tree.xview)
    
    def setup_char_mapping_tab(self):
        """设置字符映射表标签页"""
        self.char_mapping_tab.columnconfigure(0, weight=1)
        self.char_mapping_tab.rowconfigure(1, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(self.char_mapping_tab)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="生成字符映射表", command=self.generate_char_mapping).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="保存映射表", command=self.save_char_mapping).grid(row=0, column=1, padx=(0, 10))
        
        # 字符映射表统计信息
        self.char_mapping_stats_var = tk.StringVar()
        self.char_mapping_stats_var.set("总字符数: 0\n总映射数: 0\n最多数值字符: 0")
        ttk.Label(control_frame, textvariable=self.char_mapping_stats_var, justify=tk.LEFT).grid(row=0, column=2, padx=(20, 0))
        
        # 字符映射表结果
        table_frame = ttk.Frame(self.char_mapping_tab)
        table_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Treeview表格
        self.char_mapping_tree = ttk.Treeview(table_frame, 
                                            columns=('序号', '字符', '数值列表', '出现次数'), 
                                            show='headings',
                                            yscrollcommand=scrollbar_y.set,
                                            xscrollcommand=scrollbar_x.set)
        
        # 配置列
        columns_config = {
            '序号': 60, '字符': 80, '数值列表': 400, '出现次数': 80
        }
        
        for col, width in columns_config.items():
            self.char_mapping_tree.heading(col, text=col)
            self.char_mapping_tree.column(col, width=width, anchor=tk.CENTER)
        
        self.char_mapping_tree.column('数值列表', anchor=tk.W)
        
        self.char_mapping_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=self.char_mapping_tree.yview)
        scrollbar_x.config(command=self.char_mapping_tree.xview)
    
    def setup_words_tab(self):
        # 词语分析控制区域
        control_frame = ttk.Frame(self.words_tab)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="起始序号:").grid(row=0, column=0, padx=(0, 5))
        self.words_start_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.words_start_seq_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(control_frame, text="结束序号:").grid(row=0, column=2, padx=(0, 5))
        self.words_end_seq_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.words_end_seq_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Button(control_frame, text="分词分析", command=self.generate_words_analysis).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="保存词语", command=self.save_words_analysis).grid(row=0, column=5)
        
        # 词语统计信息
        self.words_stats_var = tk.StringVar()
        self.words_stats_var.set("总词语数: 0\n唯一词语数: 0\n平均词语长度: 0.0")
        ttk.Label(control_frame, textvariable=self.words_stats_var, justify=tk.LEFT).grid(row=0, column=6, padx=(20, 0))
        
        # 词语结果表格
        self.words_tab.columnconfigure(0, weight=1)
        self.words_tab.rowconfigure(1, weight=1)
        
        table_frame = ttk.Frame(self.words_tab)
        table_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Treeview表格
        self.words_tree = ttk.Treeview(table_frame, 
                                      columns=('序号', '词语', '字符长度', '词频', '句子来源'), 
                                      show='headings',
                                      yscrollcommand=scrollbar_y.set,
                                      xscrollcommand=scrollbar_x.set)
        
        # 配置列
        columns_config = {
            '序号': 60, '词语': 150, '字符长度': 80, '词频': 60, '句子来源': 300
        }
        
        for col, width in columns_config.items():
            self.words_tree.heading(col, text=col)
            self.words_tree.column(col, width=width, anchor=tk.CENTER)
        
        self.words_tree.column('词语', anchor=tk.W)
        self.words_tree.column('句子来源', anchor=tk.W)
        
        self.words_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=self.words_tree.yview)
        scrollbar_x.config(command=self.words_tree.xview)
    
    def setup_words_value_tab(self):
        # 词语数值分配控制区域
        control_frame = ttk.Frame(self.words_value_tab)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="起始序号:").grid(row=0, column=0, padx=(0, 5))
        self.words_value_start_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.words_value_start_seq_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(control_frame, text="结束序号:").grid(row=0, column=2, padx=(0, 5))
        self.words_value_end_seq_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.words_value_end_seq_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Button(control_frame, text="生成词语数值分配", command=self.generate_words_value_analysis).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="保存结果", command=self.save_words_value_analysis).grid(row=0, column=5)
        
        # 词语数值分配结果表格
        self.words_value_tab.columnconfigure(0, weight=1)
        self.words_value_tab.rowconfigure(1, weight=1)
        
        table_frame = ttk.Frame(self.words_value_tab)
        table_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Treeview表格
        self.words_value_tree = ttk.Treeview(table_frame, 
                                           columns=('序号', '句子', '词语数量', '词语组合', '数值', '总和验证'), 
                                           show='headings',
                                           yscrollcommand=scrollbar_y.set,
                                           xscrollcommand=scrollbar_x.set)
        
        # 配置列
        columns_config = {
            '序号': 80, '句子': 200, '词语数量': 80, 
            '词语组合': 300, '数值': 200, '总和验证': 80
        }
        
        for col, width in columns_config.items():
            self.words_value_tree.heading(col, text=col)
            self.words_value_tree.column(col, width=width, anchor=tk.CENTER)
        
        self.words_value_tree.column('句子', anchor=tk.W)
        self.words_value_tree.column('词语组合', anchor=tk.W)
        self.words_value_tree.column('数值', anchor=tk.W)
        
        self.words_value_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=self.words_value_tree.yview)
        scrollbar_x.config(command=self.words_value_tree.xview)
    
    def setup_example_tab(self):
        self.example_tab.columnconfigure(0, weight=1)
        self.example_tab.rowconfigure(1, weight=1)
        
        # 示例信息
        info_frame = ttk.Frame(self.example_tab)
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.example_info_var = tk.StringVar()
        self.example_info_var.set("请先生成分析结果")
        ttk.Label(info_frame, textvariable=self.example_info_var, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W)
        
        # 示例表格
        table_frame = ttk.Frame(self.example_tab)
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.example_tree = ttk.Treeview(table_frame, columns=('字符', '数值'), 
                                        show='headings',
                                        yscrollcommand=scrollbar_y.set)
        
        self.example_tree.heading('字符', text='字符')
        self.example_tree.heading('数值', text='数值')
        self.example_tree.column('字符', width=100, anchor=tk.CENTER)
        self.example_tree.column('数值', width=100, anchor=tk.CENTER)
        
        self.example_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.config(command=self.example_tree.yview)
    
    def setup_chart_tab(self):
        """设置图表分析标签页"""
        self.chart_tab.columnconfigure(0, weight=1)
        self.chart_tab.rowconfigure(1, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(self.chart_tab)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="选择句子序号:").grid(row=0, column=0, padx=(0, 5))
        self.chart_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.chart_seq_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Button(control_frame, text="生成字符数值图", command=self.generate_char_chart).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(control_frame, text="生成句子长度分布", command=self.generate_length_chart).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(control_frame, text="生成词语频率图", command=self.generate_word_freq_chart).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="保存图表", command=self.save_chart).grid(row=0, column=5)
        
        # 图表显示区域
        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chart_frame.columnconfigure(0, weight=1)
        self.chart_frame.rowconfigure(0, weight=1)
        
        # 状态标签
        self.chart_status_var = tk.StringVar()
        self.chart_status_var.set("请生成图表")
        ttk.Label(self.chart_frame, textvariable=self.chart_status_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def setup_flow_tab(self):
        """设置字符思维网络标签页"""
        self.flow_tab.columnconfigure(0, weight=1)
        self.flow_tab.rowconfigure(1, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(self.flow_tab)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="生成思维网络", command=self.generate_mind_network).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="在浏览器中打开", command=self.open_network_in_browser).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="保存网络图", command=self.save_flow_chart).grid(row=0, column=2, padx=(0, 10))
        
        # 状态标签
        self.flow_status_var = tk.StringVar()
        self.flow_status_var.set("请生成思维网络")
        ttk.Label(control_frame, textvariable=self.flow_status_var).grid(row=0, column=3, padx=(20, 0))
        
        # 网络图显示区域
        self.flow_frame = ttk.Frame(self.flow_tab)
        self.flow_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.flow_frame.columnconfigure(0, weight=1)
        self.flow_frame.rowconfigure(0, weight=1)
        
        # 状态标签
        ttk.Label(self.flow_frame, text="思维网络将在浏览器中显示，请点击'在浏览器中打开'查看").grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def setup_restore_tab(self):
        """设置句子还原测试标签页"""
        self.restore_tab.columnconfigure(0, weight=1)
        self.restore_tab.rowconfigure(1, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(self.restore_tab)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="测试模式:").grid(row=0, column=0, padx=(0, 5))
        
        self.test_mode_var = tk.StringVar(value="random")
        test_mode_frame = ttk.Frame(control_frame)
        test_mode_frame.grid(row=0, column=1, padx=(0, 15))
        
        ttk.Radiobutton(test_mode_frame, text="随机测试", variable=self.test_mode_var, value="random").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(test_mode_frame, text="指定句子", variable=self.test_mode_var, value="specific").grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(control_frame, text="句子序号:").grid(row=0, column=2, padx=(0, 5))
        self.restore_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.restore_seq_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Button(control_frame, text="生成测试", command=self.generate_restore_test).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="检查答案", command=self.check_restore_answer).grid(row=0, column=5, padx=(0, 10))
        ttk.Button(control_frame, text="显示答案", command=self.show_restore_answer).grid(row=0, column=6)
        
        # 测试结果显示区域
        self.restore_tab.columnconfigure(0, weight=1)
        self.restore_tab.rowconfigure(1, weight=1)
        
        # 创建上下分栏
        paned_window = ttk.PanedWindow(self.restore_tab, orient=tk.VERTICAL)
        paned_window.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 上半部分：测试题目
        test_frame = ttk.LabelFrame(paned_window, text="测试题目", padding="10")
        paned_window.add(test_frame, weight=1)
        
        test_frame.columnconfigure(0, weight=1)
        test_frame.rowconfigure(1, weight=1)
        
        ttk.Label(test_frame, text="请根据字符映射关系和思维网络，还原以下句子:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        self.test_question_text = scrolledtext.ScrolledText(test_frame, wrap=tk.WORD, height=8)
        self.test_question_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 下半部分：用户答案和结果
        answer_frame = ttk.LabelFrame(paned_window, text="您的答案", padding="10")
        paned_window.add(answer_frame, weight=1)
        
        answer_frame.columnconfigure(0, weight=1)
        answer_frame.rowconfigure(1, weight=1)
        
        ttk.Label(answer_frame, text="请输入您还原的句子:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        self.user_answer_text = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD, height=6)
        self.user_answer_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果显示
        self.restore_result_var = tk.StringVar()
        self.restore_result_var.set("请生成测试题目并输入您的答案")
        ttk.Label(answer_frame, textvariable=self.restore_result_var, justify=tk.LEFT).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # 当前测试数据
        self.current_test_data = None
    
    def setup_relation_tab(self):
        """设置字符词语关系分析标签页"""
        self.relation_tab.columnconfigure(0, weight=1)
        self.relation_tab.rowconfigure(1, weight=1)
        
        # 控制区域
        control_frame = ttk.Frame(self.relation_tab)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="起始序号:").grid(row=0, column=0, padx=(0, 5))
        self.relation_start_seq_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.relation_start_seq_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(control_frame, text="结束序号:").grid(row=0, column=2, padx=(0, 5))
        self.relation_end_seq_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.relation_end_seq_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Button(control_frame, text="计算关系", command=self.calculate_char_word_relation).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="保存结果", command=self.save_relation_analysis).grid(row=0, column=5)
        
        # 关系分析表格
        self.relation_tab.columnconfigure(0, weight=1)
        self.relation_tab.rowconfigure(1, weight=1)
        
        table_frame = ttk.Frame(self.relation_tab)
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame)
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Treeview表格
        self.relation_tree = ttk.Treeview(table_frame, 
                                        columns=('序号', '句子', '字符组合分析', '词语数值分配', '关系分析'), 
                                        show='headings',
                                        yscrollcommand=scrollbar_y.set,
                                        xscrollcommand=scrollbar_x.set)
        
        # 配置列
        columns_config = {
            '序号': 60, '句子': 150, '字符组合分析': 200, '词语数值分配': 200, '关系分析': 300
        }
        
        for col, width in columns_config.items():
            self.relation_tree.heading(col, text=col)
            self.relation_tree.column(col, width=width, anchor=tk.CENTER)
        
        self.relation_tree.column('句子', anchor=tk.W)
        self.relation_tree.column('字符组合分析', anchor=tk.W)
        self.relation_tree.column('词语数值分配', anchor=tk.W)
        self.relation_tree.column('关系分析', anchor=tk.W)
        
        self.relation_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=self.relation_tree.yview)
        scrollbar_x.config(command=self.relation_tree.xview)
        
        # 综合关系分析区域
        summary_frame = ttk.LabelFrame(self.relation_tab, text="综合关系分析", padding="10")
        summary_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        summary_frame.columnconfigure(0, weight=1)
        
        self.relation_summary_var = tk.StringVar()
        self.relation_summary_var.set("请先计算字符与词语数值分配之间的关系")
        ttk.Label(summary_frame, textvariable=self.relation_summary_var, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W)
    
    def calculate_char_word_relation(self):
        """计算字符组合分析与词语数值分配之间的关系"""
        if not hasattr(self, 'analysis_data') or not self.analysis_data:
            messagebox.showwarning("警告", "请先生成字符组合分析")
            return
        
        if not hasattr(self, 'words_value_data') or not self.words_value_data:
            messagebox.showwarning("警告", "请先生成词语数值分配")
            return
        
        try:
            start_seq = int(self.relation_start_seq_var.get())
            end_seq = int(self.relation_end_seq_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的序号")
            return
        
        # 在后台线程中执行关系分析
        self.progress.start()
        thread = threading.Thread(target=self._calculate_char_word_relation_thread, args=(start_seq, end_seq))
        thread.daemon = True
        thread.start()

    def _calculate_char_word_relation_thread(self, start_seq, end_seq):
        """在后台线程中计算字符词语关系"""
        try:
            # 过滤数据
            char_analysis_data = [d for d in self.analysis_data if start_seq <= d['序号'] <= end_seq]
            word_value_data = [d for d in self.words_value_data if start_seq <= d['序号'] <= end_seq]
            
            if not char_analysis_data or not word_value_data:
                self.root.after(0, lambda: messagebox.showerror("错误", "没有找到符合条件的句子"))
                return
            
            # 构建关系分析结果
            relation_data = []
            
            for char_data, word_data in zip(char_analysis_data, word_value_data):
                if char_data['序号'] != word_data['序号']:
                    continue
                
                # 解析字符组合分析
                char_combination = char_data['字符组合']
                char_values_str = char_data['数值']
                
                # 解析字符和数值
                chars = re.findall(r"'([^']*)'", char_combination)
                char_values = [int(x.strip()) for x in char_values_str.split('+')]
                
                # 解析词语数值分配
                word_combination = word_data['词语组合']
                word_values_str = word_data['数值']
                
                # 解析词语和数值
                words = re.findall(r"'([^']*)'", word_combination)
                word_values = [int(x.strip()) for x in word_values_str.split('+')]
                
                # 计算字符与词语之间的映射关系
                relation_text = self._build_char_word_mapping(chars, char_values, words, word_values)
                
                relation_data.append({
                    '序号': char_data['序号'],
                    '句子': char_data['句子'],
                    '字符组合分析': char_values_str,
                    '词语数值分配': word_values_str,
                    '关系分析': relation_text
                })
            
            # 在主线程中更新UI
            self.root.after(0, self._update_relation_results, relation_data)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"计算关系时出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)

    def _build_char_word_mapping(self, chars, char_values, words, word_values):
        """构建字符与词语的映射关系"""
        try:
            if not chars or not words:
                return "无法建立映射关系"
            
            # 分析字符如何组合成词语
            mapping_relations = []
            char_index = 0
            
            for word, word_value in zip(words, word_values):
                word_length = len(word)
                
                # 获取词语对应的字符
                word_chars = chars[char_index:char_index + word_length]
                word_char_values = char_values[char_index:char_index + word_length]
                
                # 构建映射关系字符串
                if len(word_chars) == 1:
                    # 单个字符对应单个词语
                    mapping_relations.append(f"{word_char_values[0]}={word_value}={word_chars[0]}")
                else:
                    # 多个字符对应一个词语
                    char_values_str = '+'.join(map(str, word_char_values))
                    mapping_relations.append(f"{char_values_str}={word_value}={word}")
                
                char_index += word_length
            
            # 检查是否有剩余的字符
            if char_index < len(chars):
                remaining_chars = chars[char_index:]
                remaining_values = char_values[char_index:]
                if remaining_chars:
                    char_values_str = '+'.join(map(str, remaining_values))
                    mapping_relations.append(f"{char_values_str}=?={'+'.join(remaining_chars)}")
            
            return ", ".join(mapping_relations)
            
        except Exception as e:
            return f"构建映射关系时出错: {str(e)}"

    def _update_relation_results(self, relation_data):
        """更新关系分析结果"""
        self.relation_data = relation_data
        
        # 清空现有数据
        for item in self.relation_tree.get_children():
            self.relation_tree.delete(item)
        
        # 添加新数据
        for data in relation_data:
            self.relation_tree.insert('', 'end', values=(
                data['序号'],
                data['句子'],
                data['字符组合分析'],
                data['词语数值分配'],
                data['关系分析']
            ))
        
        # 更新综合关系分析
        summary = self._generate_advanced_relation_summary(relation_data)
        self.relation_summary_var.set(summary)
        
        messagebox.showinfo("成功", f"成功分析 {len(relation_data)} 个句子的字符词语关系！")

    def _generate_advanced_relation_summary(self, relation_data):
        """生成高级关系分析摘要"""
        try:
            if not relation_data:
                return "没有可用的关系分析数据"
            
            # 收集所有映射关系
            all_mappings = {}
            char_word_mapping = {}  # 字符组合 -> 词语
            word_char_mapping = {}  # 词语 -> 字符组合
            
            for data in relation_data:
                relation_text = data['关系分析']
                mappings = relation_text.split(', ')
                
                for mapping in mappings:
                    if '=' in mapping:
                        parts = mapping.split('=')
                        if len(parts) >= 3:
                            char_part = parts[0]  # 字符数值部分
                            word_value = parts[1]  # 词语数值
                            word_part = parts[2]   # 词语部分
                            
                            # 记录映射关系
                            key = f"{char_part}→{word_value}"
                            if key not in all_mappings:
                                all_mappings[key] = []
                            all_mappings[key].append(word_part)
                            
                            # 记录双向映射
                            char_word_mapping[char_part] = word_part
                            word_char_mapping[word_part] = char_part
            
            # 保存稳定的映射关系
            self.stable_mappings = {}
            for key, words in all_mappings.items():
                if len(set(words)) == 1:  # 所有映射都指向同一个词语
                    char_part = key.split('→')[0]
                    self.stable_mappings[char_part] = words[0]
            
            # 生成摘要
            summary = f"高级关系分析摘要 ({len(relation_data)} 个句子)\n\n"
            summary += "发现的映射模式:\n"
            
            # 统计最常见的映射
            common_mappings = []
            for key, words in all_mappings.items():
                if len(set(words)) == 1:  # 所有映射都指向同一个词语
                    common_mappings.append((key, words[0], len(words)))
            
            # 按频率排序
            common_mappings.sort(key=lambda x: x[2], reverse=True)
            
            for i, (key, word, count) in enumerate(common_mappings[:10]):  # 显示前10个
                summary += f"{i+1}. {key} → '{word}' (出现{count}次)\n"
            
            summary += f"\n总映射模式数: {len(all_mappings)}"
            summary += f"\n稳定映射数: {len(common_mappings)}"
            
            # 添加使用建议
            summary += "\n\n使用建议:"
            summary += "\n1. 使用稳定映射模式来分词新句子"
            summary += "\n2. 对于未知字符组合，查找最相似的映射模式"
            summary += "\n3. 结合字符思维网络进行验证"
            
            return summary
            
        except Exception as e:
            return f"生成高级关系分析时出错: {str(e)}"

    def save_relation_analysis(self):
        """保存关系分析结果"""
        if not hasattr(self, 'relation_data') or not self.relation_data:
            messagebox.showwarning("警告", "没有可保存的关系分析数据")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存关系分析结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # 获取当前表格中的所有数据
                data = []
                for item in self.relation_tree.get_children():
                    values = self.relation_tree.item(item, 'values')
                    data.append({
                        '序号': values[0],
                        '句子': values[1],
                        '字符组合分析': values[2],
                        '词语数值分配': values[3],
                        '关系分析': values[4]
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"关系分析结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {e}")

    # 添加新方法：基于映射关系分词
    def tokenize_using_mappings(self, sentence, char_values):
        """使用已有的映射关系对句子进行分词"""
        try:
            if not hasattr(self, 'stable_mappings') or not self.stable_mappings:
                return self.tokenize_sentence(sentence)  # 回退到基本分词
            
            # 将字符数值转换为字符串格式以便匹配
            char_values_str = '+'.join(map(str, char_values))
            
            # 尝试使用映射关系进行分词
            words = []
            current_pos = 0
            chars = list(sentence)
            
            while current_pos < len(chars):
                # 尝试找到最长的匹配
                best_match = None
                best_match_length = 0
                
                for i in range(current_pos + 1, len(chars) + 1):
                    current_chars = chars[current_pos:i]
                    current_values = char_values[current_pos:i]
                    current_key = '+'.join(map(str, current_values))
                    
                    if current_key in self.stable_mappings:
                        if len(current_chars) > best_match_length:
                            best_match = self.stable_mappings[current_key]
                            best_match_length = len(current_chars)
                
                if best_match:
                    words.append(best_match)
                    current_pos += best_match_length
                else:
                    # 没有找到映射，使用单个字符
                    words.append(chars[current_pos])
                    current_pos += 1
            
            return words
            
        except Exception as e:
            print(f"使用映射分词时出错: {e}")
            return self.tokenize_sentence(sentence)  # 出错时回退到基本分词

    def generate_restore_test(self):
        """生成句子还原测试"""
        if not self.char_mapping:
            messagebox.showwarning("警告", "请先生成字符映射表和思维网络")
            return
        
        test_mode = self.test_mode_var.get()
        
        if test_mode == "specific":
            try:
                seq = int(self.restore_seq_var.get())
                # 查找指定句子
                target_sentence = None
                for s in self.sentences:
                    if s[0] == seq:
                        target_sentence = s
                        break
                
                if not target_sentence:
                    messagebox.showerror("错误", f"未找到序号为 {seq} 的句子")
                    return
                
                self._create_restore_test(target_sentence)
                
            except ValueError:
                messagebox.showerror("错误", "请输入有效的句子序号")
                return
        else:
            # 随机选择句子
            if not self.sentences:
                messagebox.showwarning("警告", "请先提取句子")
                return
            
            random_sentence = random.choice(self.sentences)
            self._create_restore_test(random_sentence)
    
    def _create_restore_test(self, sentence):
        """创建还原测试题目"""
        seq, text = sentence
        
        # 生成字符数值分配
        char_values = self.assign_values(text)
        
        # 创建测试题目 - 挖空部分字符
        test_chars = []
        answer_positions = []
        
        # 随机选择要挖空的字符位置（至少保留30%的字符）
        n = len(text)
        num_blanks = max(1, n // 3)  # 挖空约1/3的字符
        blank_positions = random.sample(range(n), num_blanks)
        
        for i, (char, value) in enumerate(char_values):
            if i in blank_positions:
                test_chars.append(f"[?({value})]")
                answer_positions.append((i, char, value))
            else:
                test_chars.append(f"{char}({value})")
        
        test_question = " ".join(test_chars)
        
        # 构建测试说明
        instructions = f"测试句子 #{seq}\n\n"
        instructions += "题目说明：\n"
        instructions += "- 方括号 [?(数值)] 表示需要还原的字符\n"
        instructions += "- 其他字符后的(数值)表示该字符的映射数值\n"
        instructions += "- 请根据字符映射表和思维网络关系还原完整的句子\n\n"
        instructions += "测试题目：\n"
        instructions += test_question
        
        # 显示测试题目
        self.test_question_text.delete(1.0, tk.END)
        self.test_question_text.insert(1.0, instructions)
        
        # 清空用户答案和结果
        self.user_answer_text.delete(1.0, tk.END)
        self.restore_result_var.set("请在上方输入您的答案")
        
        # 保存当前测试数据
        self.current_test_data = {
            'original_sentence': text,
            'seq': seq,
            'char_values': char_values,
            'answer_positions': answer_positions,
            'test_question': test_question
        }
        
        messagebox.showinfo("成功", f"已生成句子 #{seq} 的还原测试")
    
    def check_restore_answer(self):
        """检查用户答案"""
        if not self.current_test_data:
            messagebox.showwarning("警告", "请先生成测试题目")
            return
        
        user_answer = self.user_answer_text.get(1.0, tk.END).strip()
        if not user_answer:
            messagebox.showwarning("警告", "请输入您的答案")
            return
        
        original_sentence = self.current_test_data['original_sentence']
        seq = self.current_test_data['seq']
        
        # 简单的答案检查
        if user_answer == original_sentence:
            result = f"✅ 恭喜！答案完全正确！\n句子 #{seq}: {original_sentence}"
            self.restore_result_var.set(result)
            messagebox.showinfo("结果", "答案正确！")
        else:
            # 计算相似度
            similarity = self._calculate_similarity(user_answer, original_sentence)
            
            result = f"❌ 答案不完全正确\n"
            result += f"您的答案: {user_answer}\n"
            result += f"正确答案: {original_sentence}\n"
            result += f"相似度: {similarity:.1%}"
            
            self.restore_result_var.set(result)
            
            # 提供更详细的反馈
            self._provide_detailed_feedback(user_answer, original_sentence)
    
    def _calculate_similarity(self, answer, original):
        """计算两个句子的相似度"""
        # 简单的字符级别相似度计算
        if len(answer) == 0 or len(original) == 0:
            return 0.0
        
        # 使用集合计算Jaccard相似度
        set_answer = set(answer)
        set_original = set(original)
        
        intersection = len(set_answer & set_original)
        union = len(set_answer | set_original)
        
        jaccard_sim = intersection / union if union > 0 else 0
        
        # 使用序列相似度
        min_len = min(len(answer), len(original))
        if min_len == 0:
            sequence_sim = 0
        else:
            match_count = sum(1 for i in range(min_len) if answer[i] == original[i])
            sequence_sim = match_count / len(original)
        
        # 综合相似度
        overall_sim = (jaccard_sim + sequence_sim) / 2
        return overall_sim
    
    def _provide_detailed_feedback(self, user_answer, original_sentence):
        """提供详细的反馈信息"""
        feedback = "详细反馈：\n"
        
        # 长度比较
        if len(user_answer) != len(original_sentence):
            feedback += f"- 长度不匹配：您的答案有 {len(user_answer)} 个字符，正确答案有 {len(original_sentence)} 个字符\n"
        
        # 字符比较
        min_len = min(len(user_answer), len(original_sentence))
        wrong_positions = []
        
        for i in range(min_len):
            if user_answer[i] != original_sentence[i]:
                wrong_positions.append((i, user_answer[i], original_sentence[i]))
        
        if wrong_positions:
            feedback += f"- 有 {len(wrong_positions)} 个位置字符不正确：\n"
            for pos, user_char, correct_char in wrong_positions[:5]:  # 只显示前5个错误
                feedback += f"  位置 {pos+1}: 您输入 '{user_char}'，应为 '{correct_char}'\n"
            if len(wrong_positions) > 5:
                feedback += f"  还有 {len(wrong_positions) - 5} 个错误...\n"
        
        # 建议
        feedback += "\n建议：\n"
        feedback += "- 查看字符映射表确认数值对应的字符\n"
        feedback += "- 检查思维网络中字符的连接关系\n"
        feedback += "- 注意句子的语法和语义合理性\n"
        
        # 显示在结果中
        current_result = self.restore_result_var.get()
        self.restore_result_var.set(current_result + "\n\n" + feedback)
    
    def show_restore_answer(self):
        """显示正确答案"""
        if not self.current_test_data:
            messagebox.showwarning("警告", "请先生成测试题目")
            return
        
        original_sentence = self.current_test_data['original_sentence']
        seq = self.current_test_data['seq']
        char_values = self.current_test_data['char_values']
        
        # 显示完整答案
        answer_info = f"正确答案（句子 #{seq}）:\n"
        answer_info += f"完整句子: {original_sentence}\n\n"
        answer_info += "字符数值映射:\n"
        
        for char, value in char_values:
            answer_info += f"  '{char}' → {value}\n"
        
        answer_info += f"\n句子长度: {len(original_sentence)} 字符"
        answer_info += f"\n数值总和: {sum(val for _, val in char_values)}"
        
        # 在用户答案区域显示正确答案
        self.user_answer_text.delete(1.0, tk.END)
        self.user_answer_text.insert(1.0, original_sentence)
        
        self.restore_result_var.set(answer_info)
        messagebox.showinfo("正确答案", f"句子 #{seq}: {original_sentence}")

    def generate_mind_network(self):
        """生成字符思维网络"""
        if not self.char_mapping:
            messagebox.showwarning("警告", "请先生成字符映射表")
            return
        
        # 在后台线程中执行
        self.progress.start()
        self.flow_status_var.set("正在生成思维网络...")
        thread = threading.Thread(target=self._generate_mind_network_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_mind_network_thread(self):
        """在后台线程中生成思维网络"""
        try:
            # 构建思维网络数据
            network_data = self._build_mind_network_data()
            
            # 创建交互式网络图
            self._create_interactive_network(network_data)
            
            # 在主线程中更新状态
            self.root.after(0, lambda: self.flow_status_var.set("思维网络生成完成！请点击'在浏览器中打开'查看"))
            
        except Exception as e:
            error_msg = f"生成思维网络时出错: {str(e)}\n\n详细错误信息:\n{self._format_exception(e)}"
            self.root.after(0, lambda: self._show_error_dialog(error_msg))
            self.root.after(0, lambda: self.flow_status_var.set("生成失败"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _format_exception(self, e):
        """格式化异常信息"""
        import traceback
        tb_str = traceback.format_exc()
        return f"异常类型: {type(e).__name__}\n异常信息: {str(e)}\n\n堆栈跟踪:\n{tb_str}"
    
    def _show_error_dialog(self, error_msg):
        """显示可复制错误信息的对话框"""
        error_dialog = tk.Toplevel(self.root)
        error_dialog.title("思维网络生成错误")
        error_dialog.geometry("600x400")
        error_dialog.transient(self.root)
        error_dialog.grab_set()
        
        # 设置对话框位置在父窗口中心
        error_dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - error_dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - error_dialog.winfo_height()) // 2
        error_dialog.geometry(f"+{x}+{y}")
        
        # 创建框架
        main_frame = ttk.Frame(error_dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        error_dialog.columnconfigure(0, weight=1)
        error_dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 错误标题
        title_label = ttk.Label(main_frame, text="思维网络生成过程中发生错误", 
                               font=("Arial", 12, "bold"), foreground="red")
        title_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # 错误信息文本框
        error_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=70, height=15)
        error_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        error_text.insert(tk.END, error_msg)
        error_text.config(state=tk.DISABLED)  # 设置为只读
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=tk.E)
        
        # 复制按钮
        copy_button = ttk.Button(button_frame, text="复制错误信息", 
                                command=lambda: self._copy_to_clipboard(error_msg))
        copy_button.grid(row=0, column=0, padx=(0, 10))
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", 
                                 command=error_dialog.destroy)
        close_button.grid(row=0, column=1)
    
    def _copy_to_clipboard(self, text):
        """复制文本到剪贴板"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("成功", "错误信息已复制到剪贴板")
    
    def _build_mind_network_data(self):
        """构建思维网络数据 - 基于字符在句子中的顺序连接"""
        network_data = {
            'nodes': {},  # 节点: {位置: (x, y), 字符: char, 数值列表: values, 出现次数: count}
            'edges': [],  # 边: (起点, 终点, 权重, 句子来源)
        }
        
        # 为每个字符创建节点
        chars = list(self.char_mapping.keys())
        
        if not chars:
            return network_data
            
        # 计算每个字符的出现次数
        char_counts = {}
        for char, values in self.char_mapping.items():
            char_counts[char] = len(values)
        
        # 为节点分配位置（使用力导向布局的初始位置）
        # 使用圆形布局避免重叠
        radius = 10
        angle_step = 2 * np.pi / len(chars)
        
        for i, char in enumerate(chars):
            # 在圆形上分布节点
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            network_data['nodes'][char] = {
                'pos': (x, y),
                'char': char,
                'values': self.char_mapping[char],
                'count': char_counts[char]
            }
        
        # 基于字符在句子中的顺序建立连接关系
        if hasattr(self, 'analysis_data') and self.analysis_data:
            for analysis in self.analysis_data:
                sentence = analysis['句子']
                seq = analysis['序号']
                
                # 解析字符组合和数值
                char_combination = analysis['字符组合']
                values_str = analysis['数值']
                
                # 解析字符和数值
                try:
                    chars_in_sentence = re.findall(r"'([^']*)'", char_combination)
                    values = [int(x.strip()) for x in values_str.split('+')]
                    
                    # 确保字符和数值数量匹配
                    if len(chars_in_sentence) != len(values):
                        continue
                        
                    # 为句子中的字符建立顺序连接
                    for i in range(len(chars_in_sentence) - 1):
                        char1 = chars_in_sentence[i]
                        char2 = chars_in_sentence[i + 1]
                        value1 = values[i]
                        value2 = values[i + 1]
                        
                        # 确保字符在节点中
                        if char1 not in network_data['nodes'] or char2 not in network_data['nodes']:
                            continue
                            
                        # 计算连接权重（基于字符在句子中的位置关系，而不是频率）
                        # 使用简单的固定权重，因为重点是显示连接关系
                        weight = 1.0
                        
                        # 添加边
                        network_data['edges'].append({
                            'from': char1,
                            'to': char2,
                            'weight': weight,
                            'sentence': seq,
                            'sentence_text': sentence,
                            'from_value': value1,
                            'to_value': value2
                        })
                except Exception as e:
                    print(f"解析句子时出错: {e}")
                    continue
        
        return network_data
    
    def _create_interactive_network(self, network_data):
        """创建交互式网络图"""
        if not network_data['nodes']:
            return
        
        # 创建节点和边的数据
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # 计算节点大小和颜色
        counts = [node['count'] for node in network_data['nodes'].values()]
        max_count = max(counts) if counts else 1
        
        for char, node_info in network_data['nodes'].items():
            x, y = node_info['pos']
            count = node_info['count']
            values = node_info['values']
            
            node_x.append(x)
            node_y.append(y)
            
            # 节点文本
            value_str = '/'.join(map(str, sorted(values)))
            text = f"字符: {char}<br>出现次数: {count}<br>数值列表: {value_str}"
            node_text.append(text)
            
            # 节点大小基于出现次数
            size = 15 + (count / max_count) * 25
            node_size.append(size)
            
            # 节点颜色基于数值的多样性
            value_diversity = len(set(values)) / len(values) if values else 0
            node_color.append(value_diversity)
        
        # 创建节点trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node['char'] for node in network_data['nodes'].values()],
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='数值多样性',
                        side='right'
                    ),
                    thickness=15
                ),
                line=dict(width=2, color='darkblue')
            ),
            hovertext=node_text
        )
        
        # 创建边trace
        edge_traces = []
        
        # 限制边数量，避免过于混乱
        edges_to_show = network_data['edges'][:min(200, len(network_data['edges']))]
        
        for edge in edges_to_show:
            from_char = edge['from']
            to_char = edge['to']
            
            if from_char in network_data['nodes'] and to_char in network_data['nodes']:
                from_pos = network_data['nodes'][from_char]['pos']
                to_pos = network_data['nodes'][to_char]['pos']
                
                # 边的颜色和样式
                color = 'rgba(128, 128, 128, 0.5)'
                linewidth = 1
                
                # 创建边trace
                edge_trace = go.Scatter(
                    x=[from_pos[0], to_pos[0], None],
                    y=[from_pos[1], to_pos[1], None],
                    mode='lines',
                    line=dict(width=linewidth, color=color),
                    hoverinfo='text',
                    text=f"从: {from_char}({edge['from_value']}) → 到: {to_char}({edge['to_value']})<br>句子 {edge['sentence']}: {edge['sentence_text']}",
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # 创建图形
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text='字符思维网络 - 基于句子顺序连接',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="节点大小表示字符出现频率，颜色表示数值多样性，连线表示字符在句子中的顺序关系",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            # 设置图形尺寸
            width=1000,
            height=800,
            # 添加缩放和平移功能
            dragmode='pan'
        )
        
        # 保存为HTML文件
        self.network_html_path = tempfile.NamedTemporaryFile(
            suffix='.html', delete=False, prefix='mind_network_'
        ).name
        
        pyo.plot(fig, filename=self.network_html_path, auto_open=False)
        
        # 添加统计信息到文件
        with open(self.network_html_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 在文件末尾添加统计信息
        stats_html = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
            <h4>网络统计</h4>
            <p>总字符数: {len(network_data['nodes'])}</p>
            <p>总连接数: {len(network_data['edges'])}</p>
            <p>显示连接数: {len(edges_to_show)}</p>
        </div>
        """
        
        # 将统计信息插入到body标签内
        content = content.replace('</body>', stats_html + '</body>')
        
        with open(self.network_html_path, 'w', encoding='utf-8') as file:
            file.write(content)
    
    def open_network_in_browser(self):
        """在浏览器中打开网络图"""
        if hasattr(self, 'network_html_path') and os.path.exists(self.network_html_path):
            webbrowser.open('file://' + os.path.abspath(self.network_html_path))
        else:
            messagebox.showwarning("警告", "请先生成思维网络")
    
    def save_flow_chart(self):
        """保存当前思维网络图"""
        if not hasattr(self, 'network_html_path') or not os.path.exists(self.network_html_path):
            messagebox.showwarning("警告", "没有可保存的思维网络图")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存思维网络图",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import shutil
                shutil.copy2(self.network_html_path, filename)
                messagebox.showinfo("成功", f"思维网络图已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存思维网络图时出错: {e}")

    def generate_char_mapping(self):
        """生成字符映射表"""
        if not hasattr(self, 'analysis_data') or not self.analysis_data:
            messagebox.showwarning("警告", "请先生成字符组合分析")
            return
        
        # 在后台线程中执行
        self.progress.start()
        thread = threading.Thread(target=self._generate_char_mapping_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_char_mapping_thread(self):
        """在后台线程中生成字符映射表"""
        try:
            # 清空现有的字符映射表
            self.char_mapping = {}
            
            # 遍历所有分析数据，收集字符和对应的数值
            for analysis in self.analysis_data:
                # 从字符组合列解析出字符和数值
                char_combination = analysis['字符组合']
                values_str = analysis['数值']
                
                # 解析字符组合
                chars = re.findall(r"'([^']*)'", char_combination)
                # 解析数值
                values = [int(x.strip()) for x in values_str.split('+')]
                
                # 确保字符和数值数量匹配
                if len(chars) != len(values):
                    continue
                    
                # 将字符和数值对应起来
                for char, value in zip(chars, values):
                    if char not in self.char_mapping:
                        self.char_mapping[char] = []
                    if value not in self.char_mapping[char]:
                        self.char_mapping[char].append(value)
            
            # 在主线程中更新UI
            self.root.after(0, self._update_char_mapping_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"生成字符映射表时出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _update_char_mapping_results(self):
        """更新字符映射表结果"""
        # 清空现有数据
        for item in self.char_mapping_tree.get_children():
            self.char_mapping_tree.delete(item)
        
        # 添加新数据
        char_id = 1
        for char, values in sorted(self.char_mapping.items()):
            # 格式化数值列表
            values_str = '/'.join(map(str, sorted(values)))
            count = len(values)
            
            self.char_mapping_tree.insert('', 'end', values=(
                char_id, char, values_str, count
            ))
            char_id += 1
        
        # 更新统计信息
        if self.char_mapping:
            total_chars = len(self.char_mapping)
            total_mappings = sum(len(values) for values in self.char_mapping.values())
            max_values_char = max(self.char_mapping.items(), key=lambda x: len(x[1]))[0]
            max_values_count = len(self.char_mapping[max_values_char])
            
            stats_text = f"总字符数: {total_chars}\n总映射数: {total_mappings}\n最多数值字符: {max_values_char}({max_values_count})"
            self.char_mapping_stats_var.set(stats_text)
        
        messagebox.showinfo("成功", f"成功生成字符映射表，共 {len(self.char_mapping)} 个字符！")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="选择文档文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def extract_sentences(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("错误", "请选择有效的文件路径")
            return
        
        # 在后台线程中执行提取，避免界面卡顿
        self.progress.start()
        thread = threading.Thread(target=self._extract_sentences_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _extract_sentences_thread(self, file_path):
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 解析文档
            sentences = self.parse_document(content)
            
            # 在主线程中更新UI
            self.root.after(0, self._update_extract_results, sentences)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"读取文件时出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _update_extract_results(self, sentences):
        self.sentences = sentences
        
        # 清空现有数据
        for item in self.extract_tree.get_children():
            self.extract_tree.delete(item)
        
        # 添加新数据
        for seq, sentence in sentences:
            length = len(sentence)
            self.extract_tree.insert('', 'end', values=(seq, sentence, length))
        
        # 更新统计信息
        if sentences:
            df = pd.DataFrame(sentences, columns=['序号', '句子'])
            df['长度'] = df['句子'].str.len()
            
            stats_text = f"总句子数: {len(df)}\n平均长度: {df['长度'].mean():.1f}\n最长句子: {df['长度'].max()}\n最短句子: {df['长度'].min()}"
            self.stats_text.set(stats_text)
            
            messagebox.showinfo("成功", f"成功提取 {len(sentences)} 个句子！")
        else:
            messagebox.showwarning("警告", "没有找到有效的句子")
    
    def generate_analysis(self):
        if not self.sentences:
            messagebox.showwarning("警告", "请先提取句子")
            return
        
        try:
            start_seq = int(self.start_seq_var.get())
            end_seq = int(self.end_seq_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的序号")
            return
        
        # 在后台线程中执行分析
        self.progress.start()
        thread = threading.Thread(target=self._generate_analysis_thread, args=(start_seq, end_seq))
        thread.daemon = True
        thread.start()
    
    def _generate_analysis_thread(self, start_seq, end_seq):
        try:
            # 过滤句子
            filtered_sentences = [s for s in self.sentences if start_seq <= s[0] <= end_seq]
            
            if not filtered_sentences:
                self.root.after(0, lambda: messagebox.showerror("错误", "没有找到符合条件的句子"))
                return
            
            # 生成分析结果
            analysis_data = []
            
            for seq, sentence in filtered_sentences:
                length = len(sentence)
                char_values = self.assign_values(sentence)
                
                # 构建字符组合字符串
                char_combination = ' + '.join([f"'{char}'" for char, _ in char_values])
                
                # 构建数值字符串
                values_str = ' + '.join([f"{val}" for _, val in char_values])
                
                # 验证总和
                total = sum(val for _, val in char_values)
                
                analysis_data.append({
                    '序号': seq,
                    '句子': sentence,
                    '长度': length,
                    '字符组合': char_combination,
                    '数值': values_str,
                    '总和验证': total
                })
            
            # 在主线程中更新UI
            self.root.after(0, self._update_analysis_results, analysis_data, filtered_sentences)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _update_analysis_results(self, analysis_data, filtered_sentences):
        self.analysis_data = analysis_data
        
        # 清空现有数据
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
        
        # 添加新数据
        for data in analysis_data:
            self.analysis_tree.insert('', 'end', values=(
                data['序号'], data['句子'], data['长度'],
                data['字符组合'], data['数值'], data['总和验证']
            ))
        
        # 更新示例标签页
        if filtered_sentences:
            self._update_example_tab(filtered_sentences[0])
        
        messagebox.showinfo("成功", f"成功分析 {len(analysis_data)} 个句子！")
    
    def _update_example_tab(self, example_sentence):
        seq, sentence = example_sentence
        char_values = self.assign_values(sentence)
        
        # 更新示例信息
        info_text = f"示例句子 {seq}: {sentence}\n长度: {len(sentence)} 个字符"
        self.example_info_var.set(info_text)
        
        # 清空现有数据
        for item in self.example_tree.get_children():
            self.example_tree.delete(item)
        
        # 添加新数据
        for char, value in char_values:
            self.example_tree.insert('', 'end', values=(char, value))
    
    def generate_words_analysis(self):
        if not self.sentences:
            messagebox.showwarning("警告", "请先提取句子")
            return
        
        try:
            start_seq = int(self.words_start_seq_var.get())
            end_seq = int(self.words_end_seq_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的序号")
            return
        
        # 在后台线程中执行分词分析
        self.progress.start()
        thread = threading.Thread(target=self._generate_words_analysis_thread, args=(start_seq, end_seq))
        thread.daemon = True
        thread.start()
    
    def _generate_words_analysis_thread(self, start_seq, end_seq):
        try:
            # 过滤句子
            filtered_sentences = [s for s in self.sentences if start_seq <= s[0] <= end_seq]
            
            if not filtered_sentences:
                self.root.after(0, lambda: messagebox.showerror("错误", "没有找到符合条件的句子"))
                return
            
            # 使用简单的分词方法（按空格和标点分割）
            words_data = []
            word_freq = {}
            word_sources = {}
            
            for seq, sentence in filtered_sentences:
                # 使用简单的分词：按非文字字符分割
                words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', sentence)
                
                for word in words:
                    if len(word) > 1:  # 只保留长度大于1的词语
                        # 更新词频
                        word_freq[word] = word_freq.get(word, 0) + 1
                        
                        # 记录词语来源
                        if word not in word_sources:
                            word_sources[word] = []
                        if seq not in word_sources[word]:
                            word_sources[word].append(seq)
            
            # 构建词语数据
            word_id = 1
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
                sources = word_sources[word]
                source_text = f"句子: {', '.join(map(str, sources[:3]))}" + ("..." if len(sources) > 3 else "")
                
                # 计算词语的字符长度
                char_length = len(word)
                
                words_data.append({
                    '序号': word_id,
                    '词语': word,
                    '字符长度': char_length,
                    '词频': freq,
                    '句子来源': source_text
                })
                word_id += 1
            
            # 在主线程中更新UI
            self.root.after(0, self._update_words_analysis_results, words_data, filtered_sentences)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"分词分析过程中出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _update_words_analysis_results(self, words_data, filtered_sentences):
        self.words_data = words_data
        
        # 清空现有数据
        for item in self.words_tree.get_children():
            self.words_tree.delete(item)
        
        # 添加新数据
        for data in words_data:
            self.words_tree.insert('', 'end', values=(
                data['序号'], 
                data['词语'], 
                data['字符长度'],
                data['词频'], 
                data['句子来源']
            ))
        
        # 更新词语统计信息
        if words_data:
            total_words = sum(data['词频'] for data in words_data)
            unique_words = len(words_data)
            avg_length = sum(data['字符长度'] for data in words_data) / unique_words
            
            # 添加字符长度分布统计
            length_distribution = {}
            for data in words_data:
                length = data['字符长度']
                length_distribution[length] = length_distribution.get(length, 0) + 1
            
            # 找出最常见的长度
            most_common_length = max(length_distribution.items(), key=lambda x: x[1])[0] if length_distribution else 0
            
            stats_text = f"总词语数: {total_words}\n唯一词语数: {unique_words}\n平均词语长度: {avg_length:.1f}\n最常见长度: {most_common_length}字符"
            self.words_stats_var.set(stats_text)
        
        messagebox.showinfo("成功", f"成功分析 {len(filtered_sentences)} 个句子，提取 {len(words_data)} 个唯一词语！")
    
    def generate_words_value_analysis(self):
        if not self.sentences:
            messagebox.showwarning("警告", "请先提取句子")
            return
        
        try:
            start_seq = int(self.words_value_start_seq_var.get())
            end_seq = int(self.words_value_end_seq_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的序号")
            return
        
        # 在后台线程中执行词语数值分配分析
        self.progress.start()
        thread = threading.Thread(target=self._generate_words_value_analysis_thread, args=(start_seq, end_seq))
        thread.daemon = True
        thread.start()
    
    def _generate_words_value_analysis_thread(self, start_seq, end_seq):
        try:
            # 过滤句子
            filtered_sentences = [s for s in self.sentences if start_seq <= s[0] <= end_seq]
            
            if not filtered_sentences:
                self.root.after(0, lambda: messagebox.showerror("错误", "没有找到符合条件的句子"))
                return
            
            # 生成词语数值分配结果
            words_value_data = []
            
            for seq, sentence in filtered_sentences:
                # 使用分词将句子分成词语
                words = self.tokenize_sentence(sentence)
                
                # 对词语列表应用数值分配规则
                word_values = self.assign_values_to_list(words)
                
                # 构建词语组合字符串
                word_combination = ' + '.join([f"'{word}'" for word, _ in word_values])
                
                # 构建数值字符串
                values_str = ' + '.join([f"{val}" for _, val in word_values])
                
                # 验证总和
                total = sum(val for _, val in word_values)
                
                words_value_data.append({
                    '序号': seq,
                    '句子': sentence,
                    '词语数量': len(words),
                    '词语组合': word_combination,
                    '数值': values_str,
                    '总和验证': total
                })
            
            # 在主线程中更新UI
            self.root.after(0, self._update_words_value_analysis_results, words_value_data, filtered_sentences)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"词语数值分配分析过程中出错: {e}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def _update_words_value_analysis_results(self, words_value_data, filtered_sentences):
        self.words_value_data = words_value_data
        
        # 清空现有数据
        for item in self.words_value_tree.get_children():
            self.words_value_tree.delete(item)
        
        # 添加新数据
        for data in words_value_data:
            self.words_value_tree.insert('', 'end', values=(
                data['序号'], 
                data['句子'], 
                data['词语数量'],
                data['词语组合'], 
                data['数值'], 
                data['总和验证']
            ))
        
        messagebox.showinfo("成功", f"成功分析 {len(words_value_data)} 个句子的词语数值分配！")
    
    def tokenize_sentence(self, sentence):
        """将句子分成词语"""
        # 使用简单的分词：按非文字字符分割
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\s\w]', sentence)
        return words
    
    def assign_values_to_list(self, items):
        """为列表中的项目分配数值（使用与字符组合相同的规则）"""
        n = len(items)
        
        if n == 0:
            return []
        
        if n == 1:
            # 单个项目，值=100
            return [(items[0], 100)]
        
        if n == 2:
            # 两个项目，第一个=0，第二个=100
            return [(items[0], 0), (items[1], 100)]
        
        # n >= 3 的情况
        # 生成递增的差值 - 修改为完全随机
        deltas = [random.randint(1, 5) for _ in range(n - 1)]  # 扩大随机范围
        
        # 构建数值序列
        values = [0]  # 第一个项目总是0
        for d in deltas:
            values.append(values[-1] + d)
        
        # 计算当前总和
        current_sum = sum(values)
        remaining = 100 - current_sum
        
        # 将剩余值加到最后一个项目
        values[-1] += remaining
        
        return list(zip(items, values))
    
    def save_analysis(self):
        if not self.analysis_data:
            messagebox.showwarning("警告", "没有可保存的分析数据")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.analysis_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"分析结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {e}")
    
    def save_words_analysis(self):
        if not self.words_data:
            messagebox.showwarning("警告", "没有可保存的词语数据")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存词语分析结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.words_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"词语分析结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {e}")
    
    def save_words_value_analysis(self):
        if not hasattr(self, 'words_value_data') or not self.words_value_data:
            messagebox.showwarning("警告", "没有可保存的词语数值分配数据")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存词语数值分配结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.words_value_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"词语数值分配结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {e}")
    
    def save_char_mapping(self):
        """保存字符映射表"""
        if not self.char_mapping:
            messagebox.showwarning("警告", "没有可保存的字符映射表数据")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存字符映射表",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # 构建数据框
                data = []
                char_id = 1
                for char, values in sorted(self.char_mapping.items()):
                    values_str = '/'.join(map(str, sorted(values)))
                    count = len(values)
                    data.append({
                        '序号': char_id,
                        '字符': char,
                        '数值列表': values_str,
                        '出现次数': count
                    })
                    char_id += 1
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"字符映射表已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {e}")
    
    # 原有的解析和分配数值函数
    def parse_document(self, content: str) -> List[Tuple[int, str]]:
        """解析文档，提取序号和句子"""
        sentences = []
        lines = content.strip().split('\n')
        
        for line in lines:
            # 匹配 "序号. 句子" 格式
            match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
            if match:
                seq_num = int(match.group(1))
                sentence = match.group(2).strip()
                # 过滤掉空句子和特殊标记行
                if sentence and not sentence.startswith('生成时间:') and not sentence.startswith('#'):
                    sentences.append((seq_num, sentence))
        
        return sentences
    
    def assign_values(self, s: str) -> List[Tuple[str, int]]:
        """为句子中的每个字符分配数值 - 严格按照您定义的规则"""
        n = len(s)
        
        if n == 0:
            return []
        
        if n == 1:
            # 单个字符，值=100
            return [(s[0], 100)]
        
        if n == 2:
            # 两个字符，第一个=0，第二个=100
            return [(s[0], 0), (s[1], 100)]
        
        # n >= 3 的情况
        # 生成递增的差值
        deltas = [random.randint(1, 5) for _ in range(n - 1)]
        
        # 构建数值序列
        values = [0]  # 第一个字符总是0
        for d in deltas:
            values.append(values[-1] + d)
        
        # 计算当前总和
        current_sum = sum(values)
        remaining = 100 - current_sum
        
        # 将剩余值加到最后一个字符
        values[-1] += remaining
        
        return list(zip(s, values))
    
    # 图表相关方法
    def generate_char_chart(self):
        """生成字符数值图表"""
        if not self.sentences:
            messagebox.showwarning("警告", "请先提取句子")
            return
        
        try:
            seq = int(self.chart_seq_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的句子序号")
            return
        
        # 查找指定序号的句子
        target_sentence = None
        for s in self.sentences:
            if s[0] == seq:
                target_sentence = s
                break
        
        if not target_sentence:
            messagebox.showerror("错误", f"未找到序号为 {seq} 的句子")
            return
        
        # 生成字符数值分配
        char_values = self.assign_values(target_sentence[1])
        
        # 创建图表
        self._create_char_value_chart(target_sentence, char_values)
    
    def _create_char_value_chart(self, sentence, char_values):
        """创建字符数值图表"""
        # 清除现有图表
        self._clear_chart()
        
        # 创建plotly图表
        seq, text = sentence
        chars = [cv[0] for cv in char_values]
        values = [cv[1] for cv in char_values]
        
        fig = go.Figure(data=[
            go.Bar(x=chars, y=values, marker_color='skyblue')
        ])
        
        fig.update_layout(
            title=dict(
                text=f'句子 {seq} 字符数值分布: "{text}"',
                font=dict(size=14)
            ),
            xaxis_title='字符',
            yaxis_title='数值',
            showlegend=False
        )
        
        # 保存为HTML文件并在浏览器中打开
        html_path = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
        pyo.plot(fig, filename=html_path, auto_open=True)
        
        self.chart_status_var.set("图表已在浏览器中打开")
    
    def generate_length_chart(self):
        """生成句子长度分布图表"""
        if not self.sentences:
            messagebox.showwarning("警告", "请先提取句子")
            return
        
        # 计算句子长度分布
        lengths = [len(s[1]) for s in self.sentences]
        
        # 创建图表
        self._create_length_distribution_chart(lengths)
    
    def _create_length_distribution_chart(self, lengths):
        """创建句子长度分布图表"""
        # 计算频率
        length_counts = {}
        for length in lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 排序
        sorted_lengths = sorted(length_counts.keys())
        counts = [length_counts[length] for length in sorted_lengths]
        
        # 创建plotly图表
        fig = go.Figure(data=[
            go.Bar(x=sorted_lengths, y=counts, marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title=dict(
                text='句子长度分布',
                font=dict(size=14)
            ),
            xaxis_title='句子长度',
            yaxis_title='句子数量',
            showlegend=False
        )
        
        # 保存为HTML文件并在浏览器中打开
        html_path = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
        pyo.plot(fig, filename=html_path, auto_open=True)
        
        self.chart_status_var.set("图表已在浏览器中打开")
    
    def generate_word_freq_chart(self):
        """生成词语频率图表"""
        if not hasattr(self, 'words_data') or not self.words_data:
            messagebox.showwarning("警告", "请先进行词语分析")
            return
        
        # 获取前20个最频繁的词语
        top_words = self.words_data[:20]
        
        # 创建图表
        self._create_word_frequency_chart(top_words)
    
    def _create_word_frequency_chart(self, top_words):
        """创建词语频率图表"""
        words = [w['词语'] for w in top_words]
        freqs = [w['词频'] for w in top_words]
        
        # 创建plotly图表
        fig = go.Figure(data=[
            go.Bar(y=words, x=freqs, orientation='h', marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title=dict(
                text='Top 20 词语频率分布',
                font=dict(size=14)
            ),
            xaxis_title='频率',
            yaxis_title='词语',
            showlegend=False
        )
        
        # 保存为HTML文件并在浏览器中打开
        html_path = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
        pyo.plot(fig, filename=html_path, auto_open=True)
        
        self.chart_status_var.set("图表已在浏览器中打开")
    
    def _clear_chart(self):
        """清除当前图表"""
        # 对于plotly，我们不需要清除，因为每次都在新文件中生成
        pass
    
    def save_chart(self):
        """保存当前图表"""
        messagebox.showinfo("提示", "图表已自动保存为HTML文件并在浏览器中打开，您可以在浏览器中直接保存图表")

def main():
    root = tk.Tk()
    app = DocumentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()