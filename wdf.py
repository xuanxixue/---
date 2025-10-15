import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import docx
import re
import json
import os

class DocumentProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("文档句子提取工具")
        self.root.geometry("800x600")
        
        # 存储上传的文件内容
        self.uploaded_files_content = ""
        self.unique_sentences = []  # 存储处理后的句子
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="文档句子提取工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 上传按钮区域
        upload_frame = ttk.LabelFrame(main_frame, text="文档上传", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.upload_btn = ttk.Button(upload_frame, text="上传文档 (docx/txt)", command=self.upload_files)
        self.upload_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.file_list_label = ttk.Label(upload_frame, text="未选择文件")
        self.file_list_label.grid(row=0, column=1, sticky=tk.W)
        
        # 处理按钮
        self.process_btn = ttk.Button(main_frame, text="处理文档", command=self.process_documents, state="disabled")
        self.process_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 导出按钮区域
        export_frame = ttk.LabelFrame(main_frame, text="导出选项", padding="10")
        export_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.export_txt_btn = ttk.Button(export_frame, text="导出为TXT", command=self.export_txt, state="disabled")
        self.export_txt_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.export_json_btn = ttk.Button(export_frame, text="导出为JSON", command=self.export_json, state="disabled")
        self.export_json_btn.grid(row=0, column=1)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="处理结果", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 统计信息
        self.stats_label = ttk.Label(result_frame, text="请先上传并处理文档")
        self.stats_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # 句子显示区域
        self.result_text = scrolledtext.ScrolledText(result_frame, width=80, height=20, wrap=tk.WORD)
        self.result_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
    
    def upload_files(self):
        """上传多个文档文件"""
        file_types = [
            ("Word文档", "*.docx"),
            ("文本文件", "*.txt"),
            ("所有文件", "*.*")
        ]
        
        file_paths = filedialog.askopenfilenames(
            title="选择文档文件",
            filetypes=file_types
        )
        
        if file_paths:
            self.file_list_label.config(text=f"已选择 {len(file_paths)} 个文件")
            self.process_btn.config(state="normal")
            self.file_paths = file_paths
        else:
            self.file_list_label.config(text="未选择文件")
            self.process_btn.config(state="disabled")
    
    def read_document(self, file_path):
        """读取文档内容"""
        try:
            if file_path.endswith('.docx'):
                # 读取Word文档
                doc = docx.Document(file_path)
                content = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        content.append(paragraph.text)
                return '\n'.join(content)
            elif file_path.endswith('.txt'):
                # 读取文本文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                messagebox.showerror("错误", f"不支持的文件格式: {file_path}")
                return ""
        except Exception as e:
            messagebox.showerror("错误", f"读取文件 {file_path} 时出错: {str(e)}")
            return ""
    
    def merge_documents(self):
        """合并所有上传的文档为无段落文档"""
        combined_content = []
        for file_path in self.file_paths:
            content = self.read_document(file_path)
            if content:
                # 移除多余的空白字符，但保留基本的空格分隔
                cleaned_content = re.sub(r'\s+', ' ', content).strip()
                combined_content.append(cleaned_content)
        
        self.uploaded_files_content = ' '.join(combined_content)
        return self.uploaded_files_content
    
    def extract_sentences(self, text):
        """简单但有效的中文句子分割算法"""
        sentences = []
        
        # 使用正则表达式分割句子
        # 主要分割符：。！？；以及对应的英文标点
        # 保留分割符
        pattern = r'([。！？；\.!?;])'
        parts = re.split(pattern, text)
        
        # 重新组合句子
        current_sentence = ""
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # 如果是分割符
            if i + 1 < len(parts) and parts[i+1] in ['。', '！', '？', '；', '.', '!', '?', ';']:
                current_sentence += part + parts[i+1]
                sentences.append(current_sentence.strip())
                current_sentence = ""
                i += 2
            else:
                current_sentence += part
                i += 1
        
        # 处理最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 过滤空句子
        sentences = [s for s in sentences if s.strip()]
        
        return sentences
    
    def remove_duplicate_sentences(self, sentences):
        """去除完全相同的句子，但保留有细微差别的句子"""
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return unique_sentences
    
    def process_documents(self):
        """处理文档并显示结果"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            messagebox.showwarning("警告", "请先上传文档文件")
            return
        
        try:
            # 合并文档
            self.merge_documents()
            
            if not self.uploaded_files_content:
                messagebox.showwarning("警告", "没有可处理的内容")
                return
            
            # 提取句子
            sentences = self.extract_sentences(self.uploaded_files_content)
            
            # 去除完全相同的句子
            self.unique_sentences = self.remove_duplicate_sentences(sentences)
            
            # 启用导出按钮
            self.export_txt_btn.config(state="normal")
            self.export_json_btn.config(state="normal")
            
            # 显示结果
            self.display_results(sentences, self.unique_sentences)
            
        except Exception as e:
            messagebox.showerror("错误", f"处理文档时出错: {str(e)}")
    
    def display_results(self, original_sentences, unique_sentences):
        """显示处理结果"""
        # 更新统计信息
        stats_text = f"原始句子数量: {len(original_sentences)} | 去重后句子数量: {len(unique_sentences)}"
        self.stats_label.config(text=stats_text)
        
        # 显示提取的句子
        self.result_text.delete(1.0, tk.END)
        
        if unique_sentences:
            self.result_text.insert(tk.END, "提取的句子:\n\n")
            for i, sentence in enumerate(unique_sentences, 1):
                self.result_text.insert(tk.END, f"{i}. {sentence}\n\n")
        else:
            self.result_text.insert(tk.END, "未提取到任何句子")
    
    def export_txt(self):
        """导出结果为TXT文件"""
        if not self.unique_sentences:
            messagebox.showwarning("警告", "没有可导出的句子，请先处理文档")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存为TXT文件",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f"文档句子提取结果\n")
                    file.write(f"原始文档数量: {len(self.file_paths)}\n")
                    file.write(f"提取句子数量: {len(self.unique_sentences)}\n")
                    file.write("=" * 50 + "\n\n")
                    
                    for i, sentence in enumerate(self.unique_sentences, 1):
                        file.write(f"{i}. {sentence}\n\n")
                
                messagebox.showinfo("成功", f"句子已成功导出到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出TXT文件时出错: {str(e)}")
    
    def export_json(self):
        """导出结果为JSON文件"""
        if not self.unique_sentences:
            messagebox.showwarning("警告", "没有可导出的句子，请先处理文档")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存为JSON文件",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 准备导出数据
                export_data = {
                    "metadata": {
                        "source_files": self.file_paths,
                        "total_sentences": len(self.unique_sentences),
                        "export_timestamp": str(os.path.getctime(self.file_paths[0]) if self.file_paths else "")
                    },
                    "sentences": [
                        {"id": i, "content": sentence} 
                        for i, sentence in enumerate(self.unique_sentences, 1)
                    ]
                }
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(export_data, file, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("成功", f"句子已成功导出到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出JSON文件时出错: {str(e)}")

def main():
    root = tk.Tk()
    app = DocumentProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()