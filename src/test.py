from rag_utils import PDFProcessor
# 示例使用
# 创建PDF处理器实例
processor = PDFProcessor('/data/jiangziyu/data/chatglm_llm_fintech_raw_dataset/small_portion', '/data/jiangziyu/data/chatglm_llm_fintech_raw_dataset/small_portion_processed')
# 处理所有PDF文件
processor.process_all_pdfs()