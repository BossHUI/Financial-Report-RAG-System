# -*- coding: utf-8 -*-

import unittest
import torch,numpy
from src.embeddings import EmbeddingProcessor
from src.config import EMBEDDING_MODEL_NAME

class TestEmbeddingProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = EmbeddingProcessor(EMBEDDING_MODEL_NAME)
        self.test_text = "这是一个测试文档。"

    def test_batch_get_embeddings(self):
        from torch.utils.data import DataLoader
        dataloader = DataLoader([self.test_text], batch_size=1)
        embeddings = self.processor.batch_get_embeddings(dataloader)
        
        self.assertIsNotNone(embeddings)
        self.assertTrue(len(embeddings) > 0)
        self.assertTrue(isinstance(embeddings[0], numpy.ndarray))

    def test_process_pdf_files(self):
        # 测试PDF处理功能
        pass

if __name__ == '__main__':
    unittest.main()