
import os
import torch
import model
class Test():
    def __init__(self, loader_test, file_R):
        self.loader_test = loader_test
        self.root_M = file_R
        self.fileC = torch.load(self.root_M)
        self.model = self.fileC['model']


    def test_all(self,rand):
        



if name == '__test__':
    