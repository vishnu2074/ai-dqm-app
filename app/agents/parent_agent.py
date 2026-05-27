# app/agents/parent_agent.py
 
import os
 
class ParentAgent:
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
 
    def run(self):
        tables = []
        for path in self.dataset_paths:
           
           
            tables.append(path)
 
        return tables