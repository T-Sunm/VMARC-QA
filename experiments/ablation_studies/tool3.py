from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia, llm_knowledge
from src.tools.vqa_tool import vqa_tool

class Tool3VQAXExperiment(BaseExperiment):
    def __init__(self, sample_size: int = None):
        self.experiment_name = "tool3"
        super().__init__(sample_size)
    
    def setup_system(self):
        """Setup complete multi-agent system"""
        tools_registry = {
            "vqa_tool": vqa_tool,
            "arxiv": arxiv,
            "wikipedia": wikipedia,
            "llm_knowledge": llm_knowledge,
        }
        
        builder = MainGraphBuilder(tools_registry)
        return builder.create_main_workflow()