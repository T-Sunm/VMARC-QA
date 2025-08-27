# experiments/full_system/run_vivqa_x.py

from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge, dam_caption_image_tool

class FullSystemVQAXExperiment(BaseExperiment):
    def __init__(self, config_path: str, sample_size: int = None):
        super().__init__(config_path, sample_size)
        self.experiment_name = "FullSystemVQAX"
    
    def setup_system(self):
        """Setup complete multi-agent system"""
        tools_registry = {
            "vqa_tool": vqa_tool,
            "arxiv": arxiv,
            "wikipedia": wikipedia,
            "lm_knowledge": lm_knowledge,
            "analyze_image_object": dam_caption_image_tool,
        }
        
        builder = MainGraphBuilder(tools_registry)
        return builder.create_main_workflow()