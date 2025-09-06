from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge, dam_caption_image_tool

class FullSystemVQAXExperiment(BaseExperiment):
    def __init__(self, sample_size: int, test_json_path: str, test_image_dir: str):
        self.experiment_name = "full_system"
        super().__init__(sample_size, test_json_path, test_image_dir)
    
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