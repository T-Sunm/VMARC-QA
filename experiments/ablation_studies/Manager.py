from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.manager_graph import ManagerGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge, dam_caption_image_tool

class ManagerVQAXExperiment(BaseExperiment):
    def __init__(self, sample_size: int = None):
        self.experiment_name = "manager"
        super().__init__(sample_size)
    
    def setup_system(self):
        """Setup manager agent system with VQA tool, knowledge base tools, and object analysis"""
        tools_registry = {
            "vqa_tool": vqa_tool,
            "arxiv": arxiv,
            "wikipedia": wikipedia,
            "lm_knowledge": lm_knowledge,
            "analyze_image_object": dam_caption_image_tool,
        }
        
        builder = ManagerGraphBuilder(tools_registry)
        return builder.create_manager_workflow()

if __name__ == "__main__":
    # Run experiment with sample size for testing (set to None for full dataset)
    experiment = ManagerVQAXExperiment(sample_size=10)
    results = experiment.run()
    print(f"Experiment completed. Results saved to {experiment.results_dir}")