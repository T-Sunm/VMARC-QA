from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.senior_graph import SeniorGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool

class SeniorVQAXExperiment(BaseExperiment):
    def __init__(self, sample_size: int = None):
        self.experiment_name = "senior"
        super().__init__(sample_size)
    
    def setup_system(self):
        """Setup senior agent system with VQA tool and knowledge base tools"""
        tools_registry = {
            "vqa_tool": vqa_tool,
            "arxiv": arxiv,
            "wikipedia": wikipedia,
        }
        
        builder = SeniorGraphBuilder(tools_registry)
        return builder.create_senior_workflow()

if __name__ == "__main__":
    # Run experiment with sample size for testing (set to None for full dataset)
    experiment = SeniorVQAXExperiment(sample_size=10)
    results = experiment.run()
    print(f"Experiment completed. Results saved to {experiment.results_dir}")