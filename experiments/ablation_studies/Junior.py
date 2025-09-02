from experiments.base_experiment import BaseExperiment
from src.core.graph_builder.junior_graph import JuniorGraphBuilder
from src.tools.vqa_tool import vqa_tool

class JuniorVQAXExperiment(BaseExperiment):
    def __init__(self, sample_size: int = None):
        self.experiment_name = "junior"
        super().__init__(sample_size)
    
    def setup_system(self):
        """Setup junior agent system with only VQA tool"""
        tools_registry = {
            "vqa_tool": vqa_tool,
        }
        
        builder = JuniorGraphBuilder(tools_registry)
        return builder.create_junior_workflow()

if __name__ == "__main__":
    # Run experiment with sample size for testing (set to None for full dataset)
    experiment = JuniorVQAXExperiment(sample_size=10)
    results = experiment.run()
    print(f"Experiment completed. Results saved to {experiment.results_dir}")