import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from experiments.full_system.TriLLMage import FullSystemVQAXExperiment
from experiments.ablation_studies.Junior import JuniorVQAXExperiment
from experiments.ablation_studies.Senior import SeniorVQAXExperiment
from experiments.ablation_studies.Manager import ManagerVQAXExperiment

def main():
    parser = argparse.ArgumentParser(description="Visual Multi-Agent Knowledge QA System")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["full_system", "ablation_no_judge", "junior", "senior", "manager"],
                       help="Experiment to run")

    parser.add_argument("--samples", type=int, default=300,
                       help="Number of samples to process (0 for full dataset)")
    
    args = parser.parse_args()
    
    # Select and initialize the experiment
    if args.experiment == "full_system":
        experiment = FullSystemVQAXExperiment(args.samples)
    elif args.experiment == "junior":
        experiment = JuniorVQAXExperiment(args.samples)
    elif args.experiment == "senior":
        experiment = SeniorVQAXExperiment(args.samples)
    elif args.experiment == "manager":
        experiment = ManagerVQAXExperiment(args.samples)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # Run experiment
    print(f"Starting {experiment.experiment_name}...")
    results = experiment.run()
    print(f"{experiment.experiment_name} completed!")
    
    return results

if __name__ == "__main__":
    main()

