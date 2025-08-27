# main.py - Even simpler
import argparse
from experiments.full_system.TriLLMage import FullSystemVQAXExperiment

def main():
    parser = argparse.ArgumentParser(description="Visual Multi-Agent Knowledge QA System")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["full_system", "ablation_no_judge"],
                       help="Experiment to run")
                       
    parser.add_argument("--samples", type=int, default=300,
                       help="Number of samples to process")
    
    args = parser.parse_args()
    
    # Directly run the selected experiment
    if args.experiment == "full_system":
        experiment = FullSystemVQAXExperiment(args.config, args.samples)
    
    # Run experiment
    print(f"Starting {experiment.experiment_name}...")
    results = experiment.run()
    print(f"{experiment.experiment_name} completed!")
    
    return results

if __name__ == "__main__":
    main()
