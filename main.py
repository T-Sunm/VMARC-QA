import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from experiments.full_system.TriLLMage import FullSystemVQAXExperiment

def main():
    parser = argparse.ArgumentParser(description="Visual Multi-Agent Knowledge QA System")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["full_system", "ablation_no_judge"],
                       help="Experiment to run")

    parser.add_argument("--samples", type=int, default=300,
                       help="Number of samples to process")
    
    parser.add_argument("--test_json_path", type=str, 
                        default="data/ViVQA-X/ViVQA-X_test.json",
                        help="Path to the ViVQA-X test JSON file. Example: /mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json")
    parser.add_argument("--test_image_dir", type=str,
                        default="data/COCO_Images/val2014/",
                        help="Path to the COCO validation images directory. Example: /mnt/VLAI_data/COCO_Images/val2014/")
    
    args = parser.parse_args()
    
    # Directly run the selected experiment
    if args.experiment == "full_system":
        experiment = FullSystemVQAXExperiment(
            sample_size=args.samples,
            test_json_path=args.test_json_path,
            test_image_dir=args.test_image_dir
        )
    
    # Run experiment
    print(f"Starting {experiment.experiment_name}...")
    results = experiment.run()
    print(f"{experiment.experiment_name} completed!")
    
    return results

if __name__ == "__main__":
    main()

