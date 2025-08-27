# experiments/base_experiment.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import json
import os
import time
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch

from src.evaluation.metrics_x import VQAXEvaluator
from src.utils.text_processing import normalize_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, config_path: str, sample_size: int = None):
        self.config_path = config_path
        self.sample_size = sample_size
        self.evaluator = VQAXEvaluator()
        self.experiment_name = self.__class__.__name__
        
        # Create results directory
        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load and prepare data
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        """Load ViVQA-X dataset"""
        json_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json"
        coco_img_dir = "/mnt/VLAI_data/COCO_Images/val2014/"
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            img_path = os.path.join(coco_img_dir, item["image_name"])
            image = Image.open(img_path).convert("RGB")
            sample = {
                "question": item["question"],
                "image": image,
                "image_path": img_path,
                "explanation": item["explanation"],
                "answer": item["answer"],
                "question_id": item["question_id"]
            }
            samples.append(sample)
        
        # Limit samples if specified
        if self.sample_size:
            samples = samples[:self.sample_size]
            
        return samples
    
    @abstractmethod
    def setup_system(self):
        """Setup the system/graph for the experiment"""
        pass
    
    def run_single_sample(self, graph, sample: Dict) -> Dict:
        """Run inference on a single sample"""
        try:
            initial_state = {"question": sample["question"], "image": sample["image"]}
            result = graph.invoke(initial_state)
            
            return {
                "question": sample["question"],
                "image_caption": result.get('image_caption', ''),
                "rationales": result.get('rationales', []),
                "final_answer": result.get("final_answer", ""),
                "explanation": result.get("explanation", ""),
                "gold_answer": sample["answer"],
                "gold_explanation": sample["explanation"],
                "question_id": sample["question_id"],
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error processing sample {sample['question_id']}: {e}")
            return {
                "question": sample["question"],
                "question_id": sample["question_id"],
                "success": False,
                "error": str(e),
                "final_answer": "",
                "explanation": "",
                "gold_answer": sample["answer"],
                "gold_explanation": sample["explanation"]
            }
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        print(f"Starting {self.experiment_name}")
        
        # Setup system
        graph = self.setup_system()
        
        # Process all samples
        results = []
        successful_samples = 0
        
        for i, sample in enumerate(tqdm(self.data, desc=f"Processing {self.experiment_name}")):
            print(f"\n--- Sample {i+1}/{len(self.data)} (ID: {sample['question_id']}) ---")
            
            start_time = time.time()
            result = self.run_single_sample(graph, sample)
            end_time = time.time()
            
            result["processing_time"] = end_time - start_time
            results.append(result)
            
            if result["success"]:
                successful_samples += 1
                print("✅ Sample processed successfully")
            else:
                print(f"❌ Sample failed: {result['error']}")
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        # Prepare final results
        final_results = {
            "experiment_name": self.experiment_name,
            "num_samples": len(self.data),
            "successful_samples": successful_samples,
            "failed_samples": len(self.data) - successful_samples,
            "metrics": metrics,
            "detailed_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        self.save_results(final_results)
        
        # Clean up memory
        self.cleanup(graph)
        
        return final_results
    
    def compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        try:
            # Extract successful results only
            successful_results = [r for r in results if r["success"]]
            
            if not successful_results:
                return {"error": "No successful samples to evaluate"}
            
            # Prepare answers
            predicted_answers = [normalize_answer(r["final_answer"]) for r in successful_results]
            ground_truth_answers = [normalize_answer(r["gold_answer"]) for r in successful_results]
            
            # Create vocabulary for answers
            all_answers = sorted(list(set(predicted_answers + ground_truth_answers)))
            answer_to_idx = {ans: i for i, ans in enumerate(all_answers)}
            
            predicted_answer_indices = [answer_to_idx[ans] for ans in predicted_answers]
            ground_truth_answer_indices = [answer_to_idx[ans] for ans in ground_truth_answers]
            
            # Prepare explanations
            predicted_explanations = {
                str(r["question_id"]): [r["explanation"]] 
                for r in successful_results
            }
            ground_truth_explanations = {
                str(r["question_id"]): r["gold_explanation"] 
                for r in successful_results
            }
            
            # Compute metrics
            answer_metrics = self.evaluator.compute_answer_metrics(
                predicted_answer_indices, ground_truth_answer_indices
            )
            
            explanation_metrics = self.evaluator.compute_explanation_metrics(
                predicted_explanations, ground_truth_explanations
            )
            
            return {**answer_metrics, **explanation_metrics}
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"error": f"Failed to compute metrics: {e}"}
    
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\nResults saved to {filepath}")
        
        # Print summary
        if "metrics" in results and isinstance(results["metrics"], dict) and "error" not in results["metrics"]:
            print("\n--- Evaluation Results ---")
            for metric, value in results["metrics"].items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print("Metrics computation failed - check error details in output file")
    
    def cleanup(self, graph):
        """Clean up memory"""
        print("\nReleasing models from memory...")
        del graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory released.")
