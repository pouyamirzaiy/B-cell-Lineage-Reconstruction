#!/usr/bin/env python3
"""
Run Hierarchical Feature-Based Tree Construction on All 48 Datasets
================================================================

This script runs the complete hierarchical analysis on all available datasets
and compares comb resolution with the classical ClonalTree approach.

Key Questions:
1. How many comb nodes exist in each dataset?
2. Can our hierarchical approach resolve more comb nodes than classical ClonalTree?
3. Which features are most effective for comb resolution?
4. What is the overall improvement in tree resolution?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from hierarchical_tree_constructor import HierarchicalTreeConstructor

class AllDatasetsAnalyzer:
    """
    Analyzes all available datasets using the hierarchical feature-based approach
    and compares results with classical ClonalTree.
    """
    
    def __init__(self):
        self.data_path = Path("../data/sarah_kaveh_materials/respearmansimilarityanalysiscomparativereport")
        self.results_path = Path("../results")
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize the tree constructor
        self.tree_constructor = HierarchicalTreeConstructor()
        
        # Results storage
        self.all_results = {}
        self.comparison_summary = {}
        
        print("🔬 ALL DATASETS HIERARCHICAL ANALYSIS")
        print("=" * 50)
        print(f"📁 Data path: {self.data_path}")
        print(f"📁 Results path: {self.results_path}")
        print()
    
    def discover_datasets(self) -> List[str]:
        """Discover all available datasets in the data directory"""
        print("🔍 DISCOVERING DATASETS")
        print("-" * 25)
        
        datasets = []
        
        if not self.data_path.exists():
            print(f"❌ Data path not found: {self.data_path}")
            return datasets
        
        # Look for dataset directories
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name.startswith('dataset_'):
                datasets.append(item.name)
        
        datasets.sort()  # Sort for consistent ordering
        print(f"📊 Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"   • {dataset}")
        print()
        
        return datasets
    
    def analyze_single_dataset(self, dataset_name: str) -> Dict:
        """
        Analyze a single dataset using the hierarchical approach.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'dataset_01')
            
        Returns:
            Dictionary with analysis results
        """
        print(f"🧬 ANALYZING {dataset_name.upper()}")
        print("-" * 40)
        
        dataset_path = self.data_path / dataset_name
        
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return None
        
        try:
            # Load features for this dataset
            features_data = self.tree_constructor.load_features_for_dataset(dataset_path)
            
            if not features_data:
                print(f"❌ No features found for {dataset_name}")
                return None
            
            print(f"📊 Loaded {len(features_data['features'])} features for {len(features_data['proteins'])} proteins")
            
            # Calculate feature informativeness
            feature_rankings = self.tree_constructor.calculate_feature_informativeness(features_data)
            print(f"🔬 Feature rankings calculated")
            
            # Build hierarchical tree
            resolution_log = self.tree_constructor.build_hierarchical_tree(
                features_data, 
                feature_rankings,
                max_iterations=10  # Use all 10 features
            )
            
            # Extract key metrics
            initial_comb_count = resolution_log.get('initial_comb_nodes', 0)
            final_comb_count = resolution_log.get('final_comb_nodes', 0)
            iterations_used = resolution_log.get('iterations_used', 0)
            resolution_success = resolution_log.get('resolution_success', False)
            
            # Calculate improvement
            comb_reduction = initial_comb_count - final_comb_count
            improvement_percentage = (comb_reduction / initial_comb_count * 100) if initial_comb_count > 0 else 0
            
            result = {
                'dataset': dataset_name,
                'proteins_count': len(features_data['proteins']),
                'features_count': len(features_data['features']),
                'initial_comb_nodes': initial_comb_count,
                'final_comb_nodes': final_comb_count,
                'comb_reduction': comb_reduction,
                'improvement_percentage': improvement_percentage,
                'iterations_used': iterations_used,
                'resolution_success': resolution_success,
                'feature_rankings': feature_rankings,
                'resolution_log': resolution_log
            }
            
            print(f"📈 Results for {dataset_name}:")
            print(f"   • Proteins: {result['proteins_count']}")
            print(f"   • Initial comb nodes: {result['initial_comb_nodes']}")
            print(f"   • Final comb nodes: {result['final_comb_nodes']}")
            print(f"   • Comb reduction: {result['comb_reduction']}")
            print(f"   • Improvement: {result['improvement_percentage']:.1f}%")
            print(f"   • Iterations used: {result['iterations_used']}")
            print(f"   • Resolution success: {'✅' if result['resolution_success'] else '❌'}")
            print()
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {dataset_name}: {str(e)}")
            return None
    
    def run_all_datasets_analysis(self) -> Dict:
        """Run the hierarchical analysis on all discovered datasets"""
        print("🚀 RUNNING ALL DATASETS ANALYSIS")
        print("=" * 40)
        
        datasets = self.discover_datasets()
        
        if not datasets:
            print("❌ No datasets found to analyze")
            return {}
        
        start_time = time.time()
        successful_analyses = 0
        failed_analyses = 0
        
        for i, dataset in enumerate(datasets, 1):
            print(f"📊 Progress: {i}/{len(datasets)} datasets")
            
            result = self.analyze_single_dataset(dataset)
            
            if result:
                self.all_results[dataset] = result
                successful_analyses += 1
            else:
                failed_analyses += 1
            
            # Save intermediate results every 10 datasets
            if i % 10 == 0:
                self.save_intermediate_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ ANALYSIS COMPLETE!")
        print(f"   • Total datasets: {len(datasets)}")
        print(f"   • Successful analyses: {successful_analyses}")
        print(f"   • Failed analyses: {failed_analyses}")
        print(f"   • Total time: {total_time:.1f} seconds")
        print(f"   • Average time per dataset: {total_time/len(datasets):.1f} seconds")
        print()
        
        return self.all_results
    
    def generate_comparison_summary(self) -> Dict:
        """Generate a comprehensive comparison summary"""
        print("📊 GENERATING COMPARISON SUMMARY")
        print("-" * 35)
        
        if not self.all_results:
            print("❌ No results to summarize")
            return {}
        
        # Calculate overall statistics
        total_datasets = len(self.all_results)
        total_proteins = sum(r['proteins_count'] for r in self.all_results.values())
        total_initial_combs = sum(r['initial_comb_nodes'] for r in self.all_results.values())
        total_final_combs = sum(r['final_comb_nodes'] for r in self.all_results.values())
        total_comb_reduction = total_initial_combs - total_final_combs
        
        # Calculate success rates
        fully_resolved = sum(1 for r in self.all_results.values() if r['final_comb_nodes'] == 0)
        partially_resolved = sum(1 for r in self.all_results.values() if r['comb_reduction'] > 0)
        no_improvement = sum(1 for r in self.all_results.values() if r['comb_reduction'] == 0)
        
        # Calculate average improvements
        avg_improvement = np.mean([r['improvement_percentage'] for r in self.all_results.values()])
        avg_iterations = np.mean([r['iterations_used'] for r in self.all_results.values()])
        
        # Find best and worst performing datasets
        best_dataset = max(self.all_results.values(), key=lambda x: x['improvement_percentage'])
        worst_dataset = min(self.all_results.values(), key=lambda x: x['improvement_percentage'])
        
        # Feature effectiveness analysis
        feature_effectiveness = {}
        for result in self.all_results.values():
            for feature, ranking in result['feature_rankings'].items():
                if feature not in feature_effectiveness:
                    feature_effectiveness[feature] = []
                feature_effectiveness[feature].append(ranking['informativeness'])
        
        # Calculate average informativeness for each feature
        feature_avg_informativeness = {
            feature: np.mean(scores) for feature, scores in feature_effectiveness.items()
        }
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_datasets': total_datasets,
            'total_proteins': total_proteins,
            'total_initial_comb_nodes': total_initial_combs,
            'total_final_comb_nodes': total_final_combs,
            'total_comb_reduction': total_comb_reduction,
            'overall_improvement_percentage': (total_comb_reduction / total_initial_combs * 100) if total_initial_combs > 0 else 0,
            'success_rates': {
                'fully_resolved_datasets': fully_resolved,
                'partially_resolved_datasets': partially_resolved,
                'no_improvement_datasets': no_improvement,
                'full_resolution_rate': fully_resolved / total_datasets * 100,
                'any_improvement_rate': partially_resolved / total_datasets * 100
            },
            'average_metrics': {
                'improvement_percentage': avg_improvement,
                'iterations_used': avg_iterations
            },
            'best_performing_dataset': {
                'name': best_dataset['dataset'],
                'improvement_percentage': best_dataset['improvement_percentage'],
                'comb_reduction': best_dataset['comb_reduction']
            },
            'worst_performing_dataset': {
                'name': worst_dataset['dataset'],
                'improvement_percentage': worst_dataset['improvement_percentage'],
                'comb_reduction': worst_dataset['comb_reduction']
            },
            'feature_effectiveness': feature_avg_informativeness,
            'dataset_results': self.all_results
        }
        
        self.comparison_summary = summary
        
        print(f"📈 COMPARISON SUMMARY:")
        print(f"   • Total datasets analyzed: {total_datasets}")
        print(f"   • Total proteins: {total_proteins}")
        print(f"   • Total initial comb nodes: {total_initial_combs}")
        print(f"   • Total final comb nodes: {total_final_combs}")
        print(f"   • Total comb reduction: {total_comb_reduction}")
        print(f"   • Overall improvement: {summary['overall_improvement_percentage']:.1f}%")
        print(f"   • Fully resolved datasets: {fully_resolved}/{total_datasets} ({summary['success_rates']['full_resolution_rate']:.1f}%)")
        print(f"   • Any improvement: {partially_resolved}/{total_datasets} ({summary['success_rates']['any_improvement_rate']:.1f}%)")
        print(f"   • Average improvement: {avg_improvement:.1f}%")
        print(f"   • Average iterations: {avg_iterations:.1f}")
        print()
        
        return summary
    
    def save_results(self):
        """Save all results to files"""
        print("💾 SAVING RESULTS")
        print("-" * 20)
        
        # Save detailed results
        results_file = self.results_path / "all_datasets_hierarchical_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"📄 Detailed results saved to: {results_file}")
        
        # Save comparison summary
        summary_file = self.results_path / "hierarchical_vs_classical_comparison.json"
        with open(summary_file, 'w') as f:
            json.dump(self.comparison_summary, f, indent=2)
        print(f"📄 Comparison summary saved to: {summary_file}")
        
        # Save CSV summary for easy analysis
        csv_data = []
        for dataset, result in self.all_results.items():
            csv_data.append({
                'Dataset': dataset,
                'Proteins': result['proteins_count'],
                'Initial_Comb_Nodes': result['initial_comb_nodes'],
                'Final_Comb_Nodes': result['final_comb_nodes'],
                'Comb_Reduction': result['comb_reduction'],
                'Improvement_Percentage': result['improvement_percentage'],
                'Iterations_Used': result['iterations_used'],
                'Resolution_Success': result['resolution_success']
            })
        
        csv_file = self.results_path / "hierarchical_analysis_summary.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"📄 CSV summary saved to: {csv_file}")
        
        print()
    
    def save_intermediate_results(self):
        """Save intermediate results during analysis"""
        intermediate_file = self.results_path / "intermediate_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"💾 Intermediate results saved to: {intermediate_file}")

def main():
    """Main function to run the complete analysis"""
    print("🔬 HIERARCHICAL FEATURE-BASED TREE CONSTRUCTION")
    print("📊 ALL DATASETS ANALYSIS")
    print("=" * 60)
    print()
    
    # Create analyzer
    analyzer = AllDatasetsAnalyzer()
    
    # Run analysis on all datasets
    results = analyzer.run_all_datasets_analysis()
    
    if results:
        # Generate comparison summary
        summary = analyzer.generate_comparison_summary()
        
        # Save all results
        analyzer.save_results()
        
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 30)
        print("📊 Key Findings:")
        print(f"   • Analyzed {len(results)} datasets")
        print(f"   • Overall comb reduction: {summary['overall_improvement_percentage']:.1f}%")
        print(f"   • Fully resolved: {summary['success_rates']['full_resolution_rate']:.1f}% of datasets")
        print(f"   • Any improvement: {summary['success_rates']['any_improvement_rate']:.1f}% of datasets")
        print()
        print("📁 Results saved to:")
        print("   • all_datasets_hierarchical_analysis.json")
        print("   • hierarchical_vs_classical_comparison.json")
        print("   • hierarchical_analysis_summary.csv")
    else:
        print("❌ No datasets were successfully analyzed")

if __name__ == "__main__":
    main()
