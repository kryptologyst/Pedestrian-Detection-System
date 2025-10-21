"""
Visualization utilities for pedestrian detection.

This module provides enhanced visualization capabilities including
statistics plotting and detection analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from detector import DetectionResult, Detection


class DetectionVisualizer:
    """Enhanced visualization for detection results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with matplotlib style."""
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_confidence_distribution(self, results: List[DetectionResult], 
                                   save_path: Optional[str] = None) -> None:
        """
        Plot confidence score distribution across multiple detections.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the plot
        """
        all_confidences = []
        for result in results:
            confidences = [det.confidence for det in result.detections]
            all_confidences.extend(confidences)
        
        if not all_confidences:
            print("No detections to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Detection Confidence Scores')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_detection_counts(self, results: List[DetectionResult], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot detection counts across multiple images.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the plot
        """
        counts = [len(result.detections) for result in results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts)), counts, alpha=0.7)
        plt.xlabel('Image Index')
        plt.ylabel('Number of Pedestrians Detected')
        plt.title('Pedestrian Detection Counts Across Images')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_count = np.mean(counts)
        plt.axhline(y=mean_count, color='red', linestyle='--', 
                   label=f'Mean: {mean_count:.1f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detection counts plot saved to: {save_path}")
        
        plt.show()
    
    def plot_processing_times(self, results: List[DetectionResult], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot processing times for different images.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the plot
        """
        times = [result.processing_time for result in results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(times)), times, marker='o', alpha=0.7)
        plt.xlabel('Image Index')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Times Across Images')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_time = np.mean(times)
        plt.axhline(y=mean_time, color='red', linestyle='--', 
                   label=f'Mean: {mean_time:.3f}s')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Processing times plot saved to: {save_path}")
        
        plt.show()
    
    def create_detection_summary(self, results: List[DetectionResult], 
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive summary of detection results.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the summary plot
            
        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {}
        
        # Collect all statistics
        all_counts = [len(result.detections) for result in results]
        all_times = [result.processing_time for result in results]
        all_confidences = []
        
        for result in results:
            confidences = [det.confidence for det in result.detections]
            all_confidences.extend(confidences)
        
        # Calculate summary statistics
        summary = {
            'total_images': len(results),
            'total_detections': sum(all_counts),
            'avg_detections_per_image': np.mean(all_counts),
            'max_detections': max(all_counts) if all_counts else 0,
            'min_detections': min(all_counts) if all_counts else 0,
            'avg_processing_time': np.mean(all_times),
            'total_processing_time': sum(all_times),
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'max_confidence': max(all_confidences) if all_confidences else 0,
            'min_confidence': min(all_confidences) if all_confidences else 0
        }
        
        # Create summary visualization
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Detection counts
            axes[0, 0].hist(all_counts, bins=min(20, len(set(all_counts))), 
                           alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Detections per Image')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Detection Counts')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Processing times
            axes[0, 1].hist(all_times, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Processing Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Processing Times')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Confidence scores
            if all_confidences:
                axes[1, 0].hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('Confidence Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Distribution of Confidence Scores')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No detections found', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Confidence Scores')
            
            # Summary text
            summary_text = f"""
            Total Images: {summary['total_images']}
            Total Detections: {summary['total_detections']}
            Avg Detections/Image: {summary['avg_detections_per_image']:.1f}
            Avg Processing Time: {summary['avg_processing_time']:.3f}s
            Avg Confidence: {summary['avg_confidence']:.3f}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Summary Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detection summary plot saved to: {save_path}")
            plt.show()
        
        return summary
    
    def plot_bbox_analysis(self, results: List[DetectionResult], 
                          save_path: Optional[str] = None) -> None:
        """
        Analyze bounding box characteristics.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the plot
        """
        widths = []
        heights = []
        areas = []
        
        for result in results:
            for detection in result.detections:
                x1, y1, x2, y2 = detection.bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                widths.append(width)
                heights.append(height)
                areas.append(area)
        
        if not widths:
            print("No detections to analyze")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Width distribution
        axes[0].hist(widths, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Bounding Box Width (pixels)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Bounding Box Widths')
        axes[0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[1].hist(heights, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Bounding Box Height (pixels)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Bounding Box Heights')
        axes[1].grid(True, alpha=0.3)
        
        # Area distribution
        axes[2].hist(areas, bins=20, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Bounding Box Area (pixels²)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Bounding Box Areas')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bounding box analysis plot saved to: {save_path}")
        
        plt.show()


def create_detection_report(results: List[DetectionResult], 
                          output_dir: str = "reports") -> str:
    """
    Create a comprehensive detection report with visualizations.
    
    Args:
        results: List of detection results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    visualizer = DetectionVisualizer()
    
    # Generate all visualizations
    visualizer.plot_confidence_distribution(results, 
                                           str(output_path / "confidence_distribution.png"))
    visualizer.plot_detection_counts(results, 
                                   str(output_path / "detection_counts.png"))
    visualizer.plot_processing_times(results, 
                                    str(output_path / "processing_times.png"))
    visualizer.plot_bbox_analysis(results, 
                                 str(output_path / "bbox_analysis.png"))
    
    # Create summary
    summary = visualizer.create_detection_summary(results, 
                                                str(output_path / "summary.png"))
    
    # Save summary as JSON
    import json
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Detection report generated in: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    print("DetectionVisualizer - Visualization utilities for pedestrian detection")
    print("Use this module to create comprehensive analysis reports of detection results.")
