#!/usr/bin/env python3
"""
Demo script for the Advanced Computer Vision Security System
Perfect for interviews and presentations
"""

import sys
import time
import argparse
from advanced_security_system import AdvancedSecuritySystem

def demo_basic_detection():
    """Demo basic motion and person detection."""
    print("🎯 DEMO 1: Basic Detection System")
    print("=" * 50)
    print("This demo shows:")
    print("• Real-time motion detection")
    print("• Person detection (if YOLO models available)")
    print("• Performance monitoring")
    print("• Evidence collection")
    print("\nPress 'q' to quit, 'r' to reset, 'p' for performance metrics")
    print("=" * 50)
    
    # Create system with basic config
    config = {
        "video_source": 0,
        "detectors": {
            "motion": {
                "enabled": True, 
                "min_contour_area": 1000,
                "algorithm": "MOG2",
                "confidence_threshold": 0.5
            },
            "person": {
                "enabled": True, 
                "confidence_threshold": 0.3,
                "model_type": "YOLO",
                "input_size": [416, 416]
            }
        },
        "tracking": {
            "enabled": True,
            "algorithm": "CSRT",
            "max_disappeared": 30.0
        },
        "performance": {
            "monitoring_enabled": True,
            "update_interval": 1.0,
            "max_history": 1000
        },
        "analytics": {
            "enabled": True,
            "dashboard_enabled": False,
            "export_interval": 300
        },
        "output": {
            "save_detections": True,
            "save_videos": True,
            "output_directory": "advanced_detections",
            "log_level": "INFO"
        }
    }
    
    system = AdvancedSecuritySystem()
    system.config.update(config)
    system.run()

def demo_video_analysis():
    """Demo video file analysis."""
    print("🎯 DEMO 2: Video File Analysis")
    print("=" * 50)
    print("This demo shows:")
    print("• Batch processing of video files")
    print("• Detection accuracy on pre-recorded footage")
    print("• Evidence collection and reporting")
    print("• Performance analysis")
    print("=" * 50)
    
    video_file = input("Enter video file path (or press Enter for 'raw_cctv/test.mp4'): ").strip()
    if not video_file:
        video_file = "raw_cctv/test.mp4"
    
    config = {
        "video_source": video_file,
        "detectors": {
            "motion": {
                "enabled": True, 
                "min_contour_area": 800,
                "algorithm": "MOG2",
                "confidence_threshold": 0.5
            },
            "person": {
                "enabled": True, 
                "confidence_threshold": 0.2,
                "model_type": "YOLO",
                "input_size": [416, 416]
            }
        },
        "tracking": {
            "enabled": True,
            "algorithm": "CSRT",
            "max_disappeared": 30.0
        },
        "performance": {
            "monitoring_enabled": True,
            "update_interval": 1.0,
            "max_history": 1000
        },
        "analytics": {
            "enabled": True,
            "dashboard_enabled": False,
            "export_interval": 300
        },
        "output": {
            "save_detections": True,
            "save_videos": True,
            "output_directory": "advanced_detections",
            "log_level": "INFO"
        }
    }
    
    system = AdvancedSecuritySystem()
    system.config.update(config)
    system.run()

def demo_performance_optimization():
    """Demo performance optimization features."""
    print("🎯 DEMO 3: Performance Optimization")
    print("=" * 50)
    print("This demo shows:")
    print("• Real-time performance monitoring")
    print("• System optimization recommendations")
    print("• Resource usage analysis")
    print("• Bottleneck identification")
    print("\nPress 'p' for performance metrics, 'a' for analytics")
    print("=" * 50)
    
    config = {
        "video_source": 0,
        "input_resolution": [640, 480],
        "detectors": {
            "motion": {
                "enabled": True,
                "algorithm": "MOG2",
                "confidence_threshold": 0.5,
                "min_contour_area": 1000
            },
            "person": {
                "enabled": True,
                "model_type": "YOLO",
                "confidence_threshold": 0.3,
                "input_size": [416, 416]
            }
        },
        "performance": {
            "monitoring_enabled": True,
            "update_interval": 0.5
        },
        "analytics": {
            "enabled": True,
            "dashboard_enabled": False,
            "export_interval": 300
        },
        "output": {
            "save_detections": True,
            "save_videos": True,
            "output_directory": "advanced_detections",
            "log_level": "INFO"
        }
    }
    
    system = AdvancedSecuritySystem()
    system.config.update(config)
    system.run()

def demo_analytics_dashboard():
    """Demo analytics and visualization."""
    print("🎯 DEMO 4: Analytics Dashboard")
    print("=" * 50)
    print("This demo shows:")
    print("• Real-time data visualization")
    print("• Detection pattern analysis")
    print("• Performance trend analysis")
    print("• Interactive dashboard")
    print("=" * 50)
    
    config = {
        "video_source": 0,
        "detectors": {
            "motion": {
                "enabled": True,
                "algorithm": "MOG2",
                "confidence_threshold": 0.5,
                "min_contour_area": 1000
            },
            "person": {
                "enabled": True,
                "model_type": "YOLO",
                "confidence_threshold": 0.3,
                "input_size": [416, 416]
            }
        },
        "analytics": {
            "enabled": True,
            "dashboard_enabled": True
        },
        "performance": {"monitoring_enabled": True}
    }
    
    system = AdvancedSecuritySystem()
    system.config.update(config)
    system.run()

def demo_algorithm_comparison():
    """Demo different detection algorithms."""
    print("🎯 DEMO 5: Algorithm Comparison")
    print("=" * 50)
    print("This demo shows:")
    print("• Multiple motion detection algorithms")
    print("• Performance comparison")
    print("• Accuracy analysis")
    print("• Algorithm selection criteria")
    print("=" * 50)
    
    algorithms = ["MOG2", "KNN", "GMG"]
    
    for i, algorithm in enumerate(algorithms, 1):
        print(f"\nTesting {algorithm} algorithm...")
        print("Press 'q' to skip to next algorithm")
        
        config = {
            "video_source": 0,
            "detectors": {
                "motion": {
                    "enabled": True,
                    "algorithm": algorithm,
                    "min_contour_area": 1000,
                    "confidence_threshold": 0.5
                },
                "person": {
                    "enabled": False,
                    "model_type": "YOLO",
                    "confidence_threshold": 0.3,
                    "input_size": [416, 416]
                }
            },
            "performance": {"monitoring_enabled": True}
        }
        
        system = AdvancedSecuritySystem()
        system.config.update(config)
        system.run()

def show_system_info():
    """Show system information and capabilities."""
    print("🎯 ADVANCED COMPUTER VISION SECURITY SYSTEM")
    print("=" * 60)
    print("📊 CAPABILITIES:")
    print("• Motion Detection (MOG2, KNN, GMG)")
    print("• Person Detection (YOLO v3)")
    print("• Object Tracking (CSRT, KCF, MOSSE)")
    print("• Real-time Performance Monitoring")
    print("• Data Analytics & Visualization")
    print("• Evidence Collection & Reporting")
    print("• Multi-threaded Processing")
    print("• Configurable Architecture")
    print("\n📈 PERFORMANCE:")
    print("• 15-25 FPS processing")
    print("• 200-500 MB memory usage")
    print("• 30-60% CPU utilization")
    print("• < 50ms detection latency")
    print("\n🛠️ TECHNICAL STACK:")
    print("• Python 3.7+")
    print("• OpenCV 4.8+")
    print("• NumPy, Pandas, Matplotlib")
    print("• YOLO v3, CSRT Tracking")
    print("• Multi-threading, Performance Profiling")
    print("=" * 60)

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Advanced CV Security System Demo')
    parser.add_argument('--demo', type=int, choices=[1,2,3,4,5], help='Demo number to run')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    if args.info:
        show_system_info()
        return 0
    
    if args.demo:
        demos = {
            1: demo_basic_detection,
            2: demo_video_analysis,
            3: demo_performance_optimization,
            4: demo_analytics_dashboard,
            5: demo_algorithm_comparison
        }
        demos[args.demo]()
        return 0
    
    # Interactive menu
    while True:
        print("\n🎯 ADVANCED COMPUTER VISION SECURITY SYSTEM")
        print("=" * 50)
        print("Select a demo:")
        print("1. Basic Detection System")
        print("2. Video File Analysis")
        print("3. Performance Optimization")
        print("4. Analytics Dashboard")
        print("5. Algorithm Comparison")
        print("6. System Information")
        print("0. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (0-6): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            demo_basic_detection()
        elif choice == '2':
            demo_video_analysis()
        elif choice == '3':
            demo_performance_optimization()
        elif choice == '4':
            demo_analytics_dashboard()
        elif choice == '5':
            demo_algorithm_comparison()
        elif choice == '6':
            show_system_info()
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    exit(main())
