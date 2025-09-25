import os
import threading
from codecarbon import EmissionsTracker
from typing import Optional, Dict, Any

class SimpleAsyncTracker:
    def __init__(self, project_name, emissions_file_name, output_dir=None):
        self.project_name = project_name
        self.emissions_file_name = emissions_file_name
        
        # Setup directory
        if output_dir is None:
            self.emissions_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "emissions_data"
            )
        else:
            self.emissions_dir = output_dir
            
        os.makedirs(self.emissions_dir, exist_ok=True)
        
        # Create tracker with file saving disabled
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=self.emissions_dir,
            output_file=f"{emissions_file_name}_fixed.csv",
            save_to_file=True,  # Enable detailed internal tracking
            log_level="info",
            #measure_power_secs=15,  # More frequent measurements
        )
        
    def start(self):
        """Start tracking"""
        return self.tracker.start()
    
    def stop(self):
        """Stop tracking and save data asynchronously"""
        try:
            emissions_data = self.tracker.stop()
            
            # Save data in background thread (fire and forget)
            if emissions_data:
                save_thread = threading.Thread(
                    target=self._save_data_async,
                    args=(emissions_data,),
                    daemon=True
                )
                save_thread.start()
            
            return emissions_data
            
        except Exception as e:
            print(f"Error stopping tracker: {e}")
            return None
    
    def _save_data_async(self, data):
        """Save emissions data in background thread"""
        try:
            import pandas as pd
            
            # Explicitly extract emissions data fields
            emissions_dict = {
                'timestamp': data.timestamp,
                'project_name': self.project_name,
                'emissions_kg': data.emissions,
                'emissions_rate_kg_per_s': getattr(data, 'emissions_rate', 0),
                'cpu_power_w': getattr(data, 'cpu_power', 0),
                'gpu_power_w': getattr(data, 'gpu_power', 0),
                'ram_power_w': getattr(data, 'ram_power', 0),
                'cpu_energy_kwh': getattr(data, 'cpu_energy', 0),
                'gpu_energy_kwh': getattr(data, 'gpu_energy', 0),
                'ram_energy_kwh': getattr(data, 'ram_energy', 0),
                'energy_consumed_kwh': data.energy_consumed,
                'duration_seconds': data.duration,
                'emissions_count': getattr(data, 'emissions_count', 1),
                'country_name': getattr(data, 'country_name', ''),
                'country_iso_code': getattr(data, 'country_iso_code', ''),
                'region': getattr(data, 'region', ''),
                'cloud_provider': getattr(data, 'cloud_provider', ''),
                'cloud_region': getattr(data, 'cloud_region', ''),
            }
            
            filepath = os.path.join(self.emissions_dir, f"emissions_{self.emissions_file_name}.csv")
            df = pd.DataFrame([emissions_dict])
            
            # Append to existing file or create new one
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, mode='w', header=True, index=False)
                
        except Exception as e:
            print(f"Error saving emissions data asynchronously: {e}")


def setup_emissions_tracker(project_name, emissions_file_name, output_dir=None):
    """Drop-in replacement for your existing function"""
    try:
        tracker = SimpleAsyncTracker(project_name, emissions_file_name, output_dir)
        return tracker
    except Exception as e:
        print(f"Warning: Could not setup emissions tracker: {e}")
        return None