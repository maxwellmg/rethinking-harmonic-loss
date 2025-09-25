#model_emissions_tracker

from codecarbon import EmissionsTracker
import os

def setup_emissions_tracker(project_name, emissions_file_name, output_dir= None):
    try:
        # If no output_dir provided, create path to emissions_data directory
        if output_dir is None:
            # Go up one level from Run directory to parent, then into emissions_data
            emissions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emissions_data")
        else:
            emissions_dir = output_dir
            
        # Ensure the emissions_data directory exists
        os.makedirs(emissions_dir, exist_ok=True)
        
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=emissions_dir,
            output_file=f"emissions_{emissions_file_name}.csv",
            save_to_file=True,
            log_level="error"
        )
        return tracker
    except Exception as e:
        print(f"Warning: Could not setup emissions tracker: {e}")
        return None




    """try:
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            output_file=f"emissions_{emissions_file_name}.csv",
            save_to_file=True,
            log_level="error"
        )
        return tracker
    except Exception as e:
        print(f"Warning: Could not setup emissions tracker: {e}")
        return None
        """
    '''#Setup emissions tracker with error handling
    
    Args:
        project_name (str): Name for the tracking project
        emissions_file_name (str): Base name for the emissions file
        output_dir (str, optional): Directory to save emissions data. If None, uses emissions_data directory
        
    Returns:
        EmissionsTracker or None: Tracker instance or None if setup fails
    '''