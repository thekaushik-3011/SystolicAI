"""
Visualization engine for the Systolic Array.
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(activity_grid, cycle, save_path=None):
    """
    Plot the active PEs for a given cycle.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(activity_grid, cmap="YlGnBu", annot=True, cbar=False,
                linewidths=.5, vmin=0, vmax=1)
    plt.title(f"PE Activity at Cycle {cycle}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
def plot_utilization_over_time(history, save_path=None):
    """
    Plot the number of active PEs over time.
    """
    active_pes = [sum(sum(row) for row in activity) for activity in history]
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(history)), active_pes, marker='o', linestyle='-')
    plt.title("PE Utilization Over Time")
    plt.xlabel("Clock Cycle")
    plt.ylabel("Active PEs")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
