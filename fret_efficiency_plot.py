import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def fret_efficiency(r, r0):
    """
    Calculate FRET efficiency based on distance and Förster radius.
    
    Parameters
    ----------
    r : float or ndarray
        Distance between donor and acceptor (nm)
    r0 : float
        Förster radius (nm)
    
    Returns
    -------
    float or ndarray
        FRET efficiency (percentage)
    """
    return 100 / (1 + (r/r0)**6)

def plot_fret_efficiency():
    """Create FRET efficiency plot with specified requirements."""
    r = np.linspace(0, 14, 200)
    r0_values = [4, 6, 8]
    
    plt.figure(figsize=(8, 5), dpi=100)
    
    for r0 in r0_values:
        efficiency = fret_efficiency(r, r0)
        plt.plot(r, efficiency, color='black', linewidth=2, 
                 label=f'{r0} nm')
        
        # Label at 0.5 nm right of R0
        label_x = r0 + 0.5
        label_y = fret_efficiency(r0, r0)
        plt.text(label_x, label_y, f'{r0} nm', fontsize=14, 
                 fontname='Arial')
    
    plt.axhline(10, color='red', linestyle='--', linewidth=2)
    plt.axhline(90, color='red', linestyle='--', linewidth=2)
    plt.axhline(50, color='green', linestyle='--', linewidth=2)
    
    plt.xlim(0, 14)
    plt.ylim(0, 100)
    plt.xlabel('Distance (nm)', fontsize=14, fontname='Arial')
    plt.ylabel('FRET Efficiency (%)', fontsize=14, fontname='Arial')
    
    plt.tick_params(width=2, length=6)
    plt.setp(plt.gca().get_edges(), linewidth=2)
    plt.title('FRET Efficiency vs Distance', fontsize=14, fontname='Arial')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_fret_efficiency()
