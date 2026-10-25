import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from plot_lines import TECH_STYLE  # Reuse style definitions
from aup_utils import weight_function
import matplotlib.patheffects as path_effects

def plot_illustration(save_path="aup_illustration.png", font_size_axis: int = 20):
    # 1. Data Generation (Quadratic with vertex at rho=1)
    rho = np.array([1.0, 2.0, 3.2, 4.5, 5.5])
    # Quadratic function opening downwards with vertex at rho=1: y = a*(rho-1)^2 + k
    # k = 83 (max accuracy), a < 0.
    a = -1.2
    y_max = 83.0
    y = a * (rho - 1)**2 + y_max
    
    # 2. Style Setup
    style_context = 'dark_background'
    bg_color = 'black'
    fg_color = 'white'
    grid_color = 'white'
    spine_color = 'white'
    
    # Color setup (Red theme)
    c_name = 'red'
    grad_colors = TECH_STYLE[c_name]
    main_color = grad_colors[1]
    text_color = grad_colors[1]
    cmap = LinearSegmentedColormap.from_list(f"tech_{c_name}", grad_colors)

    with plt.style.context(style_context):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Calculate smooth curve for plotting
        x_smooth = np.linspace(rho.min(), rho.max(), 300)
        y_smooth = a * (x_smooth - 1)**2 + y_max
        
        # Calculate weights and weighted accuracy for filling
        # Using alpha=10.0 as per user modification
        weights = np.array([weight_function(yi, y_max, alpha=10.0) for yi in y_smooth])
        y_weighted = y_smooth * weights
        
        # 3. Plot Filled Area (AUP) - Weighted
        # Fill includes the initial rectangle from 0 to rho[0] (where weight is 1) and then the weighted curve
        # At rho=1, y=y_max=83, so weight=1. 
        fill_x = np.concatenate(([0, 0], x_smooth, [x_smooth[-1]]))
        fill_y = np.concatenate(([0, y_max], y_weighted, [0]))
        
        # Use hatching instead of solid alpha fill
        # Reduced density: hatch='/' instead of '//'
        ax.fill(fill_x, fill_y, facecolor='none', edgecolor=main_color, hatch='/', alpha=0.5, label='AUP Region')
        
        # Add "AUP" text
        # Moved slightly left (divided by 2.5 instead of 2 for x)
        center_x = rho[-1] / 3.2 
        center_y = y_weighted[0] / 2.5 
        txt = ax.text(center_x, center_y, "AUP", color=text_color, fontsize=24, 
                fontweight='bold', ha='center', va='center', alpha=1.0)
        
        # Add white outline to text
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

        # 4. Plot the Line (Original Accuracy)
        points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(x_smooth.min(), x_smooth.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_linewidth(3)
        lc.set_alpha(0.9)
        ax.add_collection(lc)

        # 5. Add Markers and Dashed Line for initial rectangle
        ax.scatter(rho, y, color='white', edgecolors=main_color, s=60, zorder=10, marker='o', linewidth=1.5)
        
        # Connect (0, y[0]) to (rho[0], y[0]) to emphasize the start
        ax.plot([0, rho[0]], [y[0], y[0]], color=main_color, linestyle='--', alpha=0.6)
        ax.scatter([0], [y[0]], color=main_color, s=30, alpha=0.6) # origin point on y-axis

        # 6. Final Styling
        ax.set_xlabel(r'Parallelism $\rho$ (TPF)', fontsize=font_size_axis, color=fg_color)
        ax.set_ylabel(r'Accuracy (%)', fontsize=font_size_axis, color=fg_color)
        ax.set_xlim(0, rho.max() * 1.1)
        ax.set_ylim(0, 90) # Adjusted ylim for y_max=83
        
        ax.grid(True, linestyle='--', alpha=0.15, color=grid_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        ax.tick_params(colors=fg_color)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"Illustration saved to {save_path}")

if __name__ == "__main__":
    plot_illustration()
