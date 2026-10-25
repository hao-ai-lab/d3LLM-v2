import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plot_lines import TECH_STYLE

def plot_histogram(data: dict, attributes: list, assigned_colors: dict = None, save_path: str = None, is_dark_mode: bool = True, font_size_axis: int = 12, font_size_legend: int = 12):
    """
    Plot horizontal gradient histogram for AUP results.
    data: {method_name: [val1, val2, ...]} corresponding to attributes order
    attributes: list of task names
    font_size_axis: font size for axis labels
    font_size_legend: font size for legend
    """
    method_names = list(data.keys())
    num_methods = len(method_names)
    num_tasks = len(attributes)
    
    # Calculate Global Max (rounded to 50)
    raw_vals = [v for vals in data.values() for v in vals]
    _max = max(raw_vals) if raw_vals else 0
    global_max = 50.0 if _max == 0 else np.ceil(_max / 50.0) * 50.0

    # Style Settings
    if is_dark_mode:
        style_context = 'dark_background'
        bg_color = 'black'
        fg_color = 'white'
        grid_color = 'white'
        grid_alpha = 0.15
    else:
        style_context = 'default'
        bg_color = '#F5F5F7'
        fg_color = 'black'
        grid_color = '#86868b'
        grid_alpha = 0.3

    with plt.style.context(style_context):
        fig, ax = plt.subplots(figsize=(12, num_tasks * 1.5 + 2))
        
        if not is_dark_mode:
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

        # Layout parameters
        bar_height = 0.8 / num_methods
        y_indices = np.arange(num_tasks)
        
        # Plot bars for each method
        for i, method in enumerate(method_names):
            vals = data[method]
            if assigned_colors:
                c_name = assigned_colors[i] if isinstance(assigned_colors, list) else assigned_colors.get(method, 'grey')
            else:
                c_name = 'grey'
            grad_colors = TECH_STYLE.get(c_name, TECH_STYLE['grey'])
            cmap = LinearSegmentedColormap.from_list(f"grad_{c_name}", grad_colors)
            
            # Calculate positions: Center group around y_index
            # i=0 -> y - (N/2 - 0.5)*h
            y_pos = y_indices - (num_methods / 2.0 - 0.5 - i) * bar_height
            
            for j, val in enumerate(vals):
                # Gradient Bar using imshow
                # Extent: left, right, bottom, top
                extent = [0, val, y_pos[j] - bar_height/2 + 0.02, y_pos[j] + bar_height/2 - 0.02]
                # Gradient array (1, 256) from 0 to 1
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                
                ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent, zorder=10)
                
                # Value label
                ax.text(val + global_max * 0.01, y_pos[j], f"{val:.1f}", 
                        va='center', ha='left', color=fg_color, fontsize=font_size_axis, fontweight='bold')

        # Dummy bars for Legend (since imshow doesn't add to legend)
        for i, method in enumerate(method_names):
            if assigned_colors:
                c_name = assigned_colors[i] if isinstance(assigned_colors, list) else assigned_colors.get(method, 'grey')
            else:
                c_name = 'grey'
            # Use the end color (brighter) for legend
            color = TECH_STYLE.get(c_name, TECH_STYLE['grey'])[1] 
            ax.barh([0], [0], color=color, label=method)

        # Styling
        ax.set_xlim(0, global_max)
        ax.set_ylim(-0.6, num_tasks - 0.4)
        ax.set_yticks(y_indices)
        ax.set_yticklabels(attributes, fontsize=font_size_axis, color=fg_color, fontweight='bold')
        ax.invert_yaxis() # First task at top
        # ax.set_xlabel('AUP Score', fontsize=font_size_axis, color=fg_color)
        
        ax.grid(True, axis='x', linestyle='--', alpha=grid_alpha, color=grid_color, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0, labelbottom=False)
        ax.tick_params(axis='y', length=0)

        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(num_methods, 5), 
                 frameon=False, labelcolor=fg_color, fontsize=font_size_legend)
        
        plt.tight_layout()
        
        if save_path:
            fc = bg_color if not is_dark_mode else 'black'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fc)
            print(f"Histogram saved to {save_path}")
