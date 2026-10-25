import os
import matplotlib.pyplot as plt
from plot_lines import TECH_STYLE

data_str = '''
Dream: TPS:				
	gsm8k_cot			
	H100	A100	Acc	token/forward (TPF)
Qwen-2.5-7B-it	57.32	50.36	74.10	1.00
Dream	27.62	8.32	83.94	1.00
Fast-dLLM-Dream	77.25	51.55	79.00	1.44
Fast-dLLM-v2-7B	150.01	109.68	77.48	2.21
dParallel-Dream	168.36	80.23	82.12	3.02
d3LLM-Dream	235.34	128.19	81.86	5.01
				
				
				
LLaDA: TPS:				
	gsm8k_cot			
	H100	A100	Acc	token/forward (TPF)
Qwen-2.5-7B-it	57.32	50.36	74.10	1.00
LLaDA	102.45	67.91	72.55	1.00
Fast-dLLM-LLaDA	114.29	79.14	74.68	2.77
D2F	102.13	76.24	73.24	2.88
dParallel-LLaDA	172.23	105.85	72.63	5.14
d3LLM-LLaDA	280.97	180.23	73.10	9.08
'''

markers = [
    '^',
    'v',
    'X',
    '+',
    's',
    'p',
    '*'
]

def parse_and_plot():
    # Split data sections
    sections = data_str.strip().split('LLaDA: TPS:')
    datasets = {
        'Dream': sections[0].split('Dream: TPS:')[1],
        'LLaDA': sections[1]
    }
    
    # Colors for 6 methods (matches main.py logic)
    colors = ['orange', 'grey', 'purple', 'green', 'blue', 'red']

    if not os.path.exists('speed_accuracy_plot'):
        os.mkdir('speed_accuracy_plot')

    for name, raw_text in datasets.items():
        # Parse lines: split by whitespace, filter valid data rows
        lines = [l.split() for l in raw_text.strip().split('\n') if l.strip()]
        data = []
        for parts in lines:
            try:
                # Expecting: Name H100 A100 Acc ...
                data.append({
                    'name': parts[0],
                    'H100': float(parts[1]),
                    'A100': float(parts[2]),
                    'Acc': float(parts[3])
                })
            except (ValueError, IndexError):
                continue # Skip headers or malformed lines
        
        # Plot for H100 and A100
        for device in ['H100', 'A100']:
            # Use dark background context for tech style
            with plt.style.context('dark_background'):
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
                ax.set_facecolor('black')
                
                # Plot each method
                for i, d in enumerate(data):
                    c_key = colors[i % len(colors)]
                    color = TECH_STYLE.get(c_key, TECH_STYLE['grey'])[1] # Use neon/bright color
                    marker = markers[i % len(markers)]
                    
                    # Scatter point
                    ax.scatter(d[device], d['Acc'], c=color, marker=marker, s=250, alpha=0.9, 
                             edgecolors='white', linewidth=1.5, zorder=10)
                    
                    # Label text below point
                    ax.annotate(d['name'], (d[device], d['Acc']), xytext=(0, -15), 
                              textcoords='offset points', color=color, fontsize=12, 
                              fontweight='bold', ha='center', va='top')

                # Axes and Title
                ax.set_xlabel(f'{device} Speed (tokens/s)', fontsize=12, color='white')
                ax.set_ylabel('Accuracy (%)', fontsize=12, color='white')
                # ax.set_title(f'{name} on {device}', fontsize=14, color='white', fontweight='bold')
                
                # Styling Details
                ax.grid(True, linestyle='--', alpha=0.15, color='white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(colors='white')
                
                # Save
                save_path = f'speed_accuracy_plot/{name}_{device}.png'
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
                plt.close()
                print(f"Saved {save_path}")

if __name__ == '__main__':
    parse_and_plot()
