import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_intensity_boxplot(image_path, filename="intensity_plot_final.png"):
    # 1. Load the selected image
    # cv2.imread loads in BGR format by default
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Prepare Data
    b, g, r = cv2.split(img)
    df = pd.DataFrame({
        'Red': r.flatten(),
        'Green': g.flatten(),
        'Blue': b.flatten()
    })

    # 3. Set Global Styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 4. Create Plot
    sns.boxplot(
        data=df,
        palette=['red', 'green', 'blue'],
        width=0.5,
        showfliers=False,
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        ax=ax
    )

    # 5. Axis and Spine Customization
    plt.ylabel('Intensity Value', fontsize=14)
    plt.xlabel('Color Channel', fontsize=14)
    plt.ylim(0, 255)  # Standard RGB range is 0-255; change back to 150 if preferred

    # Ensure all borders (spines) are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # Configure ticks to point outward
    ax.tick_params(axis='both', which='major', direction='out',
                   color='black', length=6, width=1.0,
                   bottom=True, left=True)

    # 6. Save and Show
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved successfully as {filename}")
    plt.show()

plot_intensity_boxplot("D:\pic_to_analyze.jpg")