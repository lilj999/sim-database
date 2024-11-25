import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_query_results_with_categories(results, distances, sample_id, zero_height=0.5, category_colors=None):
    """
    Plot a bar chart of query results with Hamming distances, highlighting distance=0 with a height of 0.5.
    Dynamically set the chart title based on the sample's category.

    :param results: List of data IDs (e.g., "5.0-14").
    :param distances: List of corresponding Hamming distances.
    :param sample_id: The sample ID used for the query (e.g., "1.0-67").
    :param zero_height: Height to assign for Hamming distance=0 while keeping Y-axis tick at 0.
    :param category_colors: Dict mapping category (0-5) to specific colors. E.g., {0: 'red', 1: 'blue', ...}.
    """
    # Mapping of categories to names
    cat_dic = {'0': 'Benign', '1': 'Exploits', '2': 'Fuzzers', '3': 'Generic', '4': 'Reconnaissance', '5': 'Shellcode'}

    # Default color mapping if not provided
    if category_colors is None:
        category_colors = {
            0: '#BFDFD2',
            1: '#EFCE87',
            2: '#257D8B',
            3: '#EAA558',
            4: '#68BED9',
            5: '#ED8D5A'
        }

    # Extract category from sample_id (e.g., "1.0-67" -> "1")
    sample_category = sample_id.split('-')[0].split('.')[0]
    sample_category_name = cat_dic[sample_category]  # Get the category name from the mapping

    # Ensure distances and results are sorted by Hamming distance
    sorted_results = sorted(zip(results, distances), key=lambda x: x[1])
    sorted_data_ids, sorted_distances = zip(*sorted_results)

    # Convert data IDs into "category n- sample m" format
    formatted_x_labels = [
        f"{cat_dic[data_id.split('-')[0].split('.')[0]]}-sample{data_id.split('-')[1]}"
        for data_id in sorted_data_ids
    ]

    # Extract categories for coloring
    categories = [int(data_id.split('-')[0].split('.')[0]) for data_id in sorted_data_ids]

    # Adjust Hamming distances: increase height for distance=0
    adjusted_distances = [zero_height if dist == 0 else dist for dist in sorted_distances]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    bar_positions = range(len(sorted_data_ids))
    bar_colors = [category_colors[category] for category in categories]

    bars = plt.bar(bar_positions, adjusted_distances, color=bar_colors, edgecolor='black', alpha=0.85)

    # Add text labels above each bar (integer format)
    for bar, dist in zip(bars, sorted_distances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{int(dist)}', ha='center', va='bottom', fontsize=10)

    # Add formatted labels to the X-axis
    plt.xticks(bar_positions, formatted_x_labels, rotation=45, ha='right', fontsize=10)

    # Set Y-axis ticks to reflect actual Hamming distances
    y_ticks = sorted(set(adjusted_distances))  # Include adjusted height for zero
    y_tick_labels = [0 if height == zero_height else int(height) for height in y_ticks]  # Show 0 for height=0.5
    plt.yticks(y_ticks, y_tick_labels, fontsize=10)

    # Add legend for **all categories**
    handles = [
        Patch(color=category_colors[cat], label=cat_dic[str(cat)])
        for cat in range(len(cat_dic))  # Ensure all categories are included in the legend
    ]
    plt.legend(handles=handles, title="Categories", fontsize=10, title_fontsize=12)

    # Add labels and dynamic title
    plt.title(f"Hamming Distance of Each Feature to the Sample of {sample_category_name}", fontsize=14, weight='bold')
    plt.xlabel("Queried Data", fontsize=12)
    plt.ylabel("Distance to Query", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
