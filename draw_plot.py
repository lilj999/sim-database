import matplotlib.pyplot as plt

# Data for the first dataset
samples1 = [5000, 10000, 15000, 20000, 30000, 50000, 100000, 150000]
query_times1 = [0.016608715057373047, 0.03144359588623047, 0.04663681983947754, 
                0.06900358200073242, 0.09549570083618164, 0.15640544891357422, 
                0.31186866760253906, 0.5029621124267578]

# Data for the second dataset
samples2 = [5000, 10000, 15000, 20000, 30000, 50000, 100000, 150000]
query_times2 = [0.00599980354309082, 0.025999784469604492, 0.02500009536743164, 
                0.08200287818908691, 0.05176806449890137, 0.04261350631713867, 
                0.0495762825012207, 0.10900187492370605]

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(samples1, query_times1, marker='o', label="Brute Search")
plt.plot(samples2, query_times2, marker='s', label="SimHash")

# Add labels, title, legend, and grid
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("Query Time (s)", fontsize=12)
plt.title("Comparison of Two Retrival Methods", fontsize=14, fontweight='bold')
plt.legend(title="Retrival Strategy", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()