import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

def cluster_plot(data_text, n_clusters):
    try:
        # Parse user input
        # Expect input like: [[1,30],[2,32],[1.5,31],[5,60]]
        X = np.array(eval(data_text))
        if X.ndim != 2 or X.shape[1] != 2:
            return " Please enter a 2D list with two columns (e.g. [[1,30],[2,32],[3,35]])", None

        # Agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)

        # Plot scatter
        plt.figure(figsize=(6,4))
        plt.scatter(X[:,0], X[:,1], c=labels, s=80, cmap='rainbow', edgecolors='k', alpha=0.8)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Agglomerative Clustering")
        plt.grid(True)

        # Return results
        return f" Cluster labels: {labels.tolist()}", plt.gcf()

    except Exception as e:
        return f"Error: {e}", None

# Gradio UI
demo = gr.Interface(
    fn=cluster_plot,
    inputs=[
        gr.Textbox(
            label="Enter 2D Array (Experience vs Salary)",
            placeholder="Example: [[1,30],[2,32],[1.5,31],[5,60],[10,100]]"
        ),
        gr.Slider(2, 10, value=3, step=1, label="Number of Clusters")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Plot(label="Cluster Plot")
    ],
    title="Agglomerative Clustering Visualizer",
    description="Enter a 2D array of points and choose the number of clusters to see the scatter plot."
)

if __name__ == "__main__":
    demo.launch()