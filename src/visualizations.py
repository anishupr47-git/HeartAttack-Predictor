import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

#We now visualize and see and do every elements
def plot_confusion_matrix(cm, model_name):
    """
    confusion matrix shows 4 columns or more as needed and show true positive and true negative with false positive and negative as well
    """
    #flip to make it nice on screen
    z= cm[::-1]

    #label x for x axis
    x=['Predicted: 0(No Disease)', 'Predicted: 1(Disease)']

    #label y for y axis
    y=['Actual:1(Disease)','Actual:0(No Disease)']

    #create a heatmap
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)

    #we add title to make it look clean
    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        margin=dict(t=50,l=100,b=50,r=50),
        height=400,
        width=500
    )
    return fig

def plot_metrics_comparison(metrics_dict):
    """
    Ploat a massive bar chart compairing all 12 models
    """
    #extract all models
    models= list(metrics_dict.keys())

    #extract the score for each model in the list
    accuracies = [metrics_dict[m]['Accuracy'] for m in models]
    f1s= [metrics_dict[m]['F1-Score'] for m in models]
    aucs = [metrics_dict[m]['ROC-AUC'] for m in models]

    #build bar chart
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracies, marker_color='#1f77b4'),
        go.Bar(name='F1-Score', x=models, y=f1s, marker_color='#ff7f0e'),
        go.Bar(name='ROC-AUC', x=models, y=aucs, marker_color='#2ca02c')
    ])

    #barmode=group means the bars will stand side by side
    fig.update_layout(
        barmode='group',
        title="Comprehensive Architecture Comparison",
        xaxis_title="Model Architecture",
        yaxis_title="Score (0 to 1)",
        height=500,
        template='plotly_white'
    )
    return fig

def plot_eda_target_distribution(target_dist):
    """
    Plots a piece chart of target class distribution
    """
    #Our labels for the piechart
    labels = ['No Heart Disease (0)', 'Heart Disease (1)']

    #the actual percentage
    values = [target_dist.get(0,0), target_dist.get(1,0)]

    #pie chart creation now
    fig = px.pie(values=values, names=labels, title='Target Class Distribution', hole=0.4, color_discrete_sequence=['#17becf', '#e377c2'])

    return fig

def plot_feature_correlation(df_head):
    """
    Plots heatmap for numerical column
    """
    #we look at numerical column
    nums = df_head.select_dtypes(include=[np.number])

    #corr() correlations
    corr= nums.corr()

    #show
    fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale='RdBu_r', aspect="auto")

    return fig

    