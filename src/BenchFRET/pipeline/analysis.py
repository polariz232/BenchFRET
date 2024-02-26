import numpy as np
import plotly.graph_objects as go
from BenchFRET.DeepLASI.wrapper import DeepLasiWrapper
import itertools
import pandas as pd

def visualize_single_trace(trace,normalize=True,separate_traces=False,label=None,predicted_states=None,E_FRET=True,dark_mode=True):
    print("Visualizing trace...")
    if normalize:
        trace = trace - np.mean(trace)
        trace = trace / np.std(trace)

    fig = go.Figure()
    x = np.arange(trace.shape[0])
    
    if trace.shape[1] == 2:
        fig.add_trace(go.Scatter(x=x, y=trace[:,0],line=dict(color='chartreuse', width=1),name='Donor'))
        fig.add_trace(go.Scatter(x=x, y=trace[:,1],line=dict(color='red', width=1), name='Acceptor'))
    
    if trace.shape[1] == 1:
        if dark_mode:
            fig.add_trace(go.Scatter(x=x, y=trace.ravel(),line=dict(color='yellow', width=1),name='E_FRET'))
        else:
            fig.add_trace(go.Scatter(x=x, y=trace.ravel(),line=dict(color='black', width=1),name='E_FRET'))
    if label is not None:
        label = (label - np.max(label)-2)
        fig.add_trace(go.Scatter(x=x, y=label,line=dict(color='gold', width=1), name='Fret State'))
        # for i in np.unique(label):
        #     fig.add_shape(type='line',x0=min(x),x1=max(x),y0=i,y1=i,line=dict(color='pink',width=1,dash='dash'))
    
    if predicted_states is not None:
        predicted_states = (predicted_states - np.max(predicted_states)-2)
        if dark_mode:
            fig.add_trace(go.Scatter(x=x, y=predicted_states,line=dict(color='yellow', width=1), name='Predicted State'))
        else:
            fig.add_trace(go.Scatter(x=x, y=predicted_states,line=dict(color='black', width=1), name='Predicted State'))
        # for i in np.unique(predicted_states):
        #     fig.add_shape(type='line',x0=min(x),x1=max(x),y0=i,y1=i,line=dict(color='lightgreen',width=1,dash='dash'))
   
    if E_FRET:
        E_FRET = trace[:,1]/(trace[:,0]+trace[:,1])
        E_FRET = np.clip(E_FRET,0,1)*3 -4
        fig.add_trace(go.Scatter(x=x, y=E_FRET, line=dict(color='deepskyblue', width=1), name='E_FRET'))
    
    if dark_mode:
        fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font=dict(color='white'))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')  
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')  

    fig.show()


def get_dwell_times(states, plot=True, plot_mode='probability',bin_size=10,all_together=False):
    """
    Calculate the dwell times for each state given state array of shape (None,1).
    and plot the histogram of the dwell times for each state.
    plot mode = 'probability' or ''(as for 'count' mode)
    """
    print("Calculating dwell times...")
    if all_together:
        states = np.array(states).ravel()
    dwell_times = {}
    current_state = states[0]
    dwell_start = 0

    # Initialize dwell_times dictionary with the first state
    dwell_times[current_state] = []

    for i in range(1, len(states)):
        if states[i] != current_state:
            # State has changed, calculate dwell time
            dwell_time = i - dwell_start
            dwell_times[current_state].append(dwell_time)
            
            # Update for next dwell period
            current_state = states[i]
            dwell_start = i
            
            # Ensure the state is in the dwell_times dictionary
            if current_state not in dwell_times:
                dwell_times[current_state] = []

    # add the last dwell period
    dwell_times[current_state].append(len(states) - dwell_start)
    
    if plot:
        print('Plotting histogram...')
        fig = go.Figure()
        max_time = max(max(times) for times in dwell_times.values() if times) + 0.5

        for state in dwell_times:
            fig.add_trace(go.Histogram(
                            x=dwell_times[state],
                            name=f'State {state}',
                            histnorm=plot_mode,
                            autobinx=False,
                            xbins=dict(start=0.5,end=max_time,size=bin_size)
                            ))
        
        fig.update_layout(
            title='Histogram of Dwell Times for Each State',
            xaxis_title='Dwell Time',
            yaxis_title=plot_mode,
            barmode='overlay'
        )
        fig.update_traces(opacity=0.75)
        fig.show()

    return dwell_times   
    
def plot_FRET_efficiency_histogram(traces=None,true_states=None):
    
    print("Plotting FRET efficiency histogram...")
    if type(traces) is not np.ndarray:
        traces = np.array(traces)
    if len(traces.shape) == 3 and traces.shape[2] == 2:
        traces = traces.reshape(-1,2)
        E_FRET = traces[:,1]/(traces[:,0]+traces[:,1])
    elif len(traces.shape) == 3 and traces.shape[2] == 1:
        traces = traces.reshape(-1,1)
        E_FRET = traces.ravel()
    elif len(traces.shape) == 2:
        E_FRET = traces[:,1]/(traces[:,0]+traces[:,1])
    else:
        E_FRET = traces
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=E_FRET,name='E_FRET_observed',histnorm='probability'))
    if true_states is not None:
        fig.add_trace(go.Histogram(x=true_states,name='E_FRET_true',histnorm='probability'))
    fig.update_layout(
        title='Histogram of FRET Efficiency',
        xaxis_title='E_FRET',
        yaxis_title='Probability')
    fig.show()

def get_performance(n_states,detected_states,labels,verbose=True):

    scores = []
    aligned_labels = []
    for k in range(len(labels)):
        score, aligned_label = get_single_trace_score(n_states,detected_states[k], labels[k])
        scores.append(score)
        aligned_labels.append(aligned_label)
    average = sum(scores)/len(scores)
    if verbose:
        print(f'accuracy for individual traces is {scores}')
        print(f'overall average accuracy is {average}')
    return average, scores, aligned_labels

def get_single_trace_score(n_states,predicted_states, labels, verbose=False, best_only=True): 
    score = []
    permutation_of_labels = permute_array_values(labels, n_states)
    for perm in permutation_of_labels:
        mark = 0
        for i in range(predicted_states.shape[0]):
            if predicted_states[i] == perm[i]:
                mark += 1
        mark = mark/len(predicted_states)
        score.append(mark)
    if best_only:
        best = np.argmax(score)
        score = score[best]
        aligned_labels = permutation_of_labels[best]    
    if verbose:
        print(f'individual trace best score is : {score}')    
    return score, aligned_labels

def permute_array_values(arr, n):
    # Generate all permutations of the range of values
    value_permutations = itertools.permutations(range(n))

    # Initialize an empty list to store the permuted arrays
    permuted_arrays = []

    for perm in value_permutations:
        # Create a mapping from original values to permuted values
        mapping = np.array(perm)

        # Apply the mapping to the array
        permuted_array = mapping[arr.astype(int)]
        permuted_arrays.append(permuted_array)

    return permuted_arrays

def draw_confusion_matrix(confusion_matrix):

    plt.imshow(confusion_matrix, cmap='Blues')
    # Add percentage on each block
    total = np.sum(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[0]):            
            plt.annotate(f'{confusion_matrix[i,j]:.2f}', (j, i), ha='center', va='center')

    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if confusion_matrix.shape[0] == 2:
        plt.xticks([0, 1], ['Class 0', 'Class 1'])
        plt.yticks([0, 1], ['Class 0', 'Class 1'])
    elif confusion_matrix.shape[0] == 3:
        plt.xticks([0, 1, 2], ['Class 0', 'Class 1', 'Class 2'])
        plt.yticks([0, 1, 2], ['Class 0', 'Class 1', 'Class 2'])
    else:
        plt.xticks([0, 1, 2, 3], ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        plt.yticks([0, 1, 2, 3], ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        
    plt.title('Aggregate Confusion Matrix')
    plt.show()

def draw_performance_plot(df, to_compare='average_precision',save_path=None):
    if type(to_compare) is not list:
        fig = go.Figure(data=[go.Scatter3d(
            x=df['noise'],
            y=df['trans_rate'],
            z=df[to_compare],
            mode='markers',
            marker=dict(
                size=5,
                color=pd.to_numeric(df['noise'], errors='coerce'),  # Set color based on the 'noise' column
                colorscale='Viridis',  # Choose a colorscale
                opacity=0.8
            )
        )])

        # Set the layout of the graph
        fig.update_layout(
            scene=dict(
                xaxis_title='Noise',
                yaxis_title='Transition Rate',
                zaxis_title= to_compare
            )
        )

    else:
        fig = go.Figure()
        colours = ['blue','red','green','yellow']
        for i,metric in enumerate(to_compare):
            fig.add_trace(go.Scatter3d(
                x=df['noise'],
                y=df['trans_rate'],
                z=df[metric],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colours[i],
                    opacity=0.8
                ),
                name=metric  # Specify the name for the trace
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title='Noise',
                yaxis_title='Transition Rate',
            )
        )

    # Show the graph
    fig.show()
    if save_path is not None:
        fig.write_html(save_path)

def draw_compare_plot(df_hmm, df_deep, save_path=None,to_compare='precision',names=['input1','input2']):
    '''
    compare hmm and deeplasi performance (accuracy, precision, recall) in 3D plot
    '''
    fig = go.Figure()

    # Add data from df_hmm
    fig.add_trace(go.Scatter3d(
        x=df_hmm['noise'],
        y=df_hmm['trans_rate'],
        z=df_hmm[to_compare],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            # color=pd.to_numeric(df_hmm['noise'], errors='coerce'),  # Set color based on the 'noise' column
            # colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        ),
        name=names[0]  # Specify the name for the trace
    ))

    # Add data from df_deep
    fig.add_trace(go.Scatter3d(
        x=df_deep['noise'],
        y=df_deep['trans_rate'],
        z=df_deep[to_compare],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            # color=pd.to_numeric(df_deep['noise'], errors='coerce'),  # Set color based on the 'noise' column
            # colorscale='redor',  # Choose a colorscale
            opacity=0.8
        ),
        name=names[1]  # Specify the name for the trace
    ))

    # Set the layout of the graph
    fig.update_layout(
        scene=dict(
            xaxis_title='Noise',
            yaxis_title='Transition Rate',
            zaxis_title= to_compare
        )
    )

    # Show the graph
    fig.show()
    
    if save_path is not None:
        fig.write_html(save_path)
