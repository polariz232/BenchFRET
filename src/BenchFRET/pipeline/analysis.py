import numpy as np
import plotly.graph_objects as go

def visualize_single_trace(trace,normalize=True,label=None,predicted_states=None):
    print("Visualizing trace...")
    if normalize:
        trace = trace - np.mean(trace)
        trace = trace / np.std(trace)
    fig = go.Figure()
    x = np.arange(trace.shape[0])
    if trace.shape[1] == 2:
        fig.add_trace(go.Scatter(x=x, y=trace[:,0],line=dict(color='firebrick', width=1),name='Donor'))
        fig.add_trace(go.Scatter(x=x, y=trace[:,1],line=dict(color='cornflowerblue', width=1), name='Acceptor'))
    if trace.shape[1] == 1:
        fig.add_trace(go.Scatter(x=x, y=trace.ravel(),line=dict(color='black', width=1),name='E_FRET'))

    if label is not None:
        label = (label - np.max(label)-2)
        fig.add_trace(go.Scatter(x=x, y=label,line=dict(color='gold', width=1), name='Fret State'))
        for i in np.unique(label):
            fig.add_shape(type='line',x0=min(x),x1=max(x),y0=i,y1=i,line=dict(color='pink',width=1,dash='dash'))
    
    if predicted_states is not None:
        predicted_states = (predicted_states - np.max(predicted_states)-2)
        fig.add_trace(go.Scatter(x=x, y=predicted_states,line=dict(color='green', width=1), name='Predicted State'))
        for i in np.unique(predicted_states):
            fig.add_shape(type='line',x0=min(x),x1=max(x),y0=i,y1=i,line=dict(color='lightgreen',width=1,dash='dash'))
    # E_FRET = trace[:,1]/(trace[:,0]+trace[:,1])
    # fig.add_trace(go.Scatter(x=np.arange(trace.shape[0]), y=E_FRET, mode='markers', name='E_FRET'))
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
    
def plot_FRET_efficiency_histogram(traces):
    
    print("Plotting FRET efficiency histogram...")
    if type(traces) is not np.ndarray:
        traces = np.array(traces)
    if len(traces.shape) == 3 and traces.shape[2] == 2:
        traces = traces.reshape(-1,2)
        E_FRET = traces[:,1]/(traces[:,0]+traces[:,1])
    elif len(traces.shape) == 3 and traces.shape[2] == 1:
        traces = traces.reshape(-1,1)
        E_FRET = traces.ravel()
    else:
        E_FRET = traces[:,1]/(traces[:,0]+traces[:,1])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=E_FRET,name='E_FRET',histnorm='probability'))
    fig.update_layout(
        title='Histogram of FRET Efficiency',
        xaxis_title='E_FRET',
        yaxis_title='Probability')
    fig.show()

def get_score(labels, detected_states):

    scores = []
    for k in range(len(labels)):
        score = 0
        for i in range(labels[k].shape[0]):
            if labels[k][i] == detected_states[k][i]:
                score += 1
        scores.append(score/labels[k].shape[0])
    average = sum(scores)/len(scores)
    print(f'accuracy for individual traces is {scores}')
    print(f'overall average accuracy is {average}')
    return scores, average