import numpy as np
import plotly.graph_objects as go

def visualize_single_trace(trace,normalize=True,label=None,predicted_states=None):
    if normalize:
        trace = trace - np.mean(trace)
        trace = trace / np.std(trace)
    fig = go.Figure()
    x = np.arange(trace.shape[0])
    fig.add_trace(go.Scatter(x=x, y=trace[:,0],line=dict(color='firebrick', width=1),name='Donor'))
    fig.add_trace(go.Scatter(x=x, y=trace[:,1],line=dict(color='cornflowerblue', width=1), name='Acceptor'))
    
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

def find_state_dwell_time(states,):