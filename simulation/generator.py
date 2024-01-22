from deepgapseq_trace_generator import trace_generator
import numpy as np
from tqdm import tqdm

generator = trace_generator(n_colors=2, 
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            export_mode="text_files",
                            reduce_memory=False,
                            )

training_data, training_labels, training_tmats = generator.generate_traces()