function python_run_ebayes(self,  min_states, max_states)

    self.controls.min_states = double(min_states);
    self.controls.max_states = double(max_states);

    self.run_ebayes()

end