nstates = 10;
ntraces = 10;
trace_length = 300;

data = {};
iter = 1;

for state = 2:nstates
    for n = 1:ntraces

        dat = {};
        dat{end+1} = state * ones(trace_length,1);
        dat{end+1} = n * ones(trace_length,1);
        dat{end+1} = 0 * ones(trace_length,1);
        dat{end+1} = 0 * ones(trace_length,1);

        data{end+1} = cat(2, dat{:});
        iter = iter + 1;

    end
end

traces = cat(1,data{:});