function [traces] = python_export_traces(parent, min_states, max_states)

    states = num2cell(parent.controls.min_states:parent.controls.max_states);

    min_states = int32(min_states);
    max_states = int32(max_states);

    series = parent.series;

    data = {};
    iter = 1;

    for state = states{1}:states{end}
        for n = 1:length(series)
            if ~series(n).exclude

                analysis = parent.analysis(state);

                viterbi_state = analysis.viterbi(n).state(:);
                viterbi_mean = analysis.viterbi(n).mean(:);

                range = series(n).crop.min:series(n).crop.max;
    
                dat = {};

                dat{end+1} = double(state) * ones(length(viterbi_state),1);
                dat{end+1} = double(n) * ones(length(viterbi_state),1);

                dat{end+1} = viterbi_state;
                dat{end+1} = viterbi_mean;

                catdat = cat(2, dat{:});

                data{end+1} = cat(2, dat{:});

                iter = iter + 1;
            end
        end
    end
    traces = cat(1, data{:});
end