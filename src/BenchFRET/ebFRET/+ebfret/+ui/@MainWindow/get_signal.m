function x = get_signal(self, series_index)
    if nargin < 2
        series_index = 1:length(self.series);
    end
    x = {};
    for n = series_index
        s = self.series(n);
        if ~s.exclude
            x{n} = s.signal(s.crop.min:s.crop.max);

%             disp(x{n})

%             disp("xxx")
%             disp(self.controls.clip.min)
%             disp(self.controls.clip_max)
%             disp("xxx")
%               self.controls.clip.max = 500
%               self.controls.clip.min = 200

            % clip to specified range
            if self.controls.clip.min < inf
                clip = find(x{n} < self.controls.clip.min);
                x{n}(clip) = self.controls.clip.min;
            end
            if self.controls.clip.max < inf
                clip = find(x{n} > self.controls.clip.max);
                x{n}(clip) = self.controls.clip.max;
            end
        else
            x{n} = [];
        end
    end
    if isscalar(series_index) 
        if ~isempty(x)
            x = x{n};
        else
            x = [];
        end
    end