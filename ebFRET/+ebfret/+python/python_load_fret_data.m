function python_load_fret_data(self, file_name, dataDD, dataDA)

try     

    % Define the characters to choose from
    chars = ['A':'Z' 'a':'z' '0':'9'];
    
    clip_min = [];
    clip_max = [];

    series = struct([]);
    for n = 1:length(dataDD)

        % Generate random label
        indices = randi(numel(chars), [1, 24]);
        label = chars(indices);

        datDD = dataDD(n);
        datDA = dataDA(n);

        datDD = cell2mat(datDD{1}');
        datDA = cell2mat(datDA{1}');

        clip_min = [clip_min,min(datDD)];
        clip_max = [clip_max,max(datDD)];
        clip_min = [clip_min,min(datDA)];
        clip_max = [clip_max,max(datDA)];

        len_dat = numel(datDD);
        series(n).file = file_name;
        series(n).label = label;
        series(n).group = 'group1';
        series(n).time = int32(linspace(0, len_dat-1, len_dat))';
        series(n).donor = datDD;
        series(n).acceptor = datDA;
        series(n).signal = (series(n).acceptor + eps) ./ (series(n).acceptor + series(n).donor + eps);
        series(n).crop.min = 1;
        series(n).crop.max = length(series(n).time);
    end
    
    [series.exclude] = deal(false);
%     if isempty(self.series)
%         self.series = series;
%     else
%         self.series = cat(1, self.series(:), series(:));
%     end

    self.series = series;

    self.controls.clip.max = max(clip_max)*1.2;
    self.controls.clip.min = min(clip_min)*0.8;

%       disp(self.controls.clip.max)
%       disp(self.controls.clip.min)

%     disp(self.controls.min_states)
%     disp(self.controls.max_states)
%     disp(length(self.series))
%     save('C:\napari-gapseq\src\napari_gapseq\ebfret-gapseq\dataFile.mat', "series");
%     disp("saved")

    self.reset_analysis(self.controls.min_states:self.controls.max_states);
    self.set_control(...
        'series', struct(...
                    'min', 1, ...
                    'max', length(self.series), ...
                    'value', 1));
    self.set_control('ensemble', struct('value', self.controls.min_states));

    % this is to ensure analysis does not start immediately
    self.set_control('run_analysis', false);
    % this updates enabled/disabled status of the controls in the view menu
    self.set_control('show', struct());
    self.set_control('scale_plots', self.controls.scale_plots);
    % this does a replot for good measure
    self.refresh('ensemble', 'series');

catch self
    disp(['Error occurred: ', self.message]);

end

