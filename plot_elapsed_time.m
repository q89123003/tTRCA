alg_names = {'original', 'proposed'};

trial_num_sweep = round(exp(log(2) :  log(2) : log(1024)));
channel_nums = [4 16 64];
fs = 500;
durations = [2 4 8 16];

% colors = cell(1, 2);
% colors{1} = zeros(length(channel_nums), 3);
% colors{1}(:, 1) = linspace(1, 0.6, length(channel_nums));
% colors{1}(:, 2:3) = 0.2;
% colors{2} = zeros(length(channel_nums), 3);
% colors{2}(:, 1:2) = 0.2;
% colors{2}(:, 3) = linspace(1, 0.6, length(channel_nums));
colors = [1 0 0; 0 0 1];
markers = {'o', '^', 's'};
line_styles = {'-', '--'}; %, '-.', ':'};

ytclabels = {};
ytcs = 10.^(-2:1:7);
for i = 1 : length(ytcs)
    ytclabels = [ytclabels ['10^{' num2str(log10(ytcs(i))) '}']];
end

f = figure();
f.Position = [100 100 1600 500];
tiledlayout(1, length(durations));
avg_slopes = zeros(2, length(durations), length(channel_nums));

for i_duration = 1 : length(durations)
    duration = durations(i_duration);
    nexttile;
    legends = {};
    ps = [];
    
    for i_channel_num = 1 : length(channel_nums)
        channel_num = channel_nums(i_channel_num);
        %subplot(length(channel_nums), length(durations), (i_channel_num - 1) * length(durations) + i_duration);
        
        load(sprintf('./elapsed time results/elapsed_times_dur%d_chn%d.mat', duration, channel_num));
        
        hold on;
        for i_alg = 1 : 2
            tmp_mean_time = squeeze(mean(elapsed_times(i_alg, :, :), 3));
            %p = plot(log2(trial_num_sweep), log10(tmp_mean_time) + 3, 'LineWidth', 2, 'Color', colors{i_alg}(i_channel_num, :));
            p = plot(log2(trial_num_sweep), log10(tmp_mean_time) + 3, ...
                'LineWidth', 2, 'Color', colors(i_alg, :), 'Marker', markers{i_channel_num}, 'MarkerSize', 9, 'LineStyle', line_styles{i_alg});
            %fprintf('Alg %d, slope: %f\n', i_alg, ...
            %    (log10(tmp_mean_time(end)) - log10(tmp_mean_time(1))) ...
            %    / (log2(trial_num_sweep(end)) - log2(trial_num_sweep(1))));
            ps = [ps p];
            legends = [legends sprintf('%s - %d channels', alg_names{i_alg}, channel_num)];
            
            slope_sum = 0;
            for i = 2 : length(tmp_mean_time)
                slope_sum = slope_sum + (log10(tmp_mean_time(i)) - log10(tmp_mean_time(i - 1))) / (log2(trial_num_sweep(i)) - log2(trial_num_sweep(i - 1)));
            end
            slope_mean = slope_sum / (length(tmp_mean_time) - 1);
            
            avg_slopes(i_alg, i_duration, i_channel_num) = slope_mean;
            %text(10+0.3, log10(tmp_mean_time(end))+3, sprintf('%.2f', slope_mean));
        end
        hold off;
    end
    
    title(sprintf('N_s = %d sec', duration));
    
    xlabel('number of trials (N_t)');
    xlim([0, 11]);
    xticks(log2(trial_num_sweep));
    xticklabels(strsplit(num2str(trial_num_sweep)));
    
    if i_duration == 1
        ylabel('elapsed time (ms)')
    end
    yticks(-2:1:7);
%     yticklabels(strsplit(num2str(10.^(-2:1:7))));
    yticklabels(ytclabels);
    ylim([-2, 7]);
    set(gca, 'FontSize', 15);
end

mean_avg_slopes = mean(avg_slopes, 3);

sorted_ps(1:3) = ps(1:2:5);
sorted_ps(4:6) = ps(2:2:6);
sorted_legends(1:3) = legends(1:2:5);
sorted_legends(4:6) = legends(2:2:6);
lgd = legend(sorted_ps, sorted_legends);
lgd.Layout.Tile = 'east';
set(gca, 'FontSize', 15);

%%
table_string = '';
for i_channel_num = 1 : length(channel_nums)
    channel_num = channel_nums(i_channel_num);
    
    table_string = [table_string num2str(channel_nums(i_channel_num))];
    for i_duration = 1 : length(durations)
        duration = durations(i_duration);    
        load(sprintf('./elapsed time results/component_diff_dur%d_chn%d.mat', duration, channel_num));
        tmp_diffs = mean2(component_diffs);
        
        table_string = [table_string ' & '];
        table_string = [table_string sprintf('%.2d', tmp_diffs)];
            
    end
    table_string = [table_string ' \\'];  
end
