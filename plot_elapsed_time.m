trial_num_sweep = round(exp(log(2) :  log(2) : log(512)));
channel_nums = [4 16 64 256];
wave_iter_num = 5;
iterN = 10;
durations = [2 4];
wave_num_range = [3 10];

figure();
for i_channel_num = 1 : length(channel_nums)
    channel_num = channel_nums(i_channel_num);
    for i_duration = 1 : length(durations)
        subplot(length(channel_nums), length(durations), (i_channel_num - 1) * length(durations) + i_duration);
        duration = durations(i_duration);
        load(sprintf('./elapsed time results/elapsed_times_dur%d_chn%d.mat', duration, channel_num));
        
        hold on;
        for i_alg = 1 : 2
            plot(log2(trial_num_sweep), log10(squeeze(mean(elapsed_times(i_alg, :, :), 3))), 'LineWidth', 2);
        end
        hold off;

        xlabel('log_2(# of trials)');
        xticklabel(exp(log(2) :  log(2) : log(512)));
        ylabel('elapsed time. log(sec)')
    end
end
