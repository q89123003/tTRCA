function f = bar_error_pval(results, colors, pairs, legends, positions)

[case_num point_num group_num] = size(results);

if exist('positions', 'var')
    f = figure('Renderer', 'painters', 'Position', positions);
else
    f = figure();
end


hold on;

b = bar(squeeze(mean(results, 2)));

pause(0.1);

if exist('colors','var')
    for i_group = 1 : group_num
        b(i_group).FaceColor = colors(i_group, :);
    end
end

for i_group = 1 : group_num
    tmp_m = squeeze(mean(results(:, :, i_group), 2));
    tmp_s = squeeze(std(results(:, :, i_group), 1, 2));
    tmp_x = b(i_group).XData + b(i_group).XOffset;
    
    errorbar(tmp_x, tmp_m, tmp_s / sqrt(point_num), 'k.');
end

if exist('pairs','var') && pairs(1) ~= -1
    for i_pair = 1 : size(pairs, 1)
        
        pair = pairs(i_pair, :);
        x1 = b(pair(1)).XData + b(pair(1)).XOffset;
        x2 = b(pair(2)).XData + b(pair(2)).XOffset;
        
        for i = 1 : case_num
            [p, ~] = signrank(squeeze(results(i, :, pair(1))), squeeze(results(i, :, pair(2))));
            
            if p < 0.05
                x1_tmp = x1(i);
                x2_tmp = x2(i);
                
                y_offset = 1 + 0.05 * (i_pair - 1);
                
                plot([x1_tmp x1_tmp], [y_offset y_offset+0.025], 'black');
                plot([x2_tmp x2_tmp], [y_offset y_offset+0.025], 'black');
                plot([x1_tmp x2_tmp], [y_offset+0.025 y_offset+0.025], 'black');
                text((x1_tmp+x2_tmp)/2, y_offset+0.04, '*', 'HorizontalAlignment', 'center', 'FontSize', 20);
            end
        end
    end
end

if exist('legends','var')
    legend(b, legends, 'Position', [0.8 0.1 0.2 0.1]);
end

end