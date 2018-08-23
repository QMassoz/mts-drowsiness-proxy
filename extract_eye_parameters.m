function [eye_dist, baseline, closing_samples, opening_samples, stretched_signal, blinks_info, blinks70_info] = extract_eye_parameters(subject, test)
%EXTRACT_EYE_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here

timestamps = (1:18000)*1000/30;
filename = ['eld-seq/' num2str(subject) '-' num2str(test) '.csv'];
data = table2array(readtable(filename,'Delimiter',','));

% -- I) eye_dist --
eye_dist = mean(data,2)';

% -- II) baseline --
fps = length(timestamps) / 600;
eye_median = 0.7*median(eye_dist);
baseline = zeros(size(eye_dist));

baseline(1) = mean([median(eye_dist(timestamps<1000)), max(eye_dist(timestamps<1000))]);
alpha_cst = 0.4;
if fps < 20
    alpha_cst = 2*alpha_cst-alpha_cst^2;  
end
for i=2:length(eye_dist)
    %deriv = 30*(eye_dist(i)-eye_dist(i-1)) / (timestamps(i)-timestamps(i-1));
    deriv = eye_dist(i)-eye_dist(i-1);
    dist = eye_dist(i)-baseline(i-1);
    
    alpha_deriv = exp(-15*deriv.^2);
    alpha_above = exp(-0.5*abs(max(0,dist)));
    alpha_below = exp(-2*abs(min(0,dist)));
    alpha_median = 0.5*(1+sign(eye_dist(i) - eye_median));
    alpha = alpha_cst * alpha_deriv * alpha_above * alpha_below * alpha_median;
    
    baseline(i) = (1-alpha)*baseline(i-1) + alpha*eye_dist(i);
end

% -- III) d1_norm & d2_norm --
s = eye_dist./baseline;
d1_bnorm = zeros(size(s));
% d1_fnorm = zeros(size(s));
% d1_cnorm = zeros(size(s));
% d2_cnorm = zeros(size(s));

% - backward
for i=2:length(s)
    d1_bnorm(i) = (s(i) - s(i-1)) / (timestamps(i)-timestamps(i-1));
end
% % - forward
% for i=1:length(s)-1
%     d1_fnorm(i) = (s(i+1) - s(i)) / (timestamps(i+1)-timestamps(i));
% end
% % - centered
% for i=2:length(s)-1
%     h = 0.5*(timestamps(i+1)-timestamps(i-1));
%     d1_cnorm(i) = (s(i+1)-s(i-1)) / (2*h);
%     d2_cnorm(i) = (s(i+1)-2*s(i)+s(i-1)) / h^2;
% end

d1_bnorm = 30*d1_bnorm;
% d1_fnorm = 30*d1_fnorm;
% d1_cnorm = 30*d1_cnorm;
% d2_cnorm = 30*d2_cnorm;

% IV) closing & opening
[closing_samples, opening_samples, blinks] = extract_eye_movements(s, d1_bnorm, 5e-3, -12e-3);

% V) streched_signal
stretched_signal = s;
for i=1:size(blinks,1)
    % strech closing
    idx = blinks(i,1):blinks(i,2);
    cs = s(idx);
    stretched_signal(idx) = max(cs)*(cs-min(cs))/(max(cs)-min(cs));
    % strech opening
    idx = blinks(i,3):blinks(i,4);
    os = s(idx);
    stretched_signal(idx) = max(os)*(os-min(os))/(max(os)-min(os));
    % strech closed
    stretched_signal(blinks(i,2):blinks(i,3)) = 0;
end

% VI) blinks_info
[blinks_info, blinks70_info] = analyse_blinks(blinks, s, d1_bnorm, timestamps);
blinks_info = filter_invalid_blinks(blinks_info);
blinks70_info = filter_invalid_blinks(blinks70_info);
end

function input = filter_invalid_blinks(input)
durBlink = input{:,3};
durClosing = input{:,4};
durOpening = input{:,5};
durClosed = input{:,7};

idx = (durBlink > 5000) | (durClosing > 3000) | (durOpening > 3000) | (1.5*durBlink < durClosing+durOpening+durClosed);
input(idx,:) = [];

end

function [i70, x70, t70] = find_70(x,t,thr)
    if ~exist('thr','var')
        thr = 0.7;
    end
    x(x>1)=1;
    x_norm = (x-min(x))/(max(x)-min(x));
    
    for i=2:length(x_norm)
        if (x_norm(i-1) >= thr && x_norm(i) <= thr) || ...
           (x_norm(i-1) <= thr && x_norm(i) >= thr)    
            alpha = (thr-x_norm(i-1))/(x_norm(i)-x_norm(i-1));
            i70 = alpha*i+(1-alpha)*(i-1);
            x70 = alpha*x(i)+(1-alpha)*x(i-1);
            t70 = alpha*t(i)+(1-alpha)*t(i-1);
            return;
        end
    end
    error('impossible to find 70% in above x!');
end

function [blinks_info, blinks70_info] = analyse_blinks(blinks, s, d1, timestamps)
info_names = {'beginTime', 'endTime', 'durBlink', 'durClosing', 'durOpening', 'durClosed', 'dur10Closed', ...
    'avgSpeedClosing', 'avgSpeedOpening', 'maxSpeedClosing', 'maxSpeedOpening'};

blinks_info   = zeros(size(blinks,1), length(info_names));
blinks70_info = zeros(size(blinks,1), length(info_names));
for b=1:size(blinks,1)
    row   = zeros(1,length(info_names));
    row70 = zeros(1,length(info_names));
    
    % find blinks70 limits
    idx = blinks(b,1):blinks(b,2);
    [c_i70, c_s70, c_t70] = find_70(s(idx), timestamps(idx));
    c_i70 = ceil(c_i70 + blinks(b,1));
    if c_i70 > blinks(b,2)
        c_i70 = blinks(b,2);
    end
    idx = blinks(b,3):blinks(b,4);
    [o_i70, o_s70, o_t70] = find_70(s(idx), timestamps(idx));
    o_i70 = floor(o_i70 + blinks(b,3));
    if o_i70 < blinks(b,3)
        o_i70 = blinks_(b,3);
    end
    
    % find blinks10 limits
    idx = blinks(b,1):blinks(b,2);
    [~, ~, c_t10] = find_70(s(idx), timestamps(idx), 0.1);
    idx = blinks(b,3):blinks(b,4);
    [~, ~, o_t10] = find_70(s(idx), timestamps(idx), 0.1);
    % beginTime
    row(1)   = timestamps(blinks(b,1));
    row70(1) = c_t70;
    % endTime
    row(2)   = timestamps(blinks(b,4));
    row70(2) = o_t70;
    % durBlink
    row(3)   = row(2) - row(1);
    row70(3) = row70(2) - row70(1);
    % durClosing
    row(4)   = timestamps(blinks(b,2)) - row(1);
    row70(4) = timestamps(blinks(b,2)) - row70(1);
    % durOpening
    row(5)   = row(2)   - timestamps(blinks(b,3));
    row70(5) = row70(2) - timestamps(blinks(b,3));
    % durClosed
    row(6)   = row(3)   - sum(row(4:5));
    row70(6) = row70(3) - sum(row70(4:5));
    % dur10Closed
    row(7)   = o_t10 - c_t10;
    row70(7) = row(7);
    % avgSpeedClosing
    row(8)   = mean(d1(blinks(b,1):blinks(b,2)));
    row70(8) = mean(d1(c_i70:blinks(b,2)));
    % avgSpeedOpening
    row(9)   = mean(d1(blinks(b,3):blinks(b,4)));
    row70(9) = mean(d1(blinks(b,3):o_i70));
    % maxSpeedClosing
    row(10)   = min(d1(blinks(b,1):blinks(b,2)));
    row70(10) = min(d1(c_i70:blinks(b,2)));
    % maxSpeedOpening
    row(11)   = max(d1(blinks(b,3):blinks(b,4)));
    row70(11) = max(d1(blinks(b,3):o_i70));
    
    % add row
    blinks_info(b,:) = row;
    blinks70_info(b,:) = row70;
end

blinks_info = array2table(blinks_info);
blinks_info.Properties.VariableNames = info_names;
blinks70_info = array2table(blinks70_info);
blinks70_info.Properties.VariableNames = info_names;
end
