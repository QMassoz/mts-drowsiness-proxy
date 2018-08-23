subject_range = [1:6 8 10 12:23 25:30 33:35];
test_range = [2 5 8];
histogram_nbins = 0:0.1:1;
histogram_nsizes = [5 15 30 60]; % [s]
min_step = 1;

X = [];
X_subject = [];
X_test = [];
Y = [];

% compute mean of reciprocal of RTs for each subject
meanirt = zeros(1,max(subject_range));
for subject = subject_range
    if ~exist(['rt/' num2str(subject) '-2.txt'], 'file') % PVT1 must exist for RT normalization
        continue;
    else
        RT_PVT1 = load_testRT(subject,2);
        meanirt(1,subject) = mean(squeeze(1./RT_PVT1(:,2)));
    end
end

% extract features and median RT
for subject = subject_range
    if ~exist(['rt/' num2str(subject) '-2.txt'], 'file') % PVT1 must exist for RT normalization
        continue;
    end
    for test = test_range
        if ~exist(['eld-seq/' num2str(subject) '-' num2str(test) '.csv'], 'file')
            continue;
        end
        disp(['Subject ' num2str(subject) ' - test ' num2str(test)]);
        
        % Load timestamps + RT
        timestamps = ((1:18000)*1000/30)';
        RT = load_testRT(subject,test);
        
        % Normalize RT
        RT(:,2) = 1 ./ (1./RT(:,2) - meanirt(subject) + mean(nonzeros(meanirt)));
        
        % Analyse eye
        [eye_dist, baseline, ~, ~, ~, ~, blinks70_info] = extract_eye_parameters(subject,test);
        blinks70_info = blinks70_info(:,[1 2 3 4 5 7]);
        s = (eye_dist ./ baseline)';
        s(s>1) = 1;
        
        %-------------------------%
        %- FEATURES PER STIMULUS -%
        %-------------------------%
        for q=1:size(RT,1)
            if RT(q,1) < max(histogram_nsizes)*1000 % do not include samples before the first minute
                continue;
            end
            features = [];
            features_name = {};
            
            for h=1:length(histogram_nsizes)
                % 1-- blink features
                b_data = table2array(blinks70_info(:,2:end));
                wB = getWindowMatrix(b_data, [0 -histogram_nsizes(h) RT(q,1)], [0 0 RT(q,1)]); % end of blink within minute
                if isempty(wB)
                    wB = zeros(1,size(b_data,2)-1);
                end
                features = [features mean(wB,1)];
                blinks_fname =  [blinks70_info.Properties.VariableNames(3:end)];
                for b=1:length(blinks_fname)
                    blinks_fname{b} = ['cf' num2str(histogram_nsizes(h)) 's_' blinks_fname{b}];
                end
                features_name = [features_name blinks_fname];
                
                features = [features sum(wB(:,1)>500)];
                features_name = [features_name ['f' num2str(histogram_nsizes(h)) 's_Nusleep']];
                
                wS = getWindowMatrix([timestamps s], [0 -histogram_nsizes(h) RT(q,1)], [0 0 RT(q,1)]);
                features = [features mean(wS<0.7)];
                features_name = [features_name ['f' num2str(histogram_nsizes(h)) 's_PERCLOS70']];
                
            end
            
            % add sample
            if all(~isnan(features))
                X(end+1,:) = features;
                X_subject(end+1) = subject;
                X_test(end+1) = test;
                Y(end+1,:) = [...
                    1./mean(1./getWindowMatrix(RT, [0  -1 RT(q,1)], [0 1 RT(q,1)]))...
                    1./mean(1./getWindowMatrix(RT, [0 -15 RT(q,1)], [0 5 RT(q,1)]))...
                    1./mean(1./getWindowMatrix(RT, [0 -30 RT(q,1)], [0 5 RT(q,1)]))...
                    1./mean(1./getWindowMatrix(RT, [0 -60 RT(q,1)], [0 5 RT(q,1)]))...
                    ];
            end
        end 
    end
end

%% CLASSIFICATION SVM
for j=1:4
    thr1 = 400.37; %400->400.37 because of decimal differences between matlab and torch
    thr2 = 500;
    i = Y(:,j)<thr1 | Y(:,j)>thr2;
    x = X(i,:); 
    y = Y(i,j);
    x_subject = X_subject(i);
    yt = zeros(size(y));
    yt(y>=thr2)=1; yt(y<thr1)=0;

    % linear
    C_range = [25:5:60];

    [Rval, param_opt, CMt] = SVM_crossvalidation_3sets(x,yt, x_subject, 'linear', {C_range});
    CM = sum(CMt,3);
    TPr = CM(2,2) / sum(CM(2,1:2));
    TNr = CM(1,1) / sum(CM(1,1:2));
    acc = (CM(1,1)+CM(2,2)) / sum(sum(CM));
    
    disp(['Timescale ' num2str(j) ': TNR=' num2str(TNr) ', TPR=' num2str(TPr) ', accuracy=' num2str(acc)])
end

