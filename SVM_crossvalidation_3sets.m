function [Rcross, param_opt, ConfusionMatrices, prdctIdx, prdctY] = SVM_crossvalidation_3sets(X, Y, partitioning, method, param_range)
%SVM_CROSSVALIDATION returns (1) minimal cross-validation errors and (2) the optimal parameters using
% cross-validation and exhaustive search in a given space (3) confusion matrices
%
% @param X: (Nxa) features vectors
% @param Y: (Nx1) regression targets
% @param partitioning: subjects_id for cross-validation
%           (array) v -> cross-validation vector
% @param method: (string) 'linear' or 'rbf'
% @param param_range: (cell) {C_range} if 'linear'
%                            {C_range, gamma_range} if 'rbf'
%
% @output param_opt: [C_opt] if 'linear'
%                    [C_opt gamma_opt] if 'rbf'
% @output Rcross: (vector kx1) TS accuracies
% @output ConfusionMatrices: (CxCxk) k Confusion Matrices (CxC) (line=ground truth, column=prediction)
% @output prdctIdx: (cell) indices of all the samples in TSs
% @output prdctY: (cell) y_hat of all the samples in TSs

% -- Partitioning multiple LS/VS/TS--
% Random k-fold cross-validation
Nsamples = size(X,1);
Rcross = [];
param_opt = [];

% specified-fold cross-validation
k = length(unique(partitioning));
partTS = zeros(Nsamples,k);
u = unique(partitioning);
for i=1:k
    partTS(partitioning==u(i),i) = 1;
end

% -- search optimal value --
isTypeLinear = strcmp(method, 'linear') || strcmp(method, 'ranklinear');
k = size(partTS,2);
Nclasses = length(unique(Y));
classesUnique = unique(Y)';
samplesIdx = 1:Nsamples;

% Init
if isTypeLinear
    p1_range = param_range{1};
    p2_range = 1;
    Rval = zeros(length(p1_range), k);
    CM = zeros(length(p1_range), Nclasses,Nclasses,k);
    fprintf(1, '[SVM_crossvalidation] linear: ');
else
    p1_range = param_range{1};
    p2_range = param_range{2};
    Rval = zeros(length(p1_range), length(p2_range), k);
    CM = zeros(length(p1_range), length(p2_range), Nclasses,Nclasses,k);
    fprintf(1, '[SVM_crossvalidation] rbf: ');
end
prdctIdx = [];
prdctY   = [];

% Search
nMsg = 0;

for i=1:size(partTS,2)
    if length(p1_range)~=1 || length(p2_range)~=1
        partVS = partTS;
        partVS(partVS(:,i)==1,:) = -1; % remove samples of the TS
        partVS(:,i) = []; % remove fold since only zero's
        for j=1:size(partTS,2)-1             
            ls_X = X(partVS(:,j)==0,:);
            ls_Y = Y(partVS(:,j)==0,:);
            vs_X = X(partVS(:,j)==1,:);
            vs_Y = Y(partVS(:,j)==1,:);   

            [ls_X, ~, scale_function] = SVM_scale(ls_X, [0,1]);
            vs_X = scale_function(vs_X);

            % Init options
            if isTypeLinear
                base_options = ['-s 2 -q']; % liblinear
            else
                base_options = ['-s 0 -t 2 -q'];
            end

            % Weight classes (doesn't work with rank learning...)
            nClasses = histc(ls_Y,classesUnique);
            for c=1:length(classesUnique)
                base_options = [base_options ' -w' num2str(classesUnique(c)) ' ' num2str(1/nClasses(c))];
            end
            % Find best hyperparam 
            for p1_idx = 1:length(p1_range)
                for p2_idx = 1:length(p2_range)
                    % - Configure options
                    if isTypeLinear
                        msg = ['cross-validation (' num2str(i) '/' num2str(size(partTS,2)) ') ' num2str(j) '/' num2str(size(partVS,2)) ' C=' num2str(p1_range(p1_idx))];
                        fprintf(repmat('\b',1,nMsg)); fprintf(msg); nMsg=numel(msg);
                        options = [base_options ' -c ' num2str(p1_range(p1_idx))];
                    else
                        msg = ['cross-validation (' num2str(i) '/' num2str(size(partTS,2)) ') ' num2str(j) '/' num2str(size(partVS,2)) ' C=' num2str(p1_range(p1_idx)) ', gamma=' num2str(p2_range(p2_idx))];
                        fprintf(repmat('\b',1,nMsg)); fprintf(msg); nMsg=numel(msg);
                        options = [base_options ' -c ' num2str(p1_range(p1_idx)) ' -g ' num2str(p2_range(p2_idx))];
                    end

                    % - Learn
                    if isTypeLinear
                        svrModel = liblinear_train(ls_Y,sparse(ls_X),options); %liblinear
                        [y_hat, ~, ~] = liblinear_predict(vs_Y,sparse(vs_X),svrModel,'-q');
                    else
                        svrModel = svmtrain(ls_Y,ls_X,options); %libsvm
                        [y_hat, ~, ~] = svmpredict(vs_Y,vs_X,svrModel, '-q');
                    end

                    % - Save
                    rvalFunction = @(CM) nanmean(diag(CM)./sum(CM,2));
                    if isTypeLinear
                        for m=1:length(classesUnique)
                            for n=1:length(classesUnique)
                                nb = sum(y_hat == classesUnique(n) & vs_Y == classesUnique(m));
                                CM(p1_idx, m,n,j) = nb;
                            end
                        end
                        Rval(p1_idx,j) = rvalFunction(squeeze(CM(p1_idx,:,:,j)));
                    else
                        for m=1:length(classesUnique)
                            for n=1:length(classesUnique)
                                nb = sum(y_hat == classesUnique(n) & vs_Y == classesUnique(m));
                                CM(p1_idx,p2_idx, m,n,j) = nb;
                            end
                        end
                        Rval(p1_idx,p2_idx,j) = rvalFunction(squeeze(CM(p1_idx,p2_idx,:,:,j)));
                    end
                end
            end
        end

        % Return optimum parameters
        if isTypeLinear
            SCM = sum(CM,4);
            meanRval = zeros(length(p1_range));
            for p1_idx = 1:length(p1_range)
                meanRval(p1_idx) = rvalFunction(squeeze(SCM(p1_idx,:,:)));
            end
            [~, idx] = max(meanRval(:));
            [i1] = ind2sub(size(meanRval), idx);
            C_opt = p1_range(i1);
            param_opt = [param_opt; C_opt];
        else
            SCM = sum(CM,5);
            meanRval = zeros(length(p1_range),length(p2_range));
            for p1_idx = 1:length(p1_range)
                for p2_idx = 1:length(p2_range)
                    meanRval(p1_idx,p2_idx) = rvalFunction(squeeze(SCM(p1_idx,p2_idx,:,:)));
                end
            end
            [~, idx] = max(meanRval(:));
            [i1, i2] = ind2sub(size(meanRval), idx);
            C_opt = p1_range(i1);
            gamma_opt = p2_range(i2);
            param_opt = [param_opt; C_opt gamma_opt];
        end
    else
        if isTypeLinear
            msg = ['cross-validation (' num2str(i) '/' num2str(size(partTS,2)) ')  C=' num2str(p1_range(1))];
            fprintf(repmat('\b',1,nMsg)); fprintf(msg); nMsg=numel(msg);
            C_opt = p1_range(1);
            param_opt = [param_opt; C_opt];
        else
            msg = ['cross-validation (' num2str(i) '/' num2str(size(partTS,2)) ') C=' num2str(p1_range(1)) ', gamma=' num2str(p2_range(1))];
            fprintf(repmat('\b',1,nMsg)); fprintf(msg); nMsg=numel(msg);
            C_opt = p1_range(1);
            gamma_opt = p2_range(1);
            param_opt = [param_opt; C_opt gamma_opt];
        end
    end
    
    % train on whole LS and evaluate on TS
    ls_X = X(partTS(:,i)==0,:);
    ls_Y = Y(partTS(:,i)==0,:);
    ts_X = X(partTS(:,i)==1,:);
    ts_Y = Y(partTS(:,i)==1,:);
    
    [ls_X, ~, scale_function] = SVM_scale(ls_X, [0,1]);
    ts_X = scale_function(ts_X);
    
    if isTypeLinear
        base_options = ['-s 2 -q']; % liblinear
        nClasses = histc(ls_Y,classesUnique);
        for c=1:length(classesUnique)
            base_options = [base_options ' -w' num2str(classesUnique(c)) ' ' num2str(1/nClasses(c))];
        end
        options = [base_options ' -c ' num2str(C_opt)];
        svrModel = liblinear_train(ls_Y,sparse(ls_X),options); %liblinear
        [y_hat, acc, ~] = liblinear_predict(ts_Y,sparse(ts_X),svrModel,'-q');
    else
        base_options = ['-s 0 -t 2 -q'];
        nClasses = histc(ls_Y,classesUnique);
        for c=1:length(classesUnique)
            base_options = [base_options ' -w' num2str(classesUnique(c)) ' ' num2str(1/nClasses(c))];
        end
        options = [base_options ' -c ' num2str(C_opt) ' -g ' num2str(gamma_opt)];
        svrModel = svmtrain(ls_Y,ls_X,options); %libsvm
        [y_hat, acc, ~] = svmpredict(ts_Y,ts_X,svrModel, '-q');
    end
    Rcross = [Rcross; acc(1)];
    for m=1:length(classesUnique)
        for n=1:length(classesUnique)
            nb = sum(y_hat == classesUnique(n) & ts_Y == classesUnique(m));
            ConfusionMatrices(m,n,i) = nb;
        end
    end
    prdctIdx = [prdctIdx; samplesIdx(partTS(:,i)==1)'];
    prdctY   = [prdctY; y_hat];
    
end
fprintf('\n');


end

function [X, scale_factors, scale_function] = SVM_scale(X, scale_range)
% SVM_SCALE scales column of X between scale_range

scale_factors = zeros(2,size(X,2));
for k=1:size(X,2)
    max_value = max(X(:,k));
    min_value = min(X(:,k));
    if max_value == min_value
        if max_value == 0
            scale_factors(:,k) = [0; 0];
        else
            scale_factors(:,k) = [0; scale_range(2)/max_value];
        end
    else
        scale_factors(1,k) = scale_range(1) - min_value;
        scale_factors(2,k) = scale_range(2)/(max_value-min_value);
    end
end

X = bsxfun(@plus, X, scale_factors(1,:));
X = bsxfun(@times, X, scale_factors(2,:));

scale_function = @(x) bsxfun(@times, bsxfun(@plus, x, scale_factors(1,:)), scale_factors(2,:));

end


