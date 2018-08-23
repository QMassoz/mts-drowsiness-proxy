function [wM, wTS] = getWindowMatrix(M, startW, endW)
% GETWINDOWDATA returns the sub-matrix wM of matrix M within a time window.
%
% @param M: (matrix N x (1+k)) data matrix which 1st column consists of
%                              timestamps (in ms)
% @param startW: (array or float) start of the time window
%               =[min sec msec] with sec=0 & msec=0 if not specified
% @param stopW: (array or float) end of the time window
%               =[min sec msec] with sec=0 & msec=0 if not specified
%
% @output wM: (array m x k) sub-matrix with timestamps between the time window = [startW endW[
%
% usage: wM = getWindowRT(M,0.5,1) = getWindowRT(M,[0.5 0 0], [0 60]);

% Check arguments
if length(startW) > 3 || length(endW) > 3
    error('startW and stopW must be of length 1, 2 or 3');
end
if isempty(M) || size(M,2)<2
    error('Wrong M format');
end

% Format startW & stopW
startW = reshape(startW,1,[]);
endW  = reshape(endW, 1,[]);
startW = [startW zeros(1,3-length(startW))];
endW  = [endW  zeros(1,3-length(endW))];

% Convert startW & stopW
startW = sum(startW .* [60000 1000 1]);
endW  = sum(endW  .* [60000 1000 1]);

% Return window
idx = (M(:,1) >= startW) & (M(:,1) < endW);
wM = M(idx,2:end);
wTS = M(idx,1);
    
end

