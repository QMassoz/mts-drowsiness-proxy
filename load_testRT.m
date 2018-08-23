function [M] = load_testRT(subject, test, HDDpath)
% LOAD_TESTRT loads the reaction times matrix M of a PVT test
%
% @param subject: (integer) subject ID
% @param test: (integer) test ID
% @param HDDpath: (string) path to the dir containing "ulg1415" folder (="/Volumes/Elements" by default)
%
% @output M: (matrix Nx2) reaction times matrix which columns are:
%   1st COLUMN = 'question' = appearance time (in ms) of the stimulus
%   2nd COLUMN =  'answer'  = reaction time (in ms) to the stimulus

% Filename
filename = ['rt/' num2str(subject) '-' num2str(test) '.txt'];

% Load .csv
M0 = table2cell(readtable(filename, 'ReadVariableNames', false, 'ReadRowNames', false, 'Delimiter', ' '));
M0 = M0{1};
M = table2cell(readtable(filename, 'Delimiter' , ';', 'HeaderLines',1, 'ReadVariableNames', false, 'ReadRowNames', false));
% Parsing functions
parse = @(x) sscanf(x,'%d-%d-%d_%d.%d.%d.%d')';
dif = @(x,y) sum((y(4:end)-x(4:end)).*[3600000 60000 1000 1]);

% Format the RT matrix
v0 = parse(M0);
RT = zeros(size(M));
for i=1:size(RT,1)
    v1 = parse(M{i,1}); 
    v2 = parse(M{i,2});
    RT(i,1) = dif(v0,v1); 
    RT(i,2) = dif(v1,v2);
end

% Output
M = RT;
end

