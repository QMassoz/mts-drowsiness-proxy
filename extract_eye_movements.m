function [closing, opening, blinks] = extract_eye_movements(s, d1, thr_opening, thr_closing)
%EXTRACT_EYE_MOVEMENTS Summary of this function goes here
%   Detailed explanation goes here

closing = zeros(size(s));
opening = zeros(size(s));

p_closing = (d1 <= thr_closing);
p_opening = (d1 >=  thr_opening);

% Find events
events_type = [];
events_indices = [];
for i=1:length(s)
    if p_closing(i)==1
        if isempty(events_type) || events_type(end) ~= -1 || events_indices(end,2) ~= i-1
            events_type = [events_type; -1];
            events_indices = [events_indices; i-1 i];
        else
            events_indices(end,2) = i;
        end
    elseif p_opening(i)==1
        if isempty(events_type) || events_type(end) ~= 1 || events_indices(end,2) ~= i-1
            events_type = [events_type; 1];
            events_indices = [events_indices; i-1 i];
        else
            events_indices(end,2) = i;
        end
    end
end

% Check events
valid = zeros(size(events_type));
for i=1:length(events_type)
    valid(i) = is_event_valid(events_type(i), events_indices(i,:), s, d1);
end
events_type(~valid) = [];
events_indices(~valid,:) = [];

% Delete last closing events
while events_type(end) == -1
    events_type(end) = [];
    events_indices(end,:) = [];
end

% Find matching candidates
prev_type = 1;
M = [];
for i=1:length(events_type)
    if events_type(i) == -1 && prev_type == 1
        line = zeros(1,length(events_type));
        line(i) = -1;
        M = [M; line];
    elseif ~isempty(M)
        M(end,i) = events_type(i);
    end
    prev_type = events_type(i);
end



% Find best matching candidates
nPosFunction = @(x,y) (x+1)*x*(y+1)*y/4;
blinks = zeros(size(M,1), 4);
for b=1:size(M,1)
    % Blink candidates
    closing_elist = find(M(b,:) == -1);
    opening_elist = find(M(b,:) ==  1);
    nPos = nPosFunction(length(closing_elist), length(opening_elist));
    
    % Compute a probability for each combination
    closing_candi = zeros(nPos,2);
    opening_candi = zeros(nPos,2);
    prob_candi = zeros(nPos,1);
    
    i=1;
    for c_start=1:length(closing_elist)
        for c_end=1:length(closing_elist)
            if c_start > c_end
                continue;
            end
            for o_start=1:length(opening_elist)
                for o_end=1:length(opening_elist)
                    if o_start > o_end
                        continue;
                    end
                    % Save combination + compute probability
                    closing_candi(i,:) = [events_indices(closing_elist(c_start),1) events_indices(closing_elist(c_end),2)];
                    opening_candi(i,:) = [events_indices(opening_elist(o_start),1) events_indices(opening_elist(o_end),2)];
                    prob_candi(i) = blink_prob(closing_candi(i,:),opening_candi(i,:),s,d1);
                    i = i+1;
                end
            end
        end
    end    
    
    % Keep the highest probability
    [~, pos] = max(prob_candi);
    blinks(b,:) = [closing_candi(pos,1:2) opening_candi(pos,1:2)];
end

for b=1:size(blinks,1)
    closing(blinks(b,1):blinks(b,2)) = 1;
    opening(blinks(b,3):blinks(b,4)) = 1;
end

function valid = is_event_valid(e_type, e_indices, s, d1)
    valid = false;
    vector = s(e_indices(1):e_indices(2));
    
    % opening
    if e_type == 1
        test1 = min(vector) < 0.81;
        test2 = (max(vector) - min(vector)) > 0.13;
        valid = (test1 && test2);
    end
    
    % closing
    if e_type == -1
        test1 = min(vector) < 0.81;
        test2 = (max(vector) - min(vector)) > 0.13;
        valid = (test1 && test2); 
    end
end

function p = blink_prob(c_indices, o_indices, s, d1)
    p = 0;
    closing_edges = s(c_indices);
    d_cl = closing_edges(2) - closing_edges(1);
    opening_edges = s(o_indices);
    d_op = opening_edges(2) - opening_edges(1);
    
    a_dec = (d_cl < -0.1)+0.01;
    a_inc = (d_op > 0.1)+0.01;
    a_sim = exp(-abs(d_cl+d_op));
    a_close = exp(-abs(closing_edges(2)-opening_edges(1)));
    
    p = a_close * a_sim * a_dec * a_inc;
end

end




