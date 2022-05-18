function SCCcount = Tarjan(V,E)
global SCCcount
% Implements Tarjan's algorithm for finding strongly connected components
% (SCC) in the directed graph:
% https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
% G = load('G1.mat');
% V = G.V; % vertices
% E = G.E; % edges
%% Initialization
icurrent = 1;
% index indicates the order at which the vertices are visited using depth
% first search (DFS)
index = zeros(size(V));
% lowlink(v) = the smallest index of a vertex reachable from v including v
% itself
lowlink = zeros(size(V));
% Nodes are placed on a stack in the order in which they are visited. 
% When the depth-first search recursively visits a node v and its descendants, 
% those nodes are not all necessarily popped from the stack when this recursive call returns. 
% The crucial invariant property is that a node remains on the stack after 
% it has been visited if and only if there exists a path in the input graph from 
% it to some node earlier on the stack.
stack = [];
% instack(v) = 1 if v is in stack  
instack = zeros(size(V));

SCCcount = 0;
nV = length(V);
for j = 1 : nV
    if index(j) == 0
        [index,lowlink,icurrent,stack,instack] = strongconnect(V(j),icurrent,E,index,lowlink,stack,instack);
    end
end

end

%% 
function [index,lowlink,icurrent,stack,instack] = strongconnect(v,icurrent,E,index,lowlink,stack,instack)
global SCCcount

index(v) = icurrent;
lowlink(v) = icurrent;
icurrent = icurrent + 1;
stack = [v,stack];
instack(v) = 1;

% Consider successors of v
outneib = find(E(:,1) == v);
if ~isempty(outneib)
    nneib = length(outneib);
    for j = 1 : nneib
        w = E(outneib(j),2); % v --> w
        if index(w) == 0 % w has not been visited
            [index,lowlink,icurrent,stack,instack] = strongconnect(w,icurrent,E,index,lowlink,stack,instack);
            lowlink(v) = min(lowlink(v),lowlink(w));
        else
            if instack(w) == 1 % w has been visited and remained in stack
                % hence w is in the current SCC
                lowlink(v) = min(lowlink(v),index(w));
            end
        end
    end
end

% If v is a root node, pop the stack and generate an SCC
if lowlink(v) == index(v)
    % start a new SCC
    SCCcount = SCCcount + 1;
    SCC = [];
    while 1
        w = stack(1);
        SCC = [SCC;w];
        instack(w) = 0;
        stack(1) = [];
        if w == v
            break;
        end
    end
    fname = sprintf('SCC%d',SCCcount);
    save(fname,'SCC');
end
end
    




