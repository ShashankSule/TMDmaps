function distmesh_TwoWell_FEM()
close all
fsz = 16;
beta = 1;
%% the displayed domain:
XMIN = -2; XMAX = 2;
YMIN = -2; YMAX = 2;
%% the computational domain
bbxmin = -2; bbxmax = 2;
bbymin = -2; bbymax = 2;
% bbxmin = XMIN; bbxmax = XMAX;
% bbymin = -2; bbymax = 4;

bbox = [bbxmin,bbymin;bbxmax,bbymax];
xv = linspace(bbxmin,bbxmax,100); yv = linspace(bbymin,bbymax,100);
[xx,yy] = meshgrid(xv,yv);
v = reshape(my_potential([xx(:),yy(:)]),100,100);
figure;
hold on
h = contour(xv,yv,v,[-100 : 1 : 1]);
h1 = contour(xv,yv,v,[1, 1],'r','LineWidth',4);
h2 = contour(xv,yv,v,[-100,-100],'r','LineWidth',4);
grid
set(gca,'Fontsize',20);
colorbar
xlabel('x','Fontsize',20);
ylabel('y','Fontsize',20);
%%
figure;
hold on;
% get boundary points
a = [-1;0]; % center of A
b = [1;0]; % center of B
r = 0.15;
Vbdry = 1;
t = linspace(0,2*pi,21)';
t(end) = [];
pA = [a(1)+r*cos(t),a(2)+r*sin(t)];
pB = [b(1)+r*cos(t),b(2)+r*sin(t)];
VA = max(my_potential(pA));
VB = max(my_potential(pB));

%%
Vthreshold = 0.5*(Vbdry + max(VA,VB));
hV = contour(xx,yy,v,[Vbdry,Vbdry]);
hV(:,1) = [];
hV = hV';
h = r*(t(2)-t(1));
hV = reparametrization(hV,h);
bdry = [pA;pB;hV];
plot(bdry(:,1),bdry(:,2),'.','Markersize',15);
drawnow;

fd = @(p)ddiff(my_potential(p)-Vbdry,dunion(dcircle(p,a(1),a(2),r),dcircle(p,b(1),b(2),r)));
[pts,tr] = distmesh2d(fd,@huniform,h,bbox,bdry);

%% choose number of mesh refinements
nrefine = 1;
for i = 1 : nrefine 
    [pts,tr] = refine(pts,tr);
end
%% smooth mesh
% [pts,tr] = smoothmesh(pts,tr);
triplot(tr,pts(:,1),pts(:,2),'color','r');
%% Start processing mesh
%% extract boundary nodes and boundary edges
% Find boundary nodes
edges = [tr(:,1),tr(:,2);tr(:,2),tr(:,3);tr(:,1),tr(:,3)];
e0 = edges;
edges = sort(edges,2);
[~,isort2] = sort(edges(:,2),'ascend');
edges = edges(isort2,:);
[~,isort1] = sort(edges(:,1),'ascend');
edges = edges(isort1,:);
Nedges = length(edges);
fprintf('N edges = %i, N points = %i\n',length(edges),length(pts));
% boundary edges are encountered only once as they belong to only one
% triangle
[uedges,ishrink,~] = unique(edges,'rows');
Nis = length(ishrink);
fprintf('N ishrink = %i\n',Nis);
gap = circshift(ishrink,[-1,0]) - ishrink;
if ishrink(end) == Nedges
    gap(end) = 1;
else
    gap(end) = [];
end    
i1 = find(gap == 1);
n1 = length(i1);
i2 = find(gap == 2);
n2 = length(i2);
fprintf('n1 = %i, n2 = %i\n',n1,n2);
ie1 = ishrink(i1);
bedges = edges(ie1,:);
Nbe = size(bedges,1);
ind = (1 : Nedges)';
ie2 = ind;
ie2(ie1) = [];
Nie = length(ie2);
fprintf('Nbe = %i, Nie = %i\n',Nbe,Nie);
figure; hold on
for i = 1 : Nbe
    j = bedges(i,:);
    plot([pts(j(1),1),pts(j(2),1)],[pts(j(1),2),pts(j(2),2)],'Linewidth',2);
end
bb = [bedges(:,1);bedges(:,2)];
ibnodes = unique(bb); % indices of boundary nodes
Nib = length(ibnodes);
bnodes = pts(ibnodes,:); % boundary nodes
plot(bnodes(:,1),bnodes(:,2),'.','Markersize',10);
drawnow;
Nbdry = size(bnodes,1);
% To use Tarjan's algorithm, we need the vertices to be indexed from 1 to
% Nverts
Npts = size(pts,1);
map = zeros(Npts,1);
map(ibnodes) = (1 : Nib)';
E = [map(bedges(:,1)),map(bedges(:,2));map(bedges(:,2)),map(bedges(:,1))];
% Tarjan's algorithm
NSCC = Tarjan([1 : Nib]',E);
col = parula(NSCC + 1);
for i = 1 : NSCC
    fname = sprintf('SCC%d',i);
    SCC = load(fname);
    ind = SCC.SCC;
    plot(pts(ibnodes(ind),1),pts(ibnodes(ind),2),'.','Markersize',10,'color',col(i,:));
    drawnow;
    fprintf('Boundary: SCC %d: %i nodes\n',i,length(ind));
end
daspect([1,1,1])
set(gca,'Fontsize',20);
triplot(tr,pts(:,1),pts(:,2));
% %% find neighbors
% Ned = length(uedges);
% nneib = zeros(Npts,1);
% nei = zeros(Npts);
% for j = 1 : Ned
%     v = uedges(j,1);
%     w = uedges(j,2);
%     nneib(v) = nneib(v) + 1;
%     nneib(w) = nneib(w) + 1;
%     nei(v,nneib(v)) = w;
%     nei(w,nneib(w)) = v;
% end
% maxnneib = max(nneib);
%% find boundary nodes for dirichlet0, dirichlet1, and neumann boundaries
for i = 1 : NSCC
    fname = sprintf('SCC%d',i);
    SCC = load(fname);
    ind = SCC.SCC;

    iinit = ibnodes(ind);
    vrts_init = pts(iinit,:);
    
    % check if this connected component corresponds to the outer boundary    
    if min(my_potential(vrts_init)) > Vthreshold
        neumann = iinit;
    else % this is a Dirichlet boundary
        if max(abs(vrts_init(:,1)-a(1))<r & abs(vrts_init(:,2)-a(2))<r)
            dirichlet0 = iinit;
        else
            dirichlet1 = iinit;
        end
    end
end
figure;
hold on
triplot(tr,pts(:,1),pts(:,2),'color','k');
Nbe = length(bedges);
for j = 1 : Nbe
    plot([pts(bedges(j,1),1),pts(bedges(j,2),1)],[pts(bedges(j,1),2),pts(bedges(j,2),2)],'Linewidth',2);
end
daspect([1,1,1])
%% end processing mesh
% call FEM
[u,u1,A1,b1] = FEM2d(pts,tr,neumann',[dirichlet0',dirichlet1'],dirichlet1', beta);

committor = full(u);
% graphic representation
figure(3)
clf;
trisurf(tr,pts(:,1),pts(:,2),full(u),'facecolor','interp')
view(2)
colorbar

figure(4)
clf;
trisurf(tr,pts(:,1),pts(:,2),full(u),'facecolor','interp')
view(2)
colorbar
axis([XMIN,XMAX,YMIN,YMAX]);
set(gca,'Fontsize',16);
daspect([1,1,1]);
%%
%% Calculate Reactive Current, Rates
[jtri, rate, Z] = reactive_current(full(u), pts, tr, beta);

Npts = size(pts, 1);
Ntri = size(tr, 1);

jx = zeros(Npts, 1);
jy = zeros(Npts, 1);

counts = zeros(Npts,1);
for j = 1:Ntri
    jx(tr(j,:)) = jx(tr(j,:)) + jtri(j, 1);
    jy(tr(j,:)) = jy(tr(j,:)) + jtri(j, 2);
    counts(tr(j,:)) = counts(tr(j,:)) + 1;
end
jx = jx./counts;
jy = jy./counts;

aj = sqrt(jx.^2 + jy.^2);
figure;
trimesh(tr, pts(:, 1), pts(:, 2), full(aj));
% save('../data/FEM_solution.mat','pts','tr', 'jx', 'jy', 'committor', 'Z', 'rate');
save('DistmeshTwowell_1.mat','pts','tr', 'jx', 'jy', 'committor', 'Z', 'rate');
axis square;
end

%% FEM2d
function [u,u1,A1,b1] = FEM2d(coordinates,elements3,neumann,dirichlet,dirichlet1, beta)
FreeNodes=setdiff(1:size(coordinates,1),dirichlet);
A = sparse(size(coordinates,1),size(coordinates,1));
b = sparse(size(coordinates,1),1);

% Assembly
for j = 1:size(elements3,1)
  A(elements3(j,:),elements3(j,:)) = A(elements3(j,:),elements3(j,:)) ...
      + stima3(coordinates(elements3(j,:),:)) * ...
      exp(-beta*mypot(sum(coordinates(elements3(j,:),:),1)/3));
end

% Volume Forces
for j = 1:size(elements3,1)
  b(elements3(j,:)) = 0;  
end

% Neumann conditions
for j = 1 : size(neumann,1)
  b(neumann(j,:))=b(neumann(j,:)) + norm(coordinates(neumann(j,1),:)- ...
      coordinates(neumann(j,2),:)) * myg(sum(coordinates(neumann(j,:),:))/2)/2;
end

% Dirichlet conditions 
u = sparse(size(coordinates,1),1);
u(dirichlet1) = 1;
b = b - A * u;

% Computation of the solution
u(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);

A1 = A(FreeNodes,FreeNodes);
b1 = b(FreeNodes);
u1 = u(FreeNodes);
end


%%
function M = stima3(vertices)
d = size(vertices,2);
G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
M = det([ones(1,d+1);vertices']) * G * G' / prod(1:d);
end

function [jtri, rate, Z] = reactive_current(u, coordinates, elements3, beta)
Ntri = size(elements3,1);
jtri = zeros(Ntri,2);
Z = 0;
rate = 0;
for n = 1:size(elements3,1)
    tr_idx = elements3(n,:);
    vertices = coordinates(tr_idx,:);
    utri = u(tr_idx);
    d = size(vertices,2);
    potential_avg = 0;
    for k=1:3
        potential_avg = potential_avg + my_potential(vertices(k,:));
    end
    potential_avg = potential_avg/3;
    measure = exp(-beta*potential_avg);
    G = [vertices(2:3,1)-vertices(1,1),vertices(2:3,2)-vertices(1,2)]\[utri(2)-utri(1);utri(3)-utri(1)];
    tri_area = 0.5*abs(det([ones(1,3);vertices']));
    Z = Z + tri_area*measure;
    jtri(n,:) = measure *(G)';    
    rate = rate + tri_area*measure*G'*G;
end
rate = rate/(Z*beta);
fprintf('Reaction rate: %d\n',rate);
fprintf('Z: %d\n',Z);
jtri = jtri/(Z*beta);
end

%%
function Vrm = mypot(x)
V = my_potential(x);
Vrm = V;
end

%%
function V = my_potential(x)
mu = [-1, 0 ; 1, 0.]; % gaussian means
c_inv = [2., 0.; 0. 1.];    % gaussian inverse covariance
energy = 10.0;
my_sum = 0;
for i=1:2
    z = (x - mu(i, :));
    my_sum = my_sum + exp(-diag(z*(c_inv*z')));
end
V = -energy*my_sum + x(:, 1).^4 + x(:, 2).^4;
end

%%
function Stress = myg(x)
Stress = zeros(size(x,1),1);
end
%%
function path = reparametrization(path,h)
    %reparametrization
    dp = path - circshift(path,[1,0]);
    dp(1,:) = 0;
    dl = sqrt(sum(dp.^2,2));
    lp = cumsum(dl);
    len = lp(end);
    lp = lp/len; % normalize
    npath = round(len/h);
    g1 = linspace(0,1,npath)';
    path = interp1(lp,path,g1);
end